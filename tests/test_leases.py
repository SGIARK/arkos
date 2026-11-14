"""Claims on the browser and on projects: one session at a time, and given up when it stops.

Runs against a real Postgres with migration 0 applied.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid

import pytest
import pytest_asyncio

from agent_module.events import UserEvent
from db import pool
from harness_module import leases, runner
from harness_module import session_log as slog
from model_module import client as mc
from tests.dbgate import require_db
from tests.runner_tasks import cancel_and_forget
from tool_module.envelope import ToolUnavailable
from tool_module.sandbox import manager as sandbox_manager
from tool_module.sandbox import tools as sandbox_tools

pytestmark = pytest.mark.asyncio

_seeded: list[uuid.UUID] = []


@pytest_asyncio.fixture(autouse=True)
async def _db():
    await require_db()
    yield
    cancel_and_forget()
    await asyncio.sleep(0)
    await pool.execute("DELETE FROM sessions WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM users WHERE id = ANY($1::uuid[])", _seeded)
    _seeded.clear()
    await pool.close()


async def _session(mode: str = "unattended", status: str = "running") -> str:
    user_id = uuid.uuid4()
    await pool.execute("INSERT INTO users (id) VALUES ($1)", user_id)
    _seeded.append(user_id)
    return str(
        await pool.fetchval(
            "INSERT INTO sessions (user_id, mode, status) VALUES ($1, $2, $3) RETURNING id",
            user_id,
            mode,
            status,
        )
    )


def _key() -> str:
    return leases.key("browser", str(uuid.uuid4()))


async def test_one_session_holds_a_resource_at_a_time():
    first, second = await _session(), await _session()
    resource = _key()

    assert await leases.acquire(resource, first, 60)
    assert not await leases.acquire(resource, second, 60)
    assert await leases.holder(resource) == first


async def test_the_holder_may_renew_its_own_lease():
    session_id = await _session()
    resource = _key()

    assert await leases.acquire(resource, session_id, 60)
    assert await leases.acquire(resource, session_id, 60)


async def test_an_expired_lease_is_taken_by_the_next_session():
    """A process that dies holding a lease does not hold the resource forever."""
    first, second = await _session(), await _session()
    resource = _key()

    await leases.acquire(resource, first, -1)

    assert await leases.holder(resource) is None
    assert await leases.acquire(resource, second, 60)
    assert await leases.holder(resource) == second


async def test_releasing_frees_the_resource():
    first, second = await _session(), await _session()
    resource = _key()
    await leases.acquire(resource, first, 60)

    assert await leases.release(resource, first)
    assert await leases.acquire(resource, second, 60)


async def test_one_session_cannot_release_anothers_lease():
    first, second = await _session(), await _session()
    resource = _key()
    await leases.acquire(resource, first, 60)

    assert not await leases.release(resource, second)
    assert await leases.holder(resource) == first


async def test_release_all_frees_every_resource_a_session_holds():
    session_id = await _session()
    browser = leases.key("browser", str(uuid.uuid4()))
    folder = f"folder:{uuid.uuid4()}:triage"
    await leases.acquire(browser, session_id, 60)
    await leases.acquire(folder, session_id, 60)

    assert await leases.release_all(session_id) == 2
    assert await leases.holder(browser) is None
    assert await leases.holder(folder) is None


async def test_a_lease_is_dropped_with_its_session():
    """The row has ON DELETE CASCADE, so a deleted session leaves no claim behind."""
    session_id = await _session()
    resource = _key()
    await leases.acquire(resource, session_id, 60)

    await pool.execute("DELETE FROM sessions WHERE id = $1", uuid.UUID(session_id))

    assert await leases.holder(resource) is None


async def test_concurrent_claims_produce_one_holder():
    sessions = [await _session() for _ in range(4)]
    resource = _key()

    results = await asyncio.gather(*(leases.acquire(resource, s, 60) for s in sessions))

    assert sum(results) == 1, "exactly one session may take the lease"


# --- through a running session ---------------------------------------------------


@pytest.fixture
def sandbox(monkeypatch):
    """A sandbox that records calls, so no e2b is involved."""

    class Fake:
        def __init__(self):
            self.commands = []
            self.files = {}

        async def exec(self, session_id, command, timeout=120):
            self.commands.append(command)
            return {"stdout": "ok", "stderr": "", "exit_code": 0}

        async def write_file(self, session_id, path, content):
            self.files[path] = content

        async def read_file(self, session_id, path):
            return self.files[path]

        async def pause(self, session_id):
            await sandbox_manager.renew_slot(session_id)

        async def reap(self, session_id):
            await sandbox_manager.release_slot(session_id)

    fake = Fake()
    monkeypatch.setattr(sandbox_tools.sandbox_manager, "manager", lambda: fake)
    return fake


@pytest.fixture
def impatient(monkeypatch):
    """Shrink the contended wait so a blocked call gives up inside a test."""
    values = {"leases.wait_timeout_s": 0.2, "leases.poll_s": 0.05, "leases.ttl_s": 60}
    monkeypatch.setattr(runner, "_cfg", lambda key, default: values.get(key, default))


def _text(*chunks):
    return [mc.TextDelta(text=c) for c in chunks] + [mc.Finish(reason="stop")]


def _call(name, args="{}", *, id="c1"):
    return [mc.ToolCallDelta(index=0, id=id, name=name, arguments=args), mc.Finish(reason="tool_calls")]


@pytest.fixture
def model(monkeypatch):
    def arm(*hops):
        remaining = list(hops)

        def generate(messages, tools=None, **kw):
            deltas = remaining.pop(0) if remaining else _text("done")

            async def gen():
                for d in deltas:
                    await asyncio.sleep(0)
                    yield d

            return gen()

        monkeypatch.setattr(mc, "generate", generate)

    return arm


async def _drive(session_id: str) -> None:
    await runner.start(session_id)
    task = runner._running.get(session_id)
    if task is not None:
        await asyncio.wait_for(asyncio.shield(task), timeout=45)


async def test_a_session_leaves_no_lease_behind(sandbox, model, impatient):
    session_id = await _session(mode="attended", status="idle")
    await slog.append(session_id, UserEvent(text="go"))
    model(_call("run_command", '{"command": "ls"}'), _text("done"))

    await _drive(session_id)

    assert sandbox.commands == ["ls"]
    held = await pool.fetchval("SELECT count(*) FROM resource_leases WHERE session_id = $1", uuid.UUID(session_id))
    assert held == 0, "a lease outlived the run"


async def test_the_sandbox_is_never_leased(sandbox, model, impatient):
    """A box belongs to one session, so there is nothing to serialize."""
    session_id = await _session(mode="attended", status="idle")
    await slog.append(session_id, UserEvent(text="go"))
    model(_call("run_command", '{"command": "ls"}'), _text("done"))

    await _drive(session_id)

    keys = [r["resource_key"] for r in await pool.fetch("SELECT resource_key FROM resource_leases")]
    assert not [k for k in keys if k.startswith("sandbox:")]


async def test_a_park_gives_the_leases_back(sandbox, model, impatient):
    """A parked session is not acting, so it holds nothing."""
    session_id = await _session(mode="attended", status="idle")
    await slog.append(session_id, UserEvent(text="go"))
    model(
        _call("run_command", '{"command": "ls"}'),
        _call("ask", '{"question": "which one?"}', id="c2"),
    )

    await _drive(session_id)

    row = await pool.fetchrow("SELECT status FROM sessions WHERE id = $1", uuid.UUID(session_id))
    assert row["status"] == "awaiting_approval"
    held = await pool.fetchval("SELECT count(*) FROM resource_leases WHERE session_id = $1", uuid.UUID(session_id))
    assert held == 0
    slot = await pool.fetchval("SELECT count(*) FROM session_sandboxes WHERE session_id = $1", uuid.UUID(session_id))
    assert slot == 1, "the box is hibernated on a park, not given up"


# --- giving a lease back before the session ends ----------------------------------


async def _sink(session_id: str) -> runner._Sink:
    return runner._Sink(await runner.load(session_id))


async def _stop(sink: runner._Sink) -> None:
    sink._writer.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await sink._writer


async def test_an_early_release_frees_the_resource_for_another_session(impatient):
    """A lease given back mid-session goes to the next session at once, not at session end."""
    first = await _sink(await _session())
    second = await _sink(await _session(mode="attended"))
    # One user's two sessions: the lease key is per user.
    await pool.execute(
        "UPDATE sessions SET user_id = $1 WHERE id = $2",
        uuid.UUID(first.session.user_id),
        uuid.UUID(second.session.id),
    )
    second.session.user_id = first.session.user_id
    try:
        await first._lease("printer")
        with pytest.raises(ToolUnavailable):
            await second._lease("printer")

        await first._unlease("printer")

        await second._lease("printer")
        assert await leases.holder(leases.key("printer", first.session.user_id)) == second.session.id
        assert first._leases == set()
    finally:
        await _stop(first)
        await _stop(second)


async def test_a_failed_release_leaves_the_session_end_release_armed(impatient, monkeypatch):
    """The key stays tracked, so `_release_leases` still runs `release_all`."""
    sink = await _sink(await _session())

    async def broken(resource_key, session_id):
        raise ConnectionError("db went away")

    try:
        await sink._lease("printer")
        monkeypatch.setattr(runner.leases, "release", broken)

        await sink._unlease("printer")  # does not raise

        assert sink._leases == {leases.key("printer", sink.session.user_id)}
        await sink._release_leases()
        held = await pool.fetchval(
            "SELECT count(*) FROM resource_leases WHERE session_id = $1", uuid.UUID(sink.session.id)
        )
        assert held == 0
    finally:
        await _stop(sink)


async def test_giving_back_what_was_never_taken_is_a_no_op(impatient):
    sink = await _sink(await _session())
    try:
        await sink._unlease("printer")
        await sink._unlease("sandbox")
        assert sink._leases == set()
    finally:
        await _stop(sink)


# --- browser slots ----------------------------------------------------------------


@pytest.fixture
def capped(monkeypatch):
    """A small browser cap and a short wait, so a full pool gives up inside a test."""
    values = {
        "leases.wait_timeout_s": 0.3,
        "leases.poll_s": 0.05,
        "leases.ttl_s": 60,
        "browser.max_concurrent_per_user": 2,
    }
    monkeypatch.setattr(runner, "_cfg", lambda key, default: values.get(key, default))
    return values


async def _sinks_of_one_user(count: int) -> list[runner._Sink]:
    """`count` sessions that all belong to one user."""
    first = await _session()
    user_id = (await runner.load(first)).user_id
    ids = [first]
    for _ in range(count - 1):
        ids.append(
            str(
                await pool.fetchval(
                    "INSERT INTO sessions (user_id, mode, status) VALUES ($1, 'unattended', 'running') RETURNING id",
                    uuid.UUID(user_id),
                )
            )
        )
    return [await _sink(i) for i in ids]


async def test_one_users_sessions_share_the_browser_up_to_the_cap(capped):
    first, second, third = await _sinks_of_one_user(3)
    try:
        await first._lease("browser")
        await second._lease("browser")  # does not wait for the first

        assert first._slots["browser"] != second._slots["browser"]
        with pytest.raises(ToolUnavailable, match="already runs 2 at once"):
            await third._lease("browser")
    finally:
        for sink in (first, second, third):
            await _stop(sink)


async def test_a_freed_slot_goes_to_the_waiting_session(capped):
    first, second, third = await _sinks_of_one_user(3)
    capped["leases.wait_timeout_s"] = 5
    try:
        await first._lease("browser")
        await second._lease("browser")
        waiting = asyncio.create_task(third._lease("browser"))
        await asyncio.sleep(0.2)
        assert not waiting.done(), "the third session should wait while the pool is full"

        await first._unlease("browser")

        await asyncio.wait_for(waiting, timeout=2)
        assert third._slots["browser"] == leases.slot_keys("browser", third.session.user_id, 2)[0]
    finally:
        for sink in (first, second, third):
            await _stop(sink)


async def test_parallel_claims_at_the_cap_grant_exactly_the_cap(capped):
    sinks = await _sinks_of_one_user(5)
    try:
        results = await asyncio.gather(*(s._lease("browser") for s in sinks), return_exceptions=True)

        granted = [r for r in results if r is None]
        refused = [r for r in results if isinstance(r, ToolUnavailable)]
        assert (len(granted), len(refused)) == (2, 3)
        rows = await pool.fetchval(
            "SELECT count(*) FROM resource_leases WHERE resource_key LIKE $1",
            f"browser:{sinks[0].session.user_id}:%",
        )
        assert rows == 2
    finally:
        for sink in sinks:
            await _stop(sink)


async def test_other_users_are_not_counted_against_the_cap(capped):
    mine = await _sinks_of_one_user(2)
    theirs = await _sink(await _session())
    try:
        for sink in mine:
            await sink._lease("browser")
        await theirs._lease("browser")  # a different user's pool
    finally:
        for sink in (*mine, theirs):
            await _stop(sink)


async def test_a_cap_below_one_still_lets_one_call_through(capped):
    capped["browser.max_concurrent_per_user"] = 0
    first, second = await _sinks_of_one_user(2)
    try:
        await first._lease("browser")
        with pytest.raises(ToolUnavailable, match="already runs 1 at once"):
            await second._lease("browser")
    finally:
        for sink in (first, second):
            await _stop(sink)


async def test_the_session_end_release_frees_the_browser_slot(capped):
    (sink,) = await _sinks_of_one_user(1)
    try:
        await sink._lease("browser")
        await sink._release_leases()
        held = await pool.fetchval(
            "SELECT count(*) FROM resource_leases WHERE session_id = $1", uuid.UUID(sink.session.id)
        )
        assert held == 0
        assert sink._slots == {}
    finally:
        await _stop(sink)
