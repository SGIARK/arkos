"""Claims on the browser and on projects: one session at a time, and given up when it stops.

The sandbox is not among them — a box belongs to one session, so it is capacity
rather than a lease (see test_sandbox_pool.py).

Runs against a real Postgres with migration 0 applied.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
import pytest_asyncio

from agent_module.events import UserEvent
from db import pool
from harness_module import leases, runner
from harness_module import session_log as slog
from model_module import client as mc
from tests.dbgate import require_db
from tool_module.sandbox import manager as sandbox_manager
from tool_module.sandbox import tools as sandbox_tools

pytestmark = pytest.mark.asyncio

_seeded: list[uuid.UUID] = []


@pytest_asyncio.fixture(autouse=True)
async def _db():
    await require_db()
    yield
    for task in list(runner._reapers) + list(runner._running.values()):
        task.cancel()
    runner._running.clear()
    runner._reapers.clear()
    runner._cancelling.clear()
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
    project = f"project:{uuid.uuid4()}"
    await leases.acquire(browser, session_id, 60)
    await leases.acquire(project, session_id, 60)

    assert await leases.release_all(session_id) == 2
    assert await leases.holder(browser) is None
    assert await leases.holder(project) is None


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

        async def exec(self, session_id, command, timeout=120):
            self.commands.append(command)
            return {"stdout": "ok", "stderr": "", "exit_code": 0}

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
    held = await pool.fetchval(
        "SELECT count(*) FROM resource_leases WHERE session_id = $1", uuid.UUID(session_id)
    )
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
    held = await pool.fetchval(
        "SELECT count(*) FROM resource_leases WHERE session_id = $1", uuid.UUID(session_id)
    )
    assert held == 0
