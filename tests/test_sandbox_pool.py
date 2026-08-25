"""The sandbox pool: one box per session, capped per user, reaped after the flush.

Runs against a real Postgres; the boxes are fakes with a filesystem each, so two
sessions of one user can only see the same bytes by going through the store.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from collections.abc import Callable

import pytest
import pytest_asyncio

from agent_module.events import UserEvent
from db import pool
from harness_module import runner, store, workspace
from harness_module import session_log as slog
from model_module import client as mc
from tests.dbgate import require_db
from tests.test_workspace import FakeSandbox, _sweeping
from tool_module.sandbox import manager as sandbox_manager

pytestmark = pytest.mark.asyncio

_seeded: list[uuid.UUID] = []


class FakeBoxes:
    """A manager whose boxes are dicts, one per session and never shared."""

    def __init__(self) -> None:
        self.boxes: dict[str, FakeSandbox] = {}
        self.reaped: list[str] = []
        self.paused: list[str] = []
        self.calls: list[str] = []

    def box(self, session_id: str) -> FakeSandbox:
        if session_id not in self.boxes:
            self.boxes[session_id] = _sweeping(FakeSandbox())
        return self.boxes[session_id]

    async def _open(self, session_id: str) -> FakeSandbox:
        """The box, with its handle on the session's slot as the manager records it."""
        fresh = session_id not in self.boxes
        box = self.box(session_id)
        if fresh:
            await pool.execute(
                "UPDATE session_sandboxes SET sandbox_id = $2 WHERE session_id = $1",
                uuid.UUID(session_id),
                f"fake-{session_id[:8]}",
            )
        return box

    async def exec(self, session_id: str, command: str, timeout: int = 120) -> dict:
        self.calls.append(command)
        box = await self._open(session_id)
        # `write <path> <body>` stands in for whatever shell line the model
        # would use to put a file on the disk.
        if command.startswith("write "):
            _, path, body = command.split(" ", 2)
            box.files[path] = body.encode()
            return {"stdout": "", "stderr": "", "exit_code": 0}
        # `die` stands in for a box that vanishes mid-run: the next call to this
        # session builds a fresh one with an empty disk.
        if command == "die":
            self.boxes.pop(session_id, None)
            return {"stdout": "", "stderr": "", "exit_code": 0}
        return await box.exec(session_id, command, timeout)

    async def write_file(self, session_id: str, path: str, content) -> None:
        await (await self._open(session_id)).write_file(session_id, path, content)

    async def read_file(self, session_id: str, path: str) -> str:
        return await (await self._open(session_id)).read_file(session_id, path)

    async def read_bytes(self, session_id: str, path: str) -> bytes:
        return await (await self._open(session_id)).read_bytes(session_id, path)

    async def pause(self, session_id: str) -> None:
        self.paused.append(session_id)
        await sandbox_manager.renew_slot(session_id)

    async def reap(self, session_id: str) -> None:
        self.reaped.append(session_id)
        self.boxes.pop(session_id, None)
        await sandbox_manager.release_slot(session_id)


@pytest_asyncio.fixture(autouse=True)
async def boxes(tmp_path, monkeypatch):
    await require_db()
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    fake = FakeBoxes()
    monkeypatch.setattr(runner.sandbox_manager, "manager", lambda: fake)
    yield fake
    store.use_blobs(None)
    for task in list(runner._reapers) + list(runner._running.values()):
        task.cancel()
    runner._running.clear()
    runner._reapers.clear()
    runner._teardown.clear()
    await asyncio.sleep(0)
    await pool.execute("DELETE FROM sessions WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM files WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM projects WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM users WHERE id = ANY($1::uuid[])", _seeded)
    _seeded.clear()
    await pool.close()


@pytest.fixture
def impatient(monkeypatch):
    """Shrink the wait so a blocked call gives up inside a test."""
    values = {"leases.wait_timeout_s": 0.3, "leases.poll_s": 0.02, "leases.ttl_s": 60}
    monkeypatch.setattr(runner, "_cfg", lambda key, default: values.get(key, default))


@pytest.fixture
def patient(monkeypatch):
    """Poll fast but wait long, for the session that is meant to get its box."""
    values = {"leases.wait_timeout_s": 10, "leases.poll_s": 0.02, "leases.ttl_s": 60}
    monkeypatch.setattr(runner, "_cfg", lambda key, default: values.get(key, default))


@pytest.fixture
def model(monkeypatch):
    """One `run_command`, then a text ending, read off the transcript rather than a queue.

    Two sessions can then run at once without sharing a script. `command` may be
    a function of the messages, so each session issues its own line.
    """

    def arm(command: str | Callable[[list[dict]], str] = "true"):
        def generate(messages, tools=None, **kw):
            if any(m.get("role") == "tool" for m in messages):
                deltas = [mc.TextDelta(text="done"), mc.Finish(reason="stop")]
            else:
                line = command(messages) if callable(command) else command
                deltas = [
                    mc.ToolCallDelta(
                        index=0, id="c1", name="run_command", arguments=json.dumps({"command": line})
                    ),
                    mc.Finish(reason="tool_calls"),
                ]

            async def gen():
                for d in deltas:
                    await asyncio.sleep(0)
                    yield d

            return gen()

        monkeypatch.setattr(mc, "generate", generate)

    return arm


async def _user() -> str:
    user_id = uuid.uuid4()
    await pool.execute("INSERT INTO users (id) VALUES ($1)", user_id)
    _seeded.append(user_id)
    return str(user_id)


async def _project(user_id: str, title: str) -> str:
    """A project linking the folder its title implies, as `POST /projects` makes one."""
    project_id = await pool.fetchval(
        "INSERT INTO projects (user_id, title, slug) VALUES ($1, $2, $3) RETURNING id",
        uuid.UUID(user_id),
        title,
        store.slug(title, "project"),
    )
    await pool.execute(
        "INSERT INTO project_folders (project_id, folder) VALUES ($1, $2)",
        project_id,
        store.slug(title, "project"),
    )
    return str(project_id)


async def _session(
    user_id: str, project_id: str | None = None, status: str = "idle", text: str = "go"
) -> str:
    session_id = str(
        await pool.fetchval(
            "INSERT INTO sessions (user_id, project_id, mode, status) VALUES ($1, $2, 'attended', $3) RETURNING id",
            uuid.UUID(user_id),
            uuid.UUID(project_id) if project_id else None,
            status,
        )
    )
    await slog.append(session_id, UserEvent(text=text))
    return session_id


async def _claim(session_id: str, folder: str, mode: str = "write") -> None:
    await pool.execute(
        "INSERT INTO session_claims (session_id, folder, subpath, mode) VALUES ($1, $2, '/', $3)",
        uuid.UUID(session_id),
        folder,
        mode,
    )


async def _drive(session_id: str) -> None:
    await runner.start(session_id)
    task = runner._running.get(session_id)
    if task is not None:
        await asyncio.wait_for(asyncio.shield(task), timeout=45)


async def _slots(user_id: str) -> int:
    return await pool.fetchval(
        "SELECT count(*) FROM session_sandboxes WHERE user_id = $1", uuid.UUID(user_id)
    )


def _statuses(events) -> list[str]:
    return [e.event.label for e in events if e.event.kind == "status"]


def _file(path: str, content: str) -> store.FileContent:
    return store.FileContent(path=path, content=content.encode())


# --- the box follows the session ----------------------------------------------------


async def test_two_sessions_of_one_user_run_at_once_and_flush_to_their_own_stores(boxes, model, patient):
    """Disjoint claims, no contention, and the store is the only thing the boxes share."""
    user_id = await _user()
    first_project = await _project(user_id, "One")
    second_project = await _project(user_id, "Two")
    await store.commit_tree(user_id, [_file("one/a.txt", "one"), _file("two/b.txt", "two")])
    first = await _session(user_id, first_project, text="one")
    second = await _session(user_id, second_project, text="two")
    await _claim(first, "one")
    await _claim(second, "two")

    def write_into_my_mount(messages: list[dict]) -> str:
        slug = next(m["content"] for m in messages if m.get("role") == "user")
        return f"write {workspace.MOUNT_ROOT}/{slug}/mine.txt hello from {slug}"

    model(write_into_my_mount)

    await asyncio.gather(_drive(first), _drive(second))

    assert {e.path for e in await store.read_tree(user_id, "one")} == {"one/a.txt", "one/mine.txt"}
    assert {e.path for e in await store.read_tree(user_id, "two")} == {"two/b.txt", "two/mine.txt"}
    for folder in ("one", "two"):
        entry = next(
            e for e in await store.read_tree(user_id, folder) if e.path == f"{folder}/mine.txt"
        )
        assert await store.get_blob(entry.content_hash) == f"hello from {folder}".encode()
    assert sorted(boxes.reaped) == sorted([first, second])
    assert await _slots(user_id) == 0


async def test_a_session_holds_one_slot_while_it_runs_and_none_after(boxes, model, patient):
    user_id = await _user()
    project_id = await _project(user_id, "Solo")
    await store.commit_tree(user_id, [_file("solo/a.txt", "1")])
    session_id = await _session(user_id, project_id)

    held: list[int] = []
    original = boxes.exec

    async def watching(session, command, timeout=120):
        if command == "peek":
            held.append(await _slots(user_id))
        return await original(session, command, timeout)

    boxes.exec = watching
    model("peek")

    await _drive(session_id)

    assert held == [1], "the session did not hold exactly one box while it ran"
    assert await _slots(user_id) == 0, "the slot outlived the run"
    assert boxes.reaped == [session_id]


# --- the cap --------------------------------------------------------------------------


async def test_a_session_over_the_cap_waits_and_says_so(boxes, model, impatient, monkeypatch):
    monkeypatch.setattr(sandbox_manager, "max_per_user", lambda: 1)
    user_id = await _user()
    project_id = await _project(user_id, "Busy")
    await store.commit_tree(user_id, [_file("busy/a.txt", "1")])
    holder = await _session(user_id, project_id, status="running")
    assert await sandbox_manager.claim_slot(holder)

    waiter = await _session(user_id, project_id)
    model("true")

    await _drive(waiter)

    events = await slog.get_events(waiter)
    results = [e.event for e in events if e.event.kind == "tool_result"]
    assert _statuses(events) == ["waiting for a computer"]
    assert results[0].error_kind == "timeout"
    assert not results[0].ok
    # What the model is told: nothing happened, so retrying is free.
    assert "never ran" in results[0].content
    # Waiting is not parking: the turn ended normally.
    assert (await pool.fetchrow("SELECT status FROM sessions WHERE id = $1", uuid.UUID(waiter)))["status"] == "idle"


async def test_a_waiting_session_proceeds_when_a_box_frees(boxes, model, patient, monkeypatch):
    monkeypatch.setattr(sandbox_manager, "max_per_user", lambda: 1)
    user_id = await _user()
    project_id = await _project(user_id, "Contended")
    await store.commit_tree(user_id, [_file("contended/a.txt", "1")])
    holder = await _session(user_id, project_id, status="running")
    assert await sandbox_manager.claim_slot(holder)

    waiter = await _session(user_id, project_id)
    model("true")

    # The holder's box goes away once the waiter has been turned down once, so
    # the run really passes through the wait rather than racing a timer.
    real_claim = sandbox_manager.claim_slot
    refusals = []

    async def freeing_claim(session_id: str) -> bool:
        granted = await real_claim(session_id)
        if not granted and not refusals:
            refusals.append(session_id)
            await sandbox_manager.release_slot(holder)
        return granted

    monkeypatch.setattr(sandbox_manager, "claim_slot", freeing_claim)

    await _drive(waiter)

    events = await slog.get_events(waiter)
    results = [e.event for e in events if e.event.kind == "tool_result"]
    assert refusals == [waiter], "the waiter was never turned down, so nothing was tested"
    assert _statuses(events) == ["waiting for a computer"]
    assert results[0].ok, "the session never got a computer"
    assert "true" in boxes.calls


async def test_the_cap_counts_one_users_boxes_only(boxes, monkeypatch):
    monkeypatch.setattr(sandbox_manager, "max_per_user", lambda: 1)
    mine, theirs = await _user(), await _user()
    first = await _session(mine)
    second = await _session(mine)
    other = await _session(theirs)

    assert await sandbox_manager.claim_slot(first)
    assert not await sandbox_manager.claim_slot(second), "the cap counted past itself"
    assert await sandbox_manager.claim_slot(other), "another user's box occupied this user's cap"


async def test_claiming_twice_renews_the_same_slot(boxes, monkeypatch):
    monkeypatch.setattr(sandbox_manager, "max_per_user", lambda: 1)
    user_id = await _user()
    session_id = await _session(user_id)

    assert await sandbox_manager.claim_slot(session_id)
    assert await sandbox_manager.claim_slot(session_id), "a session competed with itself for its own box"
    assert await _slots(user_id) == 1


async def test_concurrent_claims_stop_at_the_cap(boxes, monkeypatch):
    """The count and the insert are one transaction, so the cap is a cap."""
    monkeypatch.setattr(sandbox_manager, "max_per_user", lambda: 2)
    user_id = await _user()
    sessions = [await _session(user_id) for _ in range(5)]

    granted = await asyncio.gather(*(sandbox_manager.claim_slot(s) for s in sessions))

    assert sum(granted) == 2
    assert await _slots(user_id) == 2


# --- the flush comes first ------------------------------------------------------------


async def test_a_box_is_not_reaped_before_its_flush_lands(boxes, model, patient, monkeypatch):
    """The disk is the only copy of an edit until the commit lands."""
    user_id = await _user()
    project_id = await _project(user_id, "Fragile")
    await store.commit_tree(user_id, [_file("fragile/a.txt", "1")])
    session_id = await _session(user_id, project_id)
    model("true")

    async def failing_flush(*a, **kw):
        raise store.StoreError("the store said no")

    monkeypatch.setattr(workspace, "flush", failing_flush)

    # The turn ends on the failure; what matters is what it did not give up.
    with contextlib.suppress(Exception):
        await _drive(session_id)

    assert boxes.reaped == [], "the box was destroyed with the only copy of the edits"
    assert await _slots(user_id) == 1, "the slot was freed while the box still held work"


async def test_a_flush_that_lands_is_followed_by_the_reap(boxes, model, patient, monkeypatch):
    user_id = await _user()
    project_id = await _project(user_id, "Ordered")
    await store.commit_tree(user_id, [_file("ordered/a.txt", "1")])
    session_id = await _session(user_id, project_id)
    model("true")

    order: list[str] = []
    real_flush, real_reap = workspace.flush, boxes.reap

    async def recording(*a, **kw):
        order.append("flush")
        return await real_flush(*a, **kw)

    async def reaping(session):
        order.append("reap")
        await real_reap(session)

    monkeypatch.setattr(workspace, "flush", recording)
    boxes.reap = reaping

    await _drive(session_id)

    assert order == ["flush", "reap"]


async def test_a_box_that_dies_mid_run_takes_neither_the_tree_nor_the_slot(boxes, model, patient):
    """The sentinel refuses the flush, so the session keeps its box and the store keeps its tree."""
    user_id = await _user()
    project_id = await _project(user_id, "Fragile")
    await store.commit_tree(user_id, [_file("fragile/a.txt", "committed")])
    session_id = await _session(user_id, project_id)
    model("die")

    with contextlib.suppress(Exception):
        await _drive(session_id)

    tree = await store.read_tree(user_id)
    assert [e.path for e in tree] == ["fragile/a.txt"]
    assert await store.get_blob(tree[0].content_hash) == b"committed"
    assert boxes.reaped == [], "the box was reaped though its flush never landed"
    assert await _slots(user_id) == 1


# --- the slot lifecycle ---------------------------------------------------------------


@pytest.fixture
def killed(monkeypatch) -> list[str]:
    """Records the boxes a reclaim would have killed, so no SDK is involved."""
    ids: list[str] = []

    async def kill(sandbox_id: str) -> None:
        ids.append(sandbox_id)

    monkeypatch.setattr(sandbox_manager, "_kill", kill)
    return ids


async def _slot_row(session_id: str, sandbox_id: str, ttl_s: int) -> None:
    await pool.execute(
        """
        INSERT INTO session_sandboxes (session_id, user_id, sandbox_id, expires_at)
        SELECT s.id, s.user_id, $2, now() + make_interval(secs => $3) FROM sessions s WHERE s.id = $1
        """,
        uuid.UUID(session_id),
        sandbox_id,
        ttl_s,
    )


async def test_a_box_is_never_booted_without_a_slot(boxes):
    """The row is written before the box, so a box outside the pool cannot exist."""
    user_id = await _user()
    session_id = await _session(user_id)

    with pytest.raises(sandbox_manager.SandboxUnavailable):
        await sandbox_manager.SandboxManager().get_or_create(session_id)


async def test_an_expired_slot_is_reclaimed_by_the_next_claimer(boxes, killed, monkeypatch):
    """A process that died holding capacity does not hold it forever."""
    monkeypatch.setattr(sandbox_manager, "max_per_user", lambda: 1)
    user_id = await _user()
    dead = await _session(user_id, status="running")
    await _slot_row(dead, "sb-dead", -60)
    waiter = await _session(user_id)

    assert await sandbox_manager.claim_slot(waiter)

    assert killed == ["sb-dead"], "the box of the reclaimed slot was left billing"
    assert await _slots(user_id) == 1


async def test_a_finished_sessions_slot_is_reclaimed(boxes, killed, monkeypatch):
    monkeypatch.setattr(sandbox_manager, "max_per_user", lambda: 1)
    user_id = await _user()
    over = await _session(user_id, status="running")
    await _slot_row(over, "sb-over", 900)
    await pool.execute("UPDATE sessions SET status = 'completed' WHERE id = $1", uuid.UUID(over))
    waiter = await _session(user_id)

    assert await sandbox_manager.claim_slot(waiter)
    assert killed == ["sb-over"]


async def test_a_live_slot_is_not_reclaimed(boxes, killed, monkeypatch):
    monkeypatch.setattr(sandbox_manager, "max_per_user", lambda: 1)
    user_id = await _user()
    working = await _session(user_id, status="running")
    await _slot_row(working, "sb-live", 900)
    waiter = await _session(user_id)

    assert not await sandbox_manager.claim_slot(waiter), "a working session lost its box"
    assert killed == []
    assert await _slots(user_id) == 1


async def test_a_call_into_the_box_renews_the_slot(boxes):
    user_id = await _user()
    session_id = await _session(user_id, status="running")
    await _slot_row(session_id, "sb-1", 5)

    assert await sandbox_manager.renew_slot(session_id)

    remaining = await pool.fetchval(
        "SELECT expires_at - now() FROM session_sandboxes WHERE session_id = $1", uuid.UUID(session_id)
    )
    assert remaining.total_seconds() > 60, "the slot was not pushed out"


async def test_renewing_a_slot_that_is_gone_says_so(boxes):
    user_id = await _user()
    session_id = await _session(user_id)

    assert not await sandbox_manager.renew_slot(session_id)


async def test_the_startup_sweep_reclaims_what_a_dead_process_left(boxes, killed):
    user_id = await _user()
    stale = await _session(user_id, status="running")
    finished = await _session(user_id, status="running")
    live = await _session(user_id, status="running")
    await _slot_row(stale, "sb-stale", -60)
    await _slot_row(finished, "sb-finished", 900)
    await _slot_row(live, "sb-live", 900)
    await pool.execute("UPDATE sessions SET status = 'failed' WHERE id = $1", uuid.UUID(finished))

    freed = await sandbox_manager.sweep_slots()

    assert freed == 2
    assert sorted(killed) == ["sb-finished", "sb-stale"]
    assert await _slots(user_id) == 1


async def test_a_park_pauses_the_box_and_keeps_its_slot(boxes, model, patient, monkeypatch):
    """A parked session is not acting, but it is not over: the box waits with it."""
    user_id = await _user()
    project_id = await _project(user_id, "Waiting")
    await store.commit_tree(user_id, [_file("waiting/a.txt", "1")])
    session_id = await _session(user_id, project_id)

    hops = [
        [
            mc.ToolCallDelta(index=0, id="c1", name="run_command", arguments='{"command": "true"}'),
            mc.Finish(reason="tool_calls"),
        ],
        [
            mc.ToolCallDelta(index=0, id="c2", name="ask", arguments='{"question": "which one?"}'),
            mc.Finish(reason="tool_calls"),
        ],
    ]

    def generate(messages, tools=None, **kw):
        deltas = hops.pop(0) if hops else [mc.TextDelta(text="done"), mc.Finish(reason="stop")]

        async def gen():
            for d in deltas:
                await asyncio.sleep(0)
                yield d

        return gen()

    monkeypatch.setattr(mc, "generate", generate)

    await _drive(session_id)

    row = await pool.fetchrow("SELECT status FROM sessions WHERE id = $1", uuid.UUID(session_id))
    assert row["status"] == "awaiting_approval"
    assert boxes.paused == [session_id]
    assert boxes.reaped == [], "a parked session lost its computer"
    assert await _slots(user_id) == 1, "a parked session gave up its slot"
    held = await pool.fetchval(
        "SELECT count(*) FROM resource_leases WHERE session_id = $1", uuid.UUID(session_id)
    )
    assert held == 0, "a parked session is not acting, so it holds no lease"
