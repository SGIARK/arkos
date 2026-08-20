"""Claims end to end: what a session mounts, what it locks, and what it may write.

Runs against a real Postgres; the sandbox and the model are fakes.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from agent_module.events import UserEvent
from db import pool
from harness_module import api, leases, runner, store, workspace
from harness_module import session_log as slog
from model_module import client as mc
from tests.dbgate import require_db
from tests.test_workspace import FakeSandbox, _sweeping
from tool_module.sandbox import manager as sandbox_manager

pytestmark = pytest.mark.asyncio

_seeded: list[uuid.UUID] = []


@pytest_asyncio.fixture(autouse=True)
async def _db(tmp_path, monkeypatch):
    await require_db()
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    sandbox = _sweeping(FakeSandbox())

    async def reap(session_id):
        await runner.sandbox_manager.release_slot(session_id)

    async def pause(session_id):
        await runner.sandbox_manager.renew_slot(session_id)

    sandbox.reap = reap
    sandbox.pause = pause
    monkeypatch.setattr(runner.sandbox_manager, "manager", lambda: sandbox)
    api.sandbox = sandbox
    yield
    store.use_blobs(None)
    for task in list(runner._reapers) + list(runner._running.values()):
        task.cancel()
    runner._running.clear()
    runner._reapers.clear()
    runner._cancelling.clear()
    await asyncio.sleep(0)
    await pool.execute("DELETE FROM sessions WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM projects WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM users WHERE id = ANY($1::uuid[])", _seeded)
    _seeded.clear()
    await pool.close()


@pytest_asyncio.fixture
async def client(monkeypatch):
    """A signed-out client whose session creation records but does not run."""

    async def fake_start(session_id, **kw):
        return True

    monkeypatch.setattr(runner, "start", fake_start)
    transport = ASGITransport(app=api.app)
    async with AsyncClient(transport=transport, base_url="https://testserver") as c:
        yield c


async def _user() -> str:
    user_id = uuid.uuid4()
    await pool.execute("INSERT INTO users (id) VALUES ($1)", user_id)
    _seeded.append(user_id)
    return str(user_id)


async def _project(user_id: str, title: str) -> str:
    """A project whose FOLDER is named after its title, as `POST /projects` makes it.

    The slug is explicit because the column defaults to a unique fallback rather
    than deriving one — these tests assert on mount paths, so the folder has to
    be the one the title implies. `api._new_project` does the same thing with
    collision handling; this is the shape without it.
    """
    return str(
        await pool.fetchval(
            "INSERT INTO projects (user_id, title, slug) VALUES ($1, $2, $3) RETURNING id",
            uuid.UUID(user_id),
            title,
            store.slug(title, "project"),
        )
    )


async def _session(user_id: str, project_id: str | None = None, status: str = "idle") -> str:
    return str(
        await pool.fetchval(
            "INSERT INTO sessions (user_id, project_id, mode, status) VALUES ($1, $2, 'attended', $3) RETURNING id",
            uuid.UUID(user_id),
            uuid.UUID(project_id) if project_id else None,
            status,
        )
    )


async def _claim_row(session_id: str, project_id: str, mode: str, subpath: str = "/") -> None:
    await pool.execute(
        "INSERT INTO session_claims (session_id, project_id, subpath, mode) VALUES ($1, $2, $3, $4)",
        uuid.UUID(session_id),
        uuid.UUID(project_id),
        subpath,
        mode,
    )


async def _sign_in(client: AsyncClient, user_id: str) -> None:
    token = jwt.encode(
        {"sub": user_id, "aud": "authenticated", "exp": datetime.now(UTC) + timedelta(hours=1)},
        "test-supabase-secret-at-least-32-chars",
        algorithm="HS256",
    )
    assert (await client.post("/auth/session", headers={"Authorization": f"Bearer {token}"})).status_code == 204


def _file(path: str, content: str) -> store.FileContent:
    return store.FileContent(path=path, content=content.encode())


@pytest.fixture
def model(monkeypatch):
    def arm(*hops):
        remaining = list(hops)

        def generate(messages, tools=None, **kw):
            deltas = remaining.pop(0) if remaining else [mc.TextDelta(text="done"), mc.Finish(reason="stop")]

            async def gen():
                for d in deltas:
                    await asyncio.sleep(0)
                    yield d

            return gen()

        monkeypatch.setattr(mc, "generate", generate)

    return arm


def _call(name: str, args: str, call_id: str = "c1"):
    return [mc.ToolCallDelta(index=0, id=call_id, name=name, arguments=args), mc.Finish(reason="tool_calls")]


async def _drive(session_id: str) -> None:
    await runner.start(session_id)
    task = runner._running.get(session_id)
    if task is not None:
        await asyncio.wait_for(asyncio.shield(task), timeout=45)


# --- what a session declares -------------------------------------------------------


async def test_a_session_without_claims_gets_a_write_claim_on_its_own_project(client):
    await _signed(client)

    body = (await client.post("/sessions", json={"goal": "do the thing"})).json()
    claims = await workspace.claims_for(body["session_id"])

    assert len(claims) == 1
    assert claims[0].project_id == body["project_id"]
    assert claims[0].mode == "write"


async def test_declared_claims_are_recorded_and_returned(client):
    user_id = await _signed(client)
    other = await _project(user_id, "Reference")

    created = await client.post(
        "/sessions",
        json={"goal": "compare them", "claims": [{"project_id": other, "mode": "read", "subpath": "/docs"}]},
    )
    snapshot = (await client.get(f"/sessions/{created.json()['session_id']}")).json()

    assert snapshot["claims"] == [
        {"project_id": other, "title": "Reference", "subpath": "/docs", "mode": "read"}
    ]


async def test_a_claim_on_another_users_project_is_refused(client):
    theirs_user = await _user()
    theirs = await _project(theirs_user, "Secret")
    await _signed(client)

    response = await client.post("/sessions", json={"goal": "peek", "claims": [{"project_id": theirs}]})

    assert response.status_code == 404


async def test_a_claim_mode_that_is_neither_read_nor_write_is_refused(client):
    user_id = await _signed(client)
    project_id = await _project(user_id, "Mine")

    response = await client.post(
        "/sessions", json={"goal": "x", "claims": [{"project_id": project_id, "mode": "sideways"}]}
    )

    assert response.status_code == 400


async def _signed(client: AsyncClient) -> str:
    user_id = await _user()
    await _sign_in(client, user_id)
    return user_id


# --- what a session mounts ----------------------------------------------------------


async def test_both_claims_mount_and_only_the_write_one_flushes(model):
    user_id = await _user()
    writable = await _project(user_id, "Writable")
    readable = await _project(user_id, "Readable")
    await store.commit_tree(writable, [_file("a.txt", "A original")])
    await store.commit_tree(readable, [_file("b.txt", "B original")])
    session_id = await _session(user_id, writable)
    await _claim_row(session_id, writable, "write")
    await _claim_row(session_id, readable, "read")
    await slog.append(session_id, UserEvent(text="go"))

    model(_call("run_command", '{"command": "true"}'), [mc.TextDelta(text="ok"), mc.Finish(reason="stop")])
    sandbox = runner.sandbox_manager.manager()

    async def edit_then_finish(user_id_, command, timeout=120):
        sandbox.files[f"{workspace.MOUNT_ROOT}/writable/a.txt"] = b"A edited"
        sandbox.files[f"{workspace.MOUNT_ROOT}/readable/b.txt"] = b"B edited"
        return {"stdout": "", "stderr": "", "exit_code": 0}

    real_exec = sandbox.exec

    async def routed(user_id_, command, timeout=120):
        if command == "true":
            return await edit_then_finish(user_id_, command, timeout)
        return await real_exec(user_id_, command, timeout)

    sandbox.exec = routed
    await _drive(session_id)

    write_tree = {e.path: e for e in await store.read_tree(writable)}
    read_tree = {e.path: e for e in await store.read_tree(readable)}

    assert await store.get_blob(write_tree["a.txt"].content_hash) == b"A edited"
    assert await store.get_blob(read_tree["b.txt"].content_hash) == b"B original", "a read claim was written"


async def test_nothing_unclaimed_appears_in_the_sandbox(model):
    user_id = await _user()
    claimed = await _project(user_id, "Claimed")
    unclaimed = await _project(user_id, "Unclaimed")
    await store.commit_tree(claimed, [_file("mine.txt", "1")])
    await store.commit_tree(unclaimed, [_file("theirs.txt", "2")])
    session_id = await _session(user_id, claimed)
    await _claim_row(session_id, claimed, "write")
    await slog.append(session_id, UserEvent(text="go"))
    model(_call("run_command", '{"command": "ls"}'))

    await _drive(session_id)
    sandbox = runner.sandbox_manager.manager()

    assert any("mine.txt" in p for p in sandbox.files)
    assert not any("theirs.txt" in p for p in sandbox.files)
    assert not any("unclaimed" in p for p in sandbox.files)


async def test_the_discarded_edits_are_disclosed_in_the_transcript(model):
    """The person who watched the edits is reading the transcript, not system_events."""
    user_id = await _user()
    readable = await _project(user_id, "Readable")
    await store.commit_tree(readable, [_file("b.txt", "original")])
    session_id = await _session(user_id, readable)
    await _claim_row(session_id, readable, "read")
    await slog.append(session_id, UserEvent(text="go"))

    sandbox = runner.sandbox_manager.manager()
    real_exec = sandbox.exec

    async def routed(user_id_, command, timeout=120):
        if command == "edit":
            sandbox.files[f"{workspace.MOUNT_ROOT}/readable/b.txt"] = b"edited anyway"
            return {"stdout": "", "stderr": "", "exit_code": 0}
        return await real_exec(user_id_, command, timeout)

    sandbox.exec = routed
    model(_call("run_command", '{"command": "edit"}'))
    await _drive(session_id)

    labels = [e.event.label for e in await slog.get_events(session_id) if e.event.kind == "status"]

    assert any("discarded" in label and "b.txt" in label for label in labels)


# --- what a session locks ------------------------------------------------------------


async def test_a_write_claim_takes_a_lease_on_its_project(model):
    user_id = await _user()
    project_id = await _project(user_id, "Locked")
    await store.commit_tree(project_id, [_file("a.txt", "1")])
    session_id = await _session(user_id, project_id)
    await _claim_row(session_id, project_id, "write")
    await slog.append(session_id, UserEvent(text="go"))

    held: list[str | None] = []
    model(_call("run_command", '{"command": "check"}'))
    sandbox = runner.sandbox_manager.manager()
    real_exec = sandbox.exec

    async def routed(user_id_, command, timeout=120):
        if command == "check":
            held.append(await leases.holder(f"project:{project_id}"))
            return {"stdout": "", "stderr": "", "exit_code": 0}
        return await real_exec(user_id_, command, timeout)

    sandbox.exec = routed
    await _drive(session_id)

    assert held == [session_id], "the project was not leased while the session held it"
    assert await leases.holder(f"project:{project_id}") is None, "the lease outlived the run"


async def test_a_read_claim_takes_no_project_lease():
    claims = [
        workspace.Claim(project_id="p1", slug="one", mode="read"),
        workspace.Claim(project_id="p2", slug="two", mode="write"),
    ]

    assert workspace.lease_keys(claims) == ["project:p2"]


async def test_two_sessions_with_disjoint_claims_do_not_wait_on_each_other():
    """The unit of conflict is the path set, so unrelated work runs at once."""
    user_id = await _user()
    first_project = await _project(user_id, "One")
    second_project = await _project(user_id, "Two")
    first = await _session(user_id, first_project, status="running")
    second = await _session(user_id, second_project, status="running")

    assert await leases.acquire(f"project:{first_project}", first, 60)
    assert await leases.acquire(f"project:{second_project}", second, 60)
    # The box is not among what they share, so neither waits for the other's.
    assert await sandbox_manager.claim_slot(first)
    assert await sandbox_manager.claim_slot(second)


async def test_a_second_session_claiming_the_same_project_waits_and_says_so(model, monkeypatch):
    user_id = await _user()
    project_id = await _project(user_id, "Contested")
    await store.commit_tree(project_id, [_file("a.txt", "1")])
    holder = await _session(user_id, project_id, status="running")
    await leases.acquire(f"project:{project_id}", holder, 60)

    waiter = await _session(user_id, project_id)
    await _claim_row(waiter, project_id, "write")
    await slog.append(waiter, UserEvent(text="go"))
    monkeypatch.setattr(
        runner,
        "_cfg",
        lambda key, default: {"leases.wait_timeout_s": 0.2, "leases.poll_s": 0.05}.get(key, default),
    )
    model(_call("run_command", '{"command": "ls"}'), [mc.TextDelta(text="gave up"), mc.Finish(reason="stop")])

    await _drive(waiter)
    events = [e.event for e in await slog.get_events(waiter)]
    labels = [e.label for e in events if e.kind == "status"]
    results = [e for e in events if e.kind == "tool_result"]

    assert any("Contested" in label or "contested" in label for label in labels)
    assert results[0].error_kind == "timeout"
    budgets = [e for e in events if e.kind == "budget"]
    assert len(budgets) <= 2, "waiting on a lease burned hops"
