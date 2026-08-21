"""Claims end to end: what a session mounts, what it locks, and what it may write.

A claim names a FOLDER of the user's one flat store (11.9), so the lease is per
folder and two projects writing different folders never contend.

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
    await pool.execute("DELETE FROM files WHERE user_id = ANY($1::uuid[])", _seeded)
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


async def _project(user_id: str, title: str, *folders: str) -> str:
    """A project linking the folders it names, as `POST /projects` makes one.

    Linking, not owning: the folders are the store's and exist because files
    exist under them. With no folders named it links the one its title implies,
    which is the shape `POST /projects` produces for the none-case.
    """
    project_id = await pool.fetchval(
        "INSERT INTO projects (user_id, title, slug) VALUES ($1, $2, $3) RETURNING id",
        uuid.UUID(user_id),
        title,
        store.slug(title, "project"),
    )
    for folder in folders or (store.slug(title, "project"),):
        await pool.execute(
            "INSERT INTO project_folders (project_id, folder) VALUES ($1, $2)", project_id, folder
        )
    return str(project_id)


async def _session(user_id: str, project_id: str | None = None, status: str = "idle") -> str:
    return str(
        await pool.fetchval(
            "INSERT INTO sessions (user_id, project_id, mode, status) VALUES ($1, $2, 'attended', $3) RETURNING id",
            uuid.UUID(user_id),
            uuid.UUID(project_id) if project_id else None,
            status,
        )
    )


async def _claim_row(session_id: str, folder: str, mode: str, subpath: str = "/") -> None:
    await pool.execute(
        "INSERT INTO session_claims (session_id, folder, subpath, mode) VALUES ($1, $2, $3, $4)",
        uuid.UUID(session_id),
        folder,
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


async def test_a_session_without_claims_gets_a_write_claim_on_every_linked_folder(client):
    await _signed(client)

    body = (await client.post("/sessions", json={"goal": "do the thing"})).json()
    claims = await workspace.claims_for(body["session_id"])

    assert len(claims) == 1
    assert claims[0].folder == "do-the-thing", "the none-case folder was not claimed"
    assert claims[0].mode == "write"


async def test_every_linked_folder_is_claimed_at_spawn(client):
    """A project links any number, and a session spawned in it receives them all."""
    user_id = await _signed(client)
    await store.put_file(user_id, "triage/a.txt", b"1")
    await store.put_file(user_id, "notes/b.md", b"2")
    made = (await client.post("/projects", json={"title": "both", "folders": ["triage", "notes"]})).json()

    body = (
        await client.post("/sessions", json={"goal": "work", "project_id": made["id"]})
    ).json()
    claims = await workspace.claims_for(body["session_id"])

    assert sorted(c.folder for c in claims) == ["notes", "triage"]
    assert all(c.mode == "write" for c in claims)


async def test_declared_claims_are_recorded_and_returned(client):
    user_id = await _signed(client)
    await store.put_file(user_id, "reference/docs/a.md", b"1")

    created = await client.post(
        "/sessions",
        json={"goal": "compare them", "claims": [{"folder": "reference", "mode": "read", "subpath": "/docs"}]},
    )
    snapshot = (await client.get(f"/sessions/{created.json()['session_id']}")).json()

    assert snapshot["claims"] == [{"folder": "reference", "subpath": "/docs", "mode": "read"}]
    assert snapshot["folders"] == ["reference"]


async def test_a_claim_on_a_folder_that_is_not_in_this_store_is_refused(client):
    """Another user's folder is not a folder here: the store is keyed by user."""
    theirs_user = await _user()
    await store.put_file(theirs_user, "secret/a.txt", b"1")
    await _signed(client)

    response = await client.post("/sessions", json={"goal": "peek", "claims": [{"folder": "secret"}]})

    assert response.status_code == 404


async def test_a_claim_mode_that_is_neither_read_nor_write_is_refused(client):
    user_id = await _signed(client)
    await store.put_file(user_id, "mine/a.txt", b"1")

    response = await client.post(
        "/sessions", json={"goal": "x", "claims": [{"folder": "mine", "mode": "sideways"}]}
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
    await store.commit_tree(
        user_id, [_file("writable/a.txt", "A original"), _file("readable/b.txt", "B original")]
    )
    session_id = await _session(user_id, writable)
    await _claim_row(session_id, "writable", "write")
    await _claim_row(session_id, "readable", "read")
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

    tree = {e.path: e for e in await store.read_tree(user_id)}

    assert await store.get_blob(tree["writable/a.txt"].content_hash) == b"A edited"
    assert await store.get_blob(tree["readable/b.txt"].content_hash) == b"B original", (
        "a read claim was written"
    )


async def test_nothing_unclaimed_appears_in_the_sandbox(model):
    user_id = await _user()
    claimed = await _project(user_id, "Claimed")
    await store.commit_tree(
        user_id, [_file("claimed/mine.txt", "1"), _file("unclaimed/theirs.txt", "2")]
    )
    session_id = await _session(user_id, claimed)
    await _claim_row(session_id, "claimed", "write")
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
    await store.commit_tree(user_id, [_file("readable/b.txt", "original")])
    session_id = await _session(user_id, readable)
    await _claim_row(session_id, "readable", "read")
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


async def test_a_write_claim_takes_a_lease_on_its_folder(model):
    user_id = await _user()
    project_id = await _project(user_id, "Locked")
    await store.commit_tree(user_id, [_file("locked/a.txt", "1")])
    session_id = await _session(user_id, project_id)
    await _claim_row(session_id, "locked", "write")
    await slog.append(session_id, UserEvent(text="go"))

    held: list[str | None] = []
    model(_call("run_command", '{"command": "check"}'))
    sandbox = runner.sandbox_manager.manager()
    real_exec = sandbox.exec

    async def routed(user_id_, command, timeout=120):
        if command == "check":
            held.append(await leases.holder(f"folder:{user_id}:locked"))
            return {"stdout": "", "stderr": "", "exit_code": 0}
        return await real_exec(user_id_, command, timeout)

    sandbox.exec = routed
    await _drive(session_id)

    assert held == [session_id], "the folder was not leased while the session held it"
    assert await leases.holder(f"folder:{user_id}:locked") is None, "the lease outlived the run"


async def test_a_read_claim_takes_no_folder_lease():
    claims = [
        workspace.Claim(user_id="u", folder="one", mode="read"),
        workspace.Claim(user_id="u", folder="two", mode="write"),
    ]

    assert workspace.lease_keys(claims) == ["folder:u:two"]


async def test_two_projects_writing_different_folders_do_not_wait_on_each_other():
    """The unit of conflict is the FOLDER, so unrelated work runs at once."""
    user_id = await _user()
    first_project = await _project(user_id, "One")
    second_project = await _project(user_id, "Two")
    first = await _session(user_id, first_project, status="running")
    second = await _session(user_id, second_project, status="running")

    assert await leases.acquire(f"folder:{user_id}:one", first, 60)
    assert await leases.acquire(f"folder:{user_id}:two", second, 60)
    # The box is not among what they share, so neither waits for the other's.
    assert await sandbox_manager.claim_slot(first)
    assert await sandbox_manager.claim_slot(second)


async def test_two_projects_writing_the_SAME_folder_still_serialize():
    """Two projects may link one folder; only one of them may be writing it."""
    user_id = await _user()
    mine = await _session(user_id, await _project(user_id, "Mine", "shared"), status="running")
    theirs = await _session(user_id, await _project(user_id, "Theirs", "shared"), status="running")

    assert await leases.acquire(f"folder:{user_id}:shared", mine, 60)
    assert not await leases.acquire(f"folder:{user_id}:shared", theirs, 60)


async def test_a_second_session_claiming_the_same_project_waits_and_says_so(model, monkeypatch):
    user_id = await _user()
    project_id = await _project(user_id, "Contested")
    await store.commit_tree(user_id, [_file("contested/a.txt", "1")])
    holder = await _session(user_id, project_id, status="running")
    await leases.acquire(f"folder:{user_id}:contested", holder, 60)

    waiter = await _session(user_id, project_id)
    await _claim_row(waiter, "contested", "write")
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

    assert any("contested" in label for label in labels)
    assert results[0].error_kind == "timeout"
    budgets = [e for e in events if e.kind == "budget"]
    assert len(budgets) <= 2, "waiting on a lease burned hops"
