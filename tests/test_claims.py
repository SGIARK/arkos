"""Claims end to end: what a session is given, and what each claim locks."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from db import pool
from harness_module import api, leases, runner, store, workspace
from tests.dbgate import require_db

pytestmark = pytest.mark.asyncio

_seeded: list[uuid.UUID] = []


@pytest_asyncio.fixture(autouse=True)
async def _db(tmp_path):
    await require_db()
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    yield
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
    """A project linking the folders it names, as `POST /projects` makes one."""
    project_id = await pool.fetchval(
        "INSERT INTO projects (user_id, title, slug) VALUES ($1, $2, $3) RETURNING id",
        uuid.UUID(user_id),
        title,
        store.slug(title, "project"),
    )
    for folder in folders or (store.slug(title, "project"),):
        await pool.execute("INSERT INTO project_folders (project_id, folder) VALUES ($1, $2)", project_id, folder)
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

    body = (await client.post("/sessions", json={"goal": "work", "project_id": made["id"]})).json()
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

    response = await client.post("/sessions", json={"goal": "x", "claims": [{"folder": "mine", "mode": "sideways"}]})

    assert response.status_code == 400


async def _signed(client: AsyncClient) -> str:
    user_id = await _user()
    await _sign_in(client, user_id)
    return user_id


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


async def test_two_projects_writing_the_SAME_folder_still_serialize():
    """Two projects may link one folder; only one of them may be writing it."""
    user_id = await _user()
    mine = await _session(user_id, await _project(user_id, "Mine", "shared"), status="running")
    theirs = await _session(user_id, await _project(user_id, "Theirs", "shared"), status="running")

    assert await leases.acquire(f"folder:{user_id}:shared", mine, 60)
    assert not await leases.acquire(f"folder:{user_id}:shared", theirs, 60)
