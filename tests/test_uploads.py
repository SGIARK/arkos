"""Uploading and browsing files.

Runs against a real Postgres.
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
from harness_module import api, store
from harness_module import session_log as slog
from tests.dbgate import require_db
from tests.runner_tasks import cancel_and_forget

pytestmark = pytest.mark.asyncio

_seeded: list[uuid.UUID] = []


@pytest_asyncio.fixture(autouse=True)
async def blob_store(tmp_path):
    await require_db()
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    yield
    store.use_blobs(None)
    cancel_and_forget()
    await asyncio.sleep(0)
    await pool.execute("DELETE FROM sessions WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM files WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM projects WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM users WHERE id = ANY($1::uuid[])", _seeded)
    _seeded.clear()
    await pool.close()


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=api.app)
    async with AsyncClient(transport=transport, base_url="https://testserver") as c:
        yield c


async def _user() -> str:
    user_id = uuid.uuid4()
    await pool.execute("INSERT INTO users (id) VALUES ($1)", user_id)
    _seeded.append(user_id)
    return str(user_id)


async def _sign_in(client: AsyncClient, user_id: str) -> None:
    token = jwt.encode(
        {"sub": user_id, "aud": "authenticated", "exp": datetime.now(UTC) + timedelta(hours=1)},
        "test-supabase-secret-at-least-32-chars",
        algorithm="HS256",
    )
    assert (await client.post("/auth/session", headers={"Authorization": f"Bearer {token}"})).status_code == 204


async def _project(user_id: str, title: str = "Taxes") -> str:
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


async def _session(user_id: str, project_id: str, status: str = "idle", text: str = "go") -> str:
    session_id = str(
        await pool.fetchval(
            "INSERT INTO sessions (user_id, project_id, mode, status) VALUES ($1, $2, 'attended', $3) RETURNING id",
            uuid.UUID(user_id),
            uuid.UUID(project_id),
            status,
        )
    )
    await slog.append(session_id, UserEvent(text=text))
    return session_id


def _upload(name: str, body: bytes, path: str) -> dict:
    """An upload always names a path, because every file in the store is in a folder."""
    return {"files": {"file": (name, body, "application/octet-stream")}, "data": {"path": path}}


async def _signed(client: AsyncClient) -> str:
    user_id = await _user()
    await _sign_in(client, user_id)
    return user_id


async def test_an_upload_lands_in_the_store_and_lists_immediately(client):
    user_id = await _signed(client)
    project_id = await _project(user_id)

    created = await client.post("/files", **_upload("notes.md", b"hello", path="taxes/notes.md"))
    listing = await client.get("/files")
    linked = await client.get(f"/projects/{project_id}/files")

    assert created.status_code == 201
    body = created.json()
    assert body["name"] == "notes.md"
    assert body["folder"] == "taxes"
    assert body["size"] == 5
    assert uuid.UUID(body["file_id"])
    assert [f["path"] for f in listing.json()] == ["taxes/notes.md"]
    assert [f["path"] for f in linked.json()] == ["taxes/notes.md"]
    entry = (await store.read_tree(user_id))[0]
    assert await store.get_blob(entry.content_hash) == b"hello"


async def test_a_subdirectory_path_is_kept(client):
    user_id = await _signed(client)
    await _project(user_id)

    created = await client.post("/files", **_upload("q3.csv", b"1,2,3", path="taxes/data/2026/q3.csv"))

    assert created.json()["path"] == "taxes/data/2026/q3.csv"
    assert created.json()["name"] == "q3.csv"
    assert created.json()["folder"] == "taxes"
    assert [e.path for e in await store.read_tree(user_id)] == ["taxes/data/2026/q3.csv"]


async def test_re_uploading_a_path_replaces_it(client):
    user_id = await _signed(client)
    await _project(user_id)
    await client.post("/files", **_upload("a.txt", b"first", path="taxes/a.txt"))

    await client.post("/files", **_upload("a.txt", b"second", path="taxes/a.txt"))

    tree = await store.read_tree(user_id)
    assert len(tree) == 1
    assert await store.get_blob(tree[0].content_hash) == b"second"


async def test_an_oversized_upload_is_refused_in_the_standard_shape(client, monkeypatch):
    user_id = await _signed(client)
    await _project(user_id)
    monkeypatch.setattr(api, "_cfg", lambda key, default: 1 if key == "quotas.upload_max_mb" else default)

    response = await client.post("/files", **_upload("big.bin", b"x" * (2 * 1024 * 1024), path="taxes/big.bin"))

    assert response.status_code == 413
    assert response.json() == {
        "code": "file_too_large",
        "message": "1 MB is the limit for one file.",
        "retryable": False,
    }
    assert await store.read_tree(user_id) == [], "a refused upload still wrote a row"


async def test_a_path_that_climbs_out_of_the_store_is_refused(client):
    user_id = await _signed(client)
    await _project(user_id)

    response = await client.post("/files", **_upload("passwd", b"root", path="../../etc/passwd"))

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_request"


async def test_a_file_with_no_folder_is_refused(client):
    """Every file in the store is in a folder: a top-level one would be its own."""
    await _signed(client)

    response = await client.post("/files", **_upload("loose.txt", b"nowhere", path="loose.txt"))

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_request"


async def test_the_store_is_the_callers_own_and_nobody_elses(client):
    """No project id is involved: the store is keyed by user, so scoping is total."""
    theirs_user = await _user()
    await store.put_file(theirs_user, "secret/a.txt", b"peek")
    theirs = await _project(theirs_user, "Secret")
    await _signed(client)

    listing = await client.get("/files")
    linked = await client.get(f"/projects/{theirs}/files")

    assert listing.json() == []
    assert linked.status_code == 404


async def test_an_empty_file_is_content_like_any_other(client):
    """A `.gitkeep` is a file. Zero bytes hash and store like any other content."""
    user_id = await _signed(client)
    await _project(user_id)

    response = await client.post("/files", **_upload(".gitkeep", b"", path="taxes/.gitkeep"))

    assert response.status_code == 201
    assert response.json()["size"] == 0
    entry = (await store.read_tree(user_id))[0]
    assert entry.path == "taxes/.gitkeep"
    assert await store.get_blob(entry.content_hash) == b""


async def test_listing_a_hundred_file_project_boots_nothing(client):
    user_id = await _signed(client)
    project_id = await _project(user_id, "Big")
    await store.commit_tree(user_id, [store.FileContent(path=f"big/f{i:03}.txt", content=b"x") for i in range(100)])

    listing = await client.get(f"/projects/{project_id}/files")
    projects = await client.get("/projects")

    assert len(listing.json()) == 100
    assert projects.status_code == 200
