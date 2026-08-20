"""Uploading and browsing files without booting a computer.

The store is where an upload lands; a running session that already has the
project materialized is written through so it reads the file the same turn.
Runs against a real Postgres; the boxes are fakes.
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
from harness_module import api, runner, store, workspace
from harness_module import session_log as slog
from model_module import client as mc
from tests.dbgate import require_db
from tests.test_sandbox_pool import FakeBoxes
from tool_module import registry
from tool_module.envelope import ToolContext

pytestmark = pytest.mark.asyncio

_seeded: list[uuid.UUID] = []


@pytest_asyncio.fixture(autouse=True)
async def boxes(tmp_path, monkeypatch):
    await require_db()
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    fake = FakeBoxes()
    monkeypatch.setattr(runner.sandbox_manager, "manager", lambda: fake)
    monkeypatch.setattr(api.sandbox_manager, "manager", lambda: fake)
    yield fake
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
    return str(
        await pool.fetchval(
            "INSERT INTO projects (user_id, title, slug) VALUES ($1, $2, $3) RETURNING id",
            uuid.UUID(user_id),
            title,
            store.slug(title, "project"),
        )
    )


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


def _upload(name: str, body: bytes, path: str | None = None) -> dict:
    files = {"file": (name, body, "application/octet-stream")}
    return {"files": files, "data": {"path": path}} if path else {"files": files}


async def _signed(client: AsyncClient) -> str:
    user_id = await _user()
    await _sign_in(client, user_id)
    return user_id


# --- the store is where it lands ------------------------------------------------------


async def test_an_upload_lands_in_the_store_and_lists_immediately(client, boxes):
    user_id = await _signed(client)
    project_id = await _project(user_id)

    created = await client.post(f"/projects/{project_id}/files", **_upload("notes.md", b"hello"))
    listing = await client.get(f"/projects/{project_id}/files")

    assert created.status_code == 201
    body = created.json()
    assert body["name"] == "notes.md"
    assert body["size"] == 5
    assert uuid.UUID(body["file_id"])
    assert [f["path"] for f in listing.json()] == ["notes.md"]
    entry = (await store.read_tree(project_id))[0]
    assert await store.get_blob(entry.content_hash) == b"hello"
    assert boxes.boxes == {}, "browsing booted a computer"


async def test_an_upload_to_a_cold_project_is_there_at_the_next_materialize(client, boxes):
    user_id = await _signed(client)
    project_id = await _project(user_id, "Cold")
    await client.post(f"/projects/{project_id}/files", **_upload("report.txt", b"quarterly"))
    session_id = await _session(user_id, project_id)
    assert await runner.sandbox_manager.claim_slot(session_id)

    claim = workspace.Claim(project_id=project_id, slug="cold")
    await workspace.materialize(boxes, session_id, [claim])

    assert boxes.box(session_id).files[f"{workspace.MOUNT_ROOT}/cold/report.txt"] == b"quarterly"


async def test_a_subdirectory_path_is_kept(client):
    user_id = await _signed(client)
    project_id = await _project(user_id)

    created = await client.post(
        f"/projects/{project_id}/files", **_upload("q3.csv", b"1,2,3", path="data/2026/q3.csv")
    )

    assert created.json()["path"] == "data/2026/q3.csv"
    assert created.json()["name"] == "q3.csv"
    assert [e.path for e in await store.read_tree(project_id)] == ["data/2026/q3.csv"]


async def test_re_uploading_a_path_replaces_it(client):
    user_id = await _signed(client)
    project_id = await _project(user_id)
    await client.post(f"/projects/{project_id}/files", **_upload("a.txt", b"first"))

    await client.post(f"/projects/{project_id}/files", **_upload("a.txt", b"second"))

    tree = await store.read_tree(project_id)
    assert len(tree) == 1
    assert await store.get_blob(tree[0].content_hash) == b"second"


# --- what is refused ------------------------------------------------------------------


async def test_an_oversized_upload_is_refused_in_the_standard_shape(client, monkeypatch):
    user_id = await _signed(client)
    project_id = await _project(user_id)
    monkeypatch.setattr(api, "_cfg", lambda key, default: 1 if key == "quotas.upload_max_mb" else default)

    response = await client.post(f"/projects/{project_id}/files", **_upload("big.bin", b"x" * (2 * 1024 * 1024)))

    assert response.status_code == 413
    assert response.json() == {
        "code": "file_too_large",
        "message": "1 MB is the limit for one file.",
        "retryable": False,
    }
    assert await store.read_tree(project_id) == [], "a refused upload still wrote a row"


async def test_a_path_that_climbs_out_of_the_project_is_refused(client):
    user_id = await _signed(client)
    project_id = await _project(user_id)

    response = await client.post(
        f"/projects/{project_id}/files", **_upload("passwd", b"root", path="../../etc/passwd")
    )

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_request"


async def test_another_users_project_reads_as_absent(client):
    theirs = await _project(await _user(), "Secret")
    await _signed(client)

    upload = await client.post(f"/projects/{theirs}/files", **_upload("mine.txt", b"peek"))
    listing = await client.get(f"/projects/{theirs}/files")

    assert upload.status_code == 404
    assert listing.status_code == 404


async def test_an_empty_file_is_content_like_any_other(client):
    """A `.gitkeep` is a file. Zero bytes hash and store like any other content."""
    user_id = await _signed(client)
    project_id = await _project(user_id)

    response = await client.post(f"/projects/{project_id}/files", **_upload(".gitkeep", b""))

    assert response.status_code == 201
    assert response.json()["size"] == 0
    entry = (await store.read_tree(project_id))[0]
    assert entry.path == ".gitkeep"
    assert await store.get_blob(entry.content_hash) == b""


# --- browsing wakes nothing -----------------------------------------------------------


async def test_listing_a_hundred_file_project_boots_nothing(client, boxes):
    user_id = await _signed(client)
    project_id = await _project(user_id, "Big")
    await store.commit_tree(
        project_id, [store.FileContent(path=f"f{i:03}.txt", content=b"x") for i in range(100)]
    )

    listing = await client.get(f"/projects/{project_id}/files")
    projects = await client.get("/projects")

    assert len(listing.json()) == 100
    assert projects.status_code == 200
    assert boxes.boxes == {}, "a listing booted a computer"
    assert boxes.calls == []


# --- and into a running session's box --------------------------------------------------


@pytest.fixture
def patient(monkeypatch):
    values = {"leases.wait_timeout_s": 10, "leases.poll_s": 0.02, "leases.ttl_s": 60}
    monkeypatch.setattr(runner, "_cfg", lambda key, default: values.get(key, default))


async def test_a_running_session_reads_an_upload_the_same_turn(client, boxes, patient, monkeypatch):
    """The write-through: the box already holds the project, so it holds the file too."""
    user_id = await _signed(client)
    project_id = await _project(user_id, "Live")
    await store.commit_tree(project_id, [store.FileContent(path="a.txt", content=b"1")])
    session_id = await _session(user_id, project_id)

    uploaded_then_read: list[bytes] = []
    real_exec = boxes.exec

    async def routed(session, command, timeout=120):
        if command == "upload-and-read":
            # Mid-turn, from outside the session: the box is live and materialized.
            response = await client.post(
                f"/projects/{project_id}/files", **_upload("dropped.txt", b"from the composer")
            )
            assert response.status_code == 201
            uploaded_then_read.append(boxes.box(session).files[f"{workspace.MOUNT_ROOT}/live/dropped.txt"])
            return {"stdout": "read it", "stderr": "", "exit_code": 0}
        return await real_exec(session, command, timeout)

    boxes.exec = routed
    hops = [
        [
            mc.ToolCallDelta(index=0, id="c1", name="run_command", arguments='{"command": "upload-and-read"}'),
            mc.Finish(reason="tool_calls"),
        ]
    ]

    def generate(messages, tools=None, **kw):
        deltas = hops.pop(0) if hops else [mc.TextDelta(text="done"), mc.Finish(reason="stop")]

        async def gen():
            for d in deltas:
                await asyncio.sleep(0)
                yield d

        return gen()

    monkeypatch.setattr(mc, "generate", generate)
    await runner.start(session_id)
    task = runner._running.get(session_id)
    if task is not None:
        await asyncio.wait_for(asyncio.shield(task), timeout=45)

    assert uploaded_then_read == [b"from the composer"]
    # And the session's flush kept it, since the store and the box agree.
    tree = {e.path for e in await store.read_tree(project_id)}
    assert tree == {"a.txt", "dropped.txt"}


async def test_an_upload_to_a_project_no_box_holds_writes_through_to_nothing(client, boxes):
    user_id = await _signed(client)
    project_id = await _project(user_id, "Nobody")
    idle_session = await _session(user_id, project_id)
    assert await runner.sandbox_manager.claim_slot(idle_session)

    await client.post(f"/projects/{project_id}/files", **_upload("a.txt", b"1"))

    assert boxes.boxes == {}, "an upload woke a box that had materialized nothing"


async def test_a_read_claim_that_is_written_through_reports_no_discarded_edits(client, boxes):
    """An uploaded file is in the store, so a read claim is losing nothing by not committing it."""
    user_id = await _signed(client)
    project_id = await _project(user_id, "Reference")
    await store.commit_tree(project_id, [store.FileContent(path="a.txt", content=b"1")])
    session_id = await _session(user_id, project_id)
    assert await runner.sandbox_manager.claim_slot(session_id)
    claim = workspace.Claim(project_id=project_id, slug="reference", mode="read")
    manifest = (await workspace.materialize(boxes, session_id, [claim])).manifest

    await client.post(f"/projects/{project_id}/files", **_upload("added.txt", b"uploaded"))
    boxes.box(session_id).files[f"{workspace.MOUNT_ROOT}/reference/added.txt"] = b"uploaded"
    flushed = await workspace.flush(boxes, session_id, [claim], manifest)

    assert flushed.discarded == (), "an upload was reported to the human as a discarded edit"


async def test_an_upload_over_a_file_the_session_is_editing_fails_the_stale_edit(client, boxes):
    """Last write wins in the box, and the edit that lost cannot corrupt what won.

    `edit_file` matches `old_string` exactly against the file as it is now, so a
    model holding a read from before the upload is refused and has to read again.
    The bytes it would have written are still in the store either way.
    """
    user_id = await _signed(client)
    project_id = await _project(user_id, "Live")
    await store.commit_tree(project_id, [store.FileContent(path="a.txt", content=b"materialized\n")])
    session_id = await _session(user_id, project_id, status="running")
    assert await runner.sandbox_manager.claim_slot(session_id)
    await workspace.materialize(boxes, session_id, [workspace.Claim(project_id=project_id, slug="live")])
    mounted = f"{workspace.MOUNT_ROOT}/live/a.txt"
    ctx = ToolContext(user_id=user_id, session_id=session_id)
    await registry.dispatch("read_file", {"path": mounted}, ctx)

    # The human uploads over the file between the model's read and its edit.
    await client.post(f"/projects/{project_id}/files", **_upload("a.txt", b"from the composer\n"))

    stale = await registry.dispatch(
        "edit_file", {"path": mounted, "old_string": "materialized", "new_string": "edited"}, ctx
    )

    assert stale.ok is False
    assert "old_string does not appear" in stale.content
    assert boxes.box(session_id).files[mounted] == b"from the composer\n", "a stale edit rewrote the upload"

    await registry.dispatch("read_file", {"path": mounted}, ctx)
    fresh = await registry.dispatch(
        "edit_file", {"path": mounted, "old_string": "from the composer", "new_string": "edited"}, ctx
    )

    assert fresh.ok
    assert boxes.box(session_id).files[mounted] == b"edited\n"
