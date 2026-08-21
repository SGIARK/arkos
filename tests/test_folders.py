"""Folders are the filesystem; projects link to them (11.9).

The model this file pins, stated once because everything below follows from it:
**the store is ONE flat namespace per user, and a folder is a top-level path
segment in it — derived from the files, never a row and never a project.** A
project OWNS no folder; it LINKS folders, as many as it wants, and a folder
exists exactly as long as files exist under it.

What it replaced: the Files tab grouped its dropdowns by project TITLE, so
renaming a project renamed the filesystem's headers — while the schema said
`projects.slug` was the folder and a rename never moved it. The backend
separated folder from project; the frontend fused them; nothing surfaced
inheritance at all.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from db import pool
from harness_module import api, approvals, leases, runner, store, workspace
from tests.dbgate import require_db
from tests.test_api import _supabase_token

pytestmark = pytest.mark.asyncio

_seeded: list[uuid.UUID] = []

PLAN = {
    "goal": "clear the weekend backlog",
    "done_when": "every thread is answered or drafted",
    "steps": ["read the unread threads"],
    "inputs": [],
    "missing": [],
}


@pytest_asyncio.fixture(autouse=True)
async def _db(monkeypatch, tmp_path):
    await require_db()
    store.use_blobs(store.FilesystemBlobs(tmp_path))

    # The loop is out of scope here: what a folder claim MOUNTS is
    # test_workspace's, and what it is RECORDED as is this file's.
    started: list[dict] = []

    async def fake_start(session_id: str, **kw) -> bool:
        started.append({"session_id": session_id, **kw})
        return True

    monkeypatch.setattr(runner, "start", fake_start)
    api.started_here = started
    yield
    store.use_blobs(None)
    await pool.execute("DELETE FROM session_sandboxes WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM sessions WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM files WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM deleted_files WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM projects WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM users WHERE id = ANY($1::uuid[])", _seeded)
    _seeded.clear()
    await pool.close()


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=api.app)
    async with AsyncClient(transport=transport, base_url="https://testserver") as c:
        yield c


async def _signed(client: AsyncClient) -> str:
    user_id = str(uuid.uuid4())
    _seeded.append(uuid.UUID(user_id))
    response = await client.post(
        "/auth/session", headers={"Authorization": f"Bearer {_supabase_token(user_id)}"}
    )
    assert response.status_code == 204
    return user_id


# --- the store is the filesystem ----------------------------------------------------


async def test_a_folder_is_derived_and_no_table_holds_one(client):
    """It exists because a file's path starts with it. There is nothing else to it."""
    await _signed(client)

    await client.post(
        "/files", files={"file": ("a.md", b"x", "text/plain")}, data={"path": "triage/a.md"}
    )

    assert (await client.get("/folders")).json() == [{"name": "triage", "files": 1}]
    assert await pool.fetchval("SELECT to_regclass('public.folders')") is None
    assert store.folder_of("triage/deep/a.md") == "triage"


async def test_the_files_view_headers_are_the_stores_segments_not_project_titles(client):
    """The bug this card exists for: a rename renamed the filesystem's headers."""
    user_id = await _signed(client)
    await store.put_file(user_id, "triage/a.md", b"1")
    made = (await client.post("/projects", json={"title": "inbox triage", "folders": ["triage"]})).json()
    before = (await client.get("/folders")).json()

    renamed = await client.patch(f"/projects/{made['id']}", json={"title": "something else"})

    assert renamed.status_code == 200
    assert (await client.get("/folders")).json() == before == [{"name": "triage", "files": 1}]
    assert renamed.json()["folders"] == ["triage"], "the link did not follow the label"


async def test_the_files_listing_is_the_whole_store_and_not_scoped_to_a_project(client):
    """No project picker: the Files tab is the store, and a project is not a lens on it."""
    user_id = await _signed(client)
    await store.put_file(user_id, "triage/a.md", b"1")
    await store.put_file(user_id, "notes/b.md", b"2")
    made = (await client.post("/projects", json={"title": "one", "folders": ["triage"]})).json()

    everything = (await client.get("/files")).json()
    linked = (await client.get(f"/projects/{made['id']}/files")).json()

    assert [f["path"] for f in everything] == ["notes/b.md", "triage/a.md"]
    # The pane is the same rows, narrowed — same ids, same paths, so clicking one
    # there and finding it here is one file rather than two listings to reconcile.
    assert [f["path"] for f in linked] == ["triage/a.md"]
    assert linked[0]["file_id"] == next(f for f in everything if f["path"] == "triage/a.md")["file_id"]


# --- renaming, which is the one thing that changes a folder's name -------------------


async def test_anything_in_the_files_view_can_be_renamed(client):
    """Files, directories and top-level folders alike — the design renames every row."""
    user_id = await _signed(client)
    for path in ("triage/inbox/a.md", "triage/b.md"):
        await store.put_file(user_id, path, b"x")

    file_row = await client.post("/files/rename", json={"path": "triage/b.md", "name": "c.md"})
    directory = await client.post("/files/rename", json={"path": "triage/inbox", "name": "archive"})
    folder = await client.post("/files/rename", json={"path": "triage", "name": "sorted"})

    assert [r.status_code for r in (file_row, directory, folder)] == [200, 200, 200]
    assert [f["path"] for f in (await client.get("/files")).json()] == [
        "sorted/archive/a.md",
        "sorted/c.md",
    ]
    assert (await client.get("/folders")).json() == [{"name": "sorted", "files": 2}]


async def test_renaming_a_folder_moves_the_link_so_the_project_keeps_its_files(client):
    user_id = await _signed(client)
    await store.put_file(user_id, "triage/a.md", b"1")
    made = (await client.post("/projects", json={"title": "work", "folders": ["triage"]})).json()

    await client.post("/files/rename", json={"path": "triage", "name": "sorted"})

    linked = await client.get(f"/projects/{made['id']}/files")
    assert [f["path"] for f in linked.json()] == ["sorted/a.md"], "the project lost its files"
    assert (await client.get("/projects")).json()[0]["title"] == "work", "the label was not touched"


async def test_a_rename_takes_a_name_and_not_a_path(client):
    user_id = await _signed(client)
    await store.put_file(user_id, "triage/a.md", b"1")

    response = await client.post("/files/rename", json={"path": "triage/a.md", "name": "notes/a.md"})

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_request"
    assert [f["path"] for f in (await client.get("/files")).json()] == ["triage/a.md"]


async def test_renaming_onto_an_occupied_name_is_refused(client):
    user_id = await _signed(client)
    await store.put_file(user_id, "triage/a.md", b"1")
    await store.put_file(user_id, "notes/x.md", b"2")

    clash = await client.post("/files/rename", json={"path": "triage", "name": "notes"})

    assert clash.status_code == 409
    assert clash.json()["code"] == "already_exists"
    assert [f.name for f in await store.folders(user_id)] == ["notes", "triage"]


async def test_renaming_a_folder_a_run_has_mounted_is_refused(client, monkeypatch):
    """A box at `~/store/<old>/` would flush its work back under the old name.

    The session's claims and its manifest are in the runner's memory as well as
    the database, so there is no correcting it from here — and a rename that
    silently resurrected the folder and lost the turn's work is worse than one
    that says "stop the run first".
    """
    user_id = await _signed(client)
    await store.put_file(user_id, "triage/a.md", b"1")
    made = (await client.post("/projects", json={"title": "work", "folders": ["triage"]})).json()
    session_id = (
        await client.post("/sessions", json={"goal": "go", "project_id": made["id"]})
    ).json()["session_id"]
    # A live box: a slot with a nonce, an unexpired lease on it, and a running session.
    await pool.execute(
        "UPDATE sessions SET status = 'running' WHERE id = $1", uuid.UUID(session_id)
    )
    await pool.execute(
        """
        INSERT INTO session_sandboxes (session_id, user_id, sandbox_id, workspace_nonce, expires_at)
        VALUES ($1, $2, 'box-1', 'nonce-1', now() + interval '10 minutes')
        """,
        uuid.UUID(session_id),
        uuid.UUID(user_id),
    )

    refused = await client.post("/files/rename", json={"path": "triage", "name": "sorted"})

    assert refused.status_code == 409
    assert refused.json()["code"] == "folder_busy"
    assert "1 running session" in refused.json()["message"]
    assert [f.name for f in await store.folders(user_id)] == ["triage"], "the folder moved anyway"


async def test_a_file_inside_a_busy_folder_still_renames(client):
    """The refusal is about the MOUNT moving, and a file inside one does not move it.

    A live box is corrected by the same write-through a move uses, which is why
    only the top-level name needs the run stopped.
    """
    user_id = await _signed(client)
    await store.put_file(user_id, "triage/a.md", b"1")
    made = (await client.post("/projects", json={"title": "work", "folders": ["triage"]})).json()
    session_id = (
        await client.post("/sessions", json={"goal": "go", "project_id": made["id"]})
    ).json()["session_id"]
    await pool.execute("UPDATE sessions SET status = 'running' WHERE id = $1", uuid.UUID(session_id))
    await pool.execute(
        """
        INSERT INTO session_sandboxes (session_id, user_id, sandbox_id, workspace_nonce, expires_at)
        VALUES ($1, $2, 'box-1', 'nonce-1', now() + interval '10 minutes')
        """,
        uuid.UUID(session_id),
        uuid.UUID(user_id),
    )

    renamed = await client.post("/files/rename", json={"path": "triage/a.md", "name": "b.md"})

    assert renamed.status_code == 200
    assert [f["path"] for f in (await client.get("/files")).json()] == ["triage/b.md"]


async def test_dragging_a_directory_to_the_edge_makes_it_a_folder(client):
    """The edge is a real destination for a directory, and only for a directory."""
    user_id = await _signed(client)
    await store.put_file(user_id, "triage/inbox/a.md", b"1")
    await store.put_file(user_id, "triage/b.md", b"2")

    out = await client.post("/files/move", json={"from": "triage/inbox", "to": "inbox"})
    refused = await client.post("/files/move", json={"from": "triage/b.md", "to": "b.md"})

    assert out.status_code == 200
    assert refused.status_code == 409
    assert refused.json()["code"] == "move_refused"
    assert (await client.get("/folders")).json() == [
        {"name": "inbox", "files": 1},
        {"name": "triage", "files": 1},
    ]


async def test_a_directory_dragged_out_leaves_the_project_that_linked_it(client):
    """Honest, and the point of dragging it out: it is nobody's work until linked.

    The files are still in the store and still in the Files tab. What they left
    is the project — which links `triage`, not the folder that just appeared
    beside it.
    """
    user_id = await _signed(client)
    await store.put_file(user_id, "triage/inbox/a.md", b"1")
    made = (await client.post("/projects", json={"title": "work", "folders": ["triage"]})).json()

    await client.post("/files/move", json={"from": "triage/inbox", "to": "inbox"})

    assert (await client.get(f"/projects/{made['id']}/files")).json() == []
    assert [f["path"] for f in (await client.get("/files")).json()] == ["inbox/a.md"]
    # And `+ link` is how it comes back, which is why the picker reads the store.
    assert "inbox" in [f["name"] for f in (await client.get("/folders")).json()]


# --- deleting, and taking it back ----------------------------------------------------


async def test_deleting_a_file_keeps_its_bytes_so_undo_is_exact(client):
    """The rows go; the blobs do not, because nothing collects them.

    That is what makes undo a restore rather than a best effort: the same
    content comes back under the same id.
    """
    user_id = await _signed(client)
    stored = await store.put_file(user_id, "triage/a.md", b"the only copy")
    await store.put_file(user_id, "triage/b.md", b"keep me")

    gone = await client.request("DELETE", "/files", json={"path": "triage/a.md"})

    assert gone.status_code == 200
    assert gone.json()["files"] == 1
    assert [f["path"] for f in (await client.get("/files")).json()] == ["triage/b.md"]
    # The bytes never moved.
    assert await store.get_blob(store.sha256(b"the only copy")) == b"the only copy"

    back = await client.post("/files/undo", json={"batch": gone.json()["batch"]})

    assert back.status_code == 200
    listed = (await client.get("/files")).json()
    assert [f["path"] for f in listed] == ["triage/a.md", "triage/b.md"]
    assert next(f for f in listed if f["path"] == "triage/a.md")["file_id"] == stored.id


async def test_deleting_a_directory_takes_everything_under_it(client):
    user_id = await _signed(client)
    for path in ("triage/inbox/a.md", "triage/inbox/deep/b.md", "triage/other.md"):
        await store.put_file(user_id, path, b"x")

    gone = await client.request("DELETE", "/files", json={"path": "triage/inbox"})

    assert gone.json()["files"] == 2
    assert [f["path"] for f in (await client.get("/files")).json()] == ["triage/other.md"]


async def test_deleting_the_last_file_takes_the_folder_and_its_links(client):
    """A folder exists exactly as long as a file exists under it, so both go.

    A project left linking a folder that is not there is the dangling link this
    schema goes out of its way to avoid — so the link travels with the files,
    in the same batch, and comes back with them.
    """
    user_id = await _signed(client)
    await store.put_file(user_id, "triage/a.md", b"1")
    made = (await client.post("/projects", json={"title": "work", "folders": ["triage"]})).json()

    gone = (await client.request("DELETE", "/files", json={"path": "triage/a.md"})).json()

    assert gone["folders"] == ["triage"]
    assert gone["unlinked"] == 1
    assert (await client.get("/folders")).json() == []
    assert await pool.fetchval(
        "SELECT count(*) FROM project_folders WHERE project_id = $1", uuid.UUID(made["id"])
    ) == 0

    back = (await client.post("/files/undo", json={"batch": gone["batch"]})).json()

    assert back["relinked"] == 1
    assert (await client.get("/folders")).json() == [{"name": "triage", "files": 1}]
    assert [f["path"] for f in (await client.get(f"/projects/{made['id']}/files")).json()] == [
        "triage/a.md"
    ]


async def test_undo_refuses_to_overwrite_what_was_put_there_since(client):
    """Whatever is at that path arrived afterwards and is not this batch's to replace."""
    user_id = await _signed(client)
    await store.put_file(user_id, "triage/a.md", b"first")
    gone = (await client.request("DELETE", "/files", json={"path": "triage/a.md"})).json()
    await store.put_file(user_id, "triage/a.md", b"second")

    clash = await client.post("/files/undo", json={"batch": gone["batch"]})

    assert clash.status_code == 409
    assert clash.json()["code"] == "already_exists"
    entry = (await store.read_tree(user_id))[0]
    assert await store.get_blob(entry.content_hash) == b"second", "undo overwrote a newer file"


async def test_undo_restores_the_batch_it_names_and_not_the_newest(client):
    """One click removed one thing; undo puts back what THAT click removed."""
    user_id = await _signed(client)
    await store.put_file(user_id, "triage/a.md", b"1")
    await store.put_file(user_id, "triage/b.md", b"2")
    first = (await client.request("DELETE", "/files", json={"path": "triage/a.md"})).json()
    await client.request("DELETE", "/files", json={"path": "triage/b.md"})

    await client.post("/files/undo", json={"batch": first["batch"]})

    assert [f["path"] for f in (await client.get("/files")).json()] == ["triage/a.md"]


async def test_undoing_twice_finds_nothing_the_second_time(client):
    user_id = await _signed(client)
    await store.put_file(user_id, "triage/a.md", b"1")
    gone = (await client.request("DELETE", "/files", json={"path": "triage/a.md"})).json()

    assert (await client.post("/files/undo", json={"batch": gone["batch"]})).status_code == 200
    again = await client.post("/files/undo", json={"batch": gone["batch"]})

    assert again.status_code == 404


async def test_deleting_in_a_folder_a_run_has_mounted_is_refused(client):
    """A box still holding the file would put it back at its next flush."""
    user_id = await _signed(client)
    await store.put_file(user_id, "triage/a.md", b"1")
    made = (await client.post("/projects", json={"title": "work", "folders": ["triage"]})).json()
    session_id = (
        await client.post("/sessions", json={"goal": "go", "project_id": made["id"]})
    ).json()["session_id"]
    await pool.execute("UPDATE sessions SET status = 'running' WHERE id = $1", uuid.UUID(session_id))
    await pool.execute(
        """
        INSERT INTO session_sandboxes (session_id, user_id, sandbox_id, workspace_nonce, expires_at)
        VALUES ($1, $2, 'box-1', 'nonce-1', now() + interval '10 minutes')
        """,
        uuid.UUID(session_id),
        uuid.UUID(user_id),
    )

    refused = await client.request("DELETE", "/files", json={"path": "triage/a.md"})

    assert refused.status_code == 409
    assert refused.json()["code"] == "folder_busy"
    assert [f["path"] for f in (await client.get("/files")).json()] == ["triage/a.md"]


async def test_another_users_batch_cannot_be_undone(client):
    """Undo is scoped by the same key the store is: the batch is not addressable."""
    theirs = str(uuid.uuid4())
    _seeded.append(uuid.UUID(theirs))
    await pool.execute("INSERT INTO users (id) VALUES ($1)", uuid.UUID(theirs))
    await store.put_file(theirs, "secret/a.md", b"theirs")
    gone = await store.delete_path(theirs, "secret/a.md")
    await _signed(client)

    stolen = await client.post("/files/undo", json={"batch": gone.batch})

    assert stolen.status_code == 404
    assert (await client.get("/files")).json() == []


# --- projects link folders ----------------------------------------------------------


async def test_a_project_linking_two_folders_claims_both_at_spawn(client):
    user_id = await _signed(client)
    await store.put_file(user_id, "triage/a.md", b"1")
    await store.put_file(user_id, "notes/b.md", b"2")

    made = (
        await client.post("/projects", json={"title": "both", "folders": ["triage", "notes"]})
    ).json()
    session_id = (
        await client.post("/sessions", json={"goal": "work", "project_id": made["id"]})
    ).json()["session_id"]

    assert made["folders"] == ["triage", "notes"]
    assert made["files"] == 2
    claims = await workspace.claims_for(session_id)
    assert sorted(c.folder for c in claims) == ["notes", "triage"]
    assert all(c.mode == "write" for c in claims)
    # And the window renders exactly the two, in claim order.
    assert sorted((await client.get(f"/sessions/{session_id}")).json()["folders"]) == [
        "notes",
        "triage",
    ]


async def test_a_project_linking_nothing_gets_one_fresh_empty_folder(client):
    """Not "no files": the none-case reserves a folder and links it like any other."""
    await _signed(client)

    made = (await client.post("/projects", json={"title": "ski trip"})).json()

    assert made["folders"] == ["ski-trip"]
    assert made["files"] == 0, "the sentinel is structure, not content"
    assert (await client.get("/folders")).json() == [{"name": "ski-trip", "files": 0}]
    assert [f["path"] for f in (await client.get("/files")).json()] == ["ski-trip/.keep"]


async def test_a_link_added_later_reaches_the_next_session_not_the_running_one(client):
    """Claims are fixed per session, so this is a fact rather than a surprise."""
    user_id = await _signed(client)
    await store.put_file(user_id, "triage/a.md", b"1")
    await store.put_file(user_id, "notes/b.md", b"2")
    made = (await client.post("/projects", json={"title": "work", "folders": ["triage"]})).json()
    running = (
        await client.post("/sessions", json={"goal": "start", "project_id": made["id"]})
    ).json()["session_id"]

    linked = await client.post(f"/projects/{made['id']}/folders", json={"folder": "notes"})

    # The surface shows it at once...
    assert linked.json()["folders"] == ["triage", "notes"]
    assert await pool.fetchval(
        "SELECT count(*) FROM project_folders WHERE project_id = $1", uuid.UUID(made["id"])
    ) == 2
    # ...the running session's claims are unchanged...
    assert [c.folder for c in await workspace.claims_for(running)] == ["triage"]
    # ...and the next session's include it.
    later = (
        await client.post("/sessions", json={"goal": "again", "project_id": made["id"]})
    ).json()["session_id"]
    assert sorted(c.folder for c in await workspace.claims_for(later)) == ["notes", "triage"]


async def test_deleting_a_project_takes_its_links_and_no_files(client):
    user_id = await _signed(client)
    await store.put_file(user_id, "triage/a.md", b"1")
    made = (await client.post("/projects", json={"title": "work", "folders": ["triage"]})).json()

    await pool.execute("DELETE FROM projects WHERE id = $1", uuid.UUID(made["id"]))

    assert await pool.fetchval(
        "SELECT count(*) FROM project_folders WHERE project_id = $1", uuid.UUID(made["id"])
    ) == 0
    assert [f["path"] for f in (await client.get("/files")).json()] == ["triage/a.md"]
    assert (await client.get("/folders")).json() == [{"name": "triage", "files": 1}]


# --- leases are per folder ----------------------------------------------------------


async def test_two_projects_writing_different_folders_do_not_contend(client):
    user_id = await _signed(client)
    await store.put_file(user_id, "triage/a.md", b"1")
    await store.put_file(user_id, "notes/b.md", b"2")
    first = (await client.post("/projects", json={"title": "one", "folders": ["triage"]})).json()
    second = (await client.post("/projects", json={"title": "two", "folders": ["notes"]})).json()
    a = (await client.post("/sessions", json={"goal": "a", "project_id": first["id"]})).json()
    b = (await client.post("/sessions", json={"goal": "b", "project_id": second["id"]})).json()

    keys_a = workspace.lease_keys(await workspace.claims_for(a["session_id"]))
    keys_b = workspace.lease_keys(await workspace.claims_for(b["session_id"]))

    assert keys_a == [f"folder:{user_id}:triage"]
    assert keys_b == [f"folder:{user_id}:notes"]
    assert await leases.acquire(keys_a[0], a["session_id"], 60)
    assert await leases.acquire(keys_b[0], b["session_id"], 60), "different folders contended"


async def test_two_projects_writing_the_same_folder_serialize(client):
    """A folder may be linked twice over; only one session may be writing it."""
    user_id = await _signed(client)
    await store.put_file(user_id, "shared/a.md", b"1")
    first = (await client.post("/projects", json={"title": "one", "folders": ["shared"]})).json()
    second = (await client.post("/projects", json={"title": "two", "folders": ["shared"]})).json()
    a = (await client.post("/sessions", json={"goal": "a", "project_id": first["id"]})).json()
    b = (await client.post("/sessions", json={"goal": "b", "project_id": second["id"]})).json()

    key = f"folder:{user_id}:shared"
    assert workspace.lease_keys(await workspace.claims_for(b["session_id"])) == [key]
    assert await leases.acquire(key, a["session_id"], 60)
    assert not await leases.acquire(key, b["session_id"], 60), "the same folder did not serialize"


# --- the plan lands in the first linked folder --------------------------------------


async def test_an_approved_plan_lands_in_the_first_linked_folder(client):
    """First LINKED, not first alphabetically: the order they were chosen in."""
    user_id = await _signed(client)
    await store.put_file(user_id, "triage/a.md", b"1")
    await store.put_file(user_id, "archive/b.md", b"2")
    made = (
        await client.post("/projects", json={"title": "work", "folders": ["triage", "archive"]})
    ).json()
    session_id = (
        await client.post("/sessions", json={"goal": "work", "project_id": made["id"]})
    ).json()["session_id"]
    await pool.execute(
        "UPDATE sessions SET status = 'awaiting_approval' WHERE id = $1", uuid.UUID(session_id)
    )
    row = await approvals.create(
        session_id, "c1", "plan", PLAN["goal"], tool_name="propose_plan", tool_args=PLAN
    )

    response = await client.post(f"/approvals/{row.id}/respond", json={"answer": "approve"})

    assert response.status_code == 202
    assert await runner.plan_folder(session_id) == "triage"
    saved = await pool.fetchval(
        "SELECT count(*) FROM files WHERE user_id = $1 AND path = 'triage/plan.md'",
        uuid.UUID(user_id),
    )
    assert saved == 1, "the plan did not land in the first linked folder"


async def test_the_sandbox_sees_the_plan_at_the_mounted_folder_path(client):
    """`~/store/<folder>/plan.md` is what the prompt promises, so materialize must agree."""
    user_id = await _signed(client)
    await store.put_file(user_id, "triage/a.md", b"1")
    made = (await client.post("/projects", json={"title": "work", "folders": ["triage"]})).json()
    session_id = (
        await client.post("/sessions", json={"goal": "work", "project_id": made["id"]})
    ).json()["session_id"]
    await store.put_file(user_id, "triage/plan.md", b"# the plan\n")

    claims = await workspace.claims_for(session_id)

    assert [c.mount for c in claims] == ["/home/user/store/triage"]
    mounted = {f"/home/user/store/{e.path}" for e in await store.read_tree(user_id, claims[0].prefix)}
    assert "/home/user/store/triage/plan.md" in mounted


# --- the home session mints nothing --------------------------------------------------


async def test_a_fresh_account_has_no_project_and_no_folder(client):
    """A project existed only to hold a directory. None is held, so it is unmade."""
    await _signed(client)

    me = (await client.get("/auth/me")).json()

    assert me["home_session_id"], "no home session was made"
    assert (await client.get("/projects")).json() == []
    assert (await client.get("/folders")).json() == []
    assert await pool.fetchval(
        "SELECT project_id FROM sessions WHERE id = $1", uuid.UUID(me["home_session_id"])
    ) is None


async def test_a_session_with_no_project_says_so_rather_than_naming_itself(client):
    """The window's header is the PROJECT's name, and a session without one has none.

    It used to fall back to the session's own title, which is what printed
    "Chat ▸ Chat" in the home chat's header — the same name twice with a crumb
    between them, promising a container that does not exist.
    """
    await _signed(client)
    home = (await client.get("/auth/me")).json()["home_session_id"]
    made = (await client.post("/projects", json={"title": "inbox triage"})).json()
    in_project = (
        await client.post("/sessions", json={"goal": "work", "project_id": made["id"]})
    ).json()["session_id"]

    orphan = (await client.get(f"/sessions/{home}")).json()
    housed = (await client.get(f"/sessions/{in_project}")).json()

    assert orphan["project_id"] is None
    assert orphan["project_title"] is None
    # And a session that HAS one carries the label, so the header does not have
    # to be told it by whatever surface opened the window.
    assert housed["project_title"] == "inbox triage"


async def test_the_home_session_claims_nothing_and_cannot_approve_a_plan(client):
    """It has nowhere durable to write, and saying so beats promising plan.md."""
    await _signed(client)
    home = (await client.get("/auth/me")).json()["home_session_id"]
    await pool.execute("UPDATE sessions SET status = 'awaiting_approval' WHERE id = $1", uuid.UUID(home))
    row = await approvals.create(
        home, "c1", "plan", PLAN["goal"], tool_name="propose_plan", tool_args=PLAN
    )

    response = await client.post(f"/approvals/{row.id}/respond", json={"answer": "approve"})

    assert await workspace.claims_for(home) == []
    assert response.status_code == 409
    assert response.json()["code"] == "no_folder"
    assert [r.id for r in await approvals.open_for(home)] == [row.id], "the plan was spent"


# --- the amendment check -------------------------------------------------------------


# The module mark makes every test async; this one reads a file.
@pytest.mark.asyncio(loop_scope="function")
async def test_contracts_records_the_store_and_the_links():
    """The card's own check: contracts is law, and this card changed the law."""
    contracts = (
        __import__("pathlib").Path(__file__).resolve().parent.parent / "docs" / "contracts.md"
    ).read_text()

    assert "project_folders" in contracts
    assert "`files` `{user_id, path, content_hash, size, mtime}`" in contracts.replace("**", "")
    assert "folder:{user_id}:{name}" in contracts
    assert "~/store/<folder>/" in contracts

