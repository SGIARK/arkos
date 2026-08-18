"""Snapshots: a saved copy of a project's tree, and the way back to it.

Cheap because blobs are immutable — a snapshot copies rows and a restore points
the tree back at bytes that never moved. Runs against a real Postgres.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from db import pool
from harness_module import store, workspace
from tests.dbgate import require_db
from tests.test_workspace import FakeSandbox, _sweeping
from tool_module.sandbox import manager as sandbox_manager

pytestmark = pytest.mark.asyncio

_seeded: list[uuid.UUID] = []


@pytest_asyncio.fixture(autouse=True)
async def blobs(tmp_path):
    await require_db()
    backend = store.FilesystemBlobs(tmp_path)
    store.use_blobs(backend)
    yield backend
    store.use_blobs(None)
    await pool.execute("DELETE FROM sessions WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM projects WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM users WHERE id = ANY($1::uuid[])", _seeded)
    _seeded.clear()
    await pool.close()


async def _project(title: str = "Taxes") -> str:
    user_id = uuid.uuid4()
    await pool.execute("INSERT INTO users (id) VALUES ($1)", user_id)
    _seeded.append(user_id)
    return str(
        await pool.fetchval(
            "INSERT INTO projects (user_id, title) VALUES ($1, $2) RETURNING id", user_id, title
        )
    )


def _file(path: str, content: str) -> store.FileContent:
    return store.FileContent(path=path, content=content.encode())


async def _tree(project_id: str) -> dict[str, str]:
    return {e.path: e.content_hash for e in await store.read_tree(project_id)}


# --- taking one ---------------------------------------------------------------------


async def test_a_snapshot_records_the_tree_as_it_stands():
    project_id = await _project()
    await store.commit_tree(project_id, [_file("a.txt", "one"), _file("b.txt", "two")])

    snapshot_id = await store.snapshot_project(project_id, "before the edit")

    listed = await store.list_snapshots(project_id)
    assert [(s.id, s.label, s.files) for s in listed] == [(snapshot_id, "before the edit", 2)]


async def test_a_snapshot_copies_rows_and_not_bytes(blobs):
    project_id = await _project()
    await store.commit_tree(project_id, [_file("a.txt", "one")])
    before = sum(1 for _ in blobs.root.rglob("*") if _.is_file())

    await store.snapshot_project(project_id)

    assert sum(1 for _ in blobs.root.rglob("*") if _.is_file()) == before


async def test_an_empty_project_snapshots_to_an_empty_tree():
    """A state worth being able to return to: the project before anything was in it."""
    project_id = await _project()

    snapshot_id = await store.snapshot_project(project_id, "empty")
    await store.commit_tree(project_id, [_file("added.txt", "later")])
    await store.restore_snapshot(snapshot_id)

    assert await store.read_tree(project_id) == []


# --- going back to one --------------------------------------------------------------


async def test_a_restore_puts_the_tree_back_and_materializes_the_old_bytes():
    project_id = await _project()
    await store.commit_tree(project_id, [_file("a.txt", "original"), _file("gone-later.txt", "keep me")])
    snapshot_id = await store.snapshot_project(project_id, "good state")

    await store.commit_tree(project_id, [_file("a.txt", "edited"), _file("new.txt", "added after")])
    restored = await store.restore_snapshot(snapshot_id)

    assert {e.path for e in restored} == {"a.txt", "gone-later.txt"}
    assert "new.txt" not in await _tree(project_id), "a restore merged with what came after"

    # And a box filled from the restored tree holds the old bytes, not the new.
    session_id = str(
        await pool.fetchval(
            "INSERT INTO sessions (user_id, project_id, mode, status) "
            "VALUES ((SELECT user_id FROM projects WHERE id = $1), $1, 'attended', 'idle') RETURNING id",
            uuid.UUID(project_id),
        )
    )
    assert await sandbox_manager.claim_slot(session_id)
    sandbox = _sweeping(FakeSandbox())
    await workspace.materialize(sandbox, session_id, [workspace.Claim(project_id=project_id, slug="taxes")])

    assert sandbox.files[f"{workspace.MOUNT_ROOT}/taxes/a.txt"] == b"original"
    assert f"{workspace.MOUNT_ROOT}/taxes/new.txt" not in sandbox.files


async def test_restoring_leaves_other_projects_alone():
    mine, theirs = await _project("Mine"), await _project("Theirs")
    await store.commit_tree(mine, [_file("a.txt", "mine")])
    await store.commit_tree(theirs, [_file("a.txt", "theirs")])
    snapshot_id = await store.snapshot_project(mine)
    await store.commit_tree(mine, [_file("a.txt", "changed")])

    await store.restore_snapshot(snapshot_id)

    assert await store.get_blob((await store.read_tree(theirs))[0].content_hash) == b"theirs"


async def test_a_restore_whose_bytes_are_gone_is_refused(blobs):
    """The commit rule, applied backwards: no tree may point at bytes that are not there."""
    project_id = await _project()
    await store.commit_tree(project_id, [_file("a.txt", "one")])
    snapshot_id = await store.snapshot_project(project_id)
    await store.commit_tree(project_id, [_file("a.txt", "two")])
    for blob in blobs.root.rglob("*"):
        if blob.is_file():
            blob.unlink()

    with pytest.raises(store.StoreError, match="not in the store"):
        await store.restore_snapshot(snapshot_id)

    assert list(await _tree(project_id)) == ["a.txt"], "a refused restore still moved the tree"


async def test_restoring_a_snapshot_that_does_not_exist_says_so():
    with pytest.raises(store.StoreError, match="no snapshot"):
        await store.restore_snapshot(str(uuid.uuid4()))


# --- keeping the pile down ------------------------------------------------------------


async def test_pruning_keeps_the_newest():
    project_id = await _project()
    await store.commit_tree(project_id, [_file("a.txt", "one")])
    taken = [await store.snapshot_project(project_id, f"s{i}") for i in range(5)]

    deleted = await store.prune_snapshots(project_id, keep=2)

    assert deleted == 3
    assert [s.id for s in await store.list_snapshots(project_id)] == taken[-1:-3:-1]


async def test_pruning_to_keep_everything_deletes_nothing():
    project_id = await _project()
    await store.snapshot_project(project_id)

    assert await store.prune_snapshots(project_id, keep=50) == 0
