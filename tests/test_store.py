"""Blobs and trees: content addressing, and a commit that cannot half-happen.

Runs against a real Postgres with migration 0001 applied; blobs go to a tmp_path.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from db import pool
from harness_module import store

pytestmark = pytest.mark.asyncio

_seeded: list[uuid.UUID] = []


@pytest_asyncio.fixture(autouse=True)
async def _db(tmp_path):
    try:
        await pool.fetchval("SELECT 1")
    except Exception as e:  # noqa: BLE001 - any connection failure means skip
        await pool.close()
        pytest.skip(f"needs the arkos database (migrations applied): {e}")
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    yield
    store.use_blobs(None)
    await pool.execute("DELETE FROM projects WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM users WHERE id = ANY($1::uuid[])", _seeded)
    _seeded.clear()
    await pool.close()


async def _project() -> str:
    user_id = uuid.uuid4()
    await pool.execute("INSERT INTO users (id) VALUES ($1)", user_id)
    _seeded.append(user_id)
    return str(
        await pool.fetchval("INSERT INTO projects (user_id, title) VALUES ($1, 'p') RETURNING id", user_id)
    )


def _file(path: str, content: str) -> store.FileContent:
    return store.FileContent(path=path, content=content.encode())


# --- blobs -------------------------------------------------------------------------


async def test_a_blob_round_trips_under_its_hash():
    content_hash = await store.put_blob(b"alpha")

    assert content_hash == store.sha256(b"alpha")
    assert await store.get_blob(content_hash) == b"alpha"


async def test_the_same_content_is_one_blob(tmp_path):
    first = await store.put_blob(b"shared")
    second = await store.put_blob(b"shared")

    assert first == second
    on_disk = list(tmp_path.rglob("*"))
    assert len([p for p in on_disk if p.is_file()]) == 1


async def test_the_same_content_in_two_projects_stores_one_blob(tmp_path):
    first, second = await _project(), await _project()

    await store.commit_tree(first, [_file("a.txt", "identical")])
    await store.commit_tree(second, [_file("b/other.txt", "identical")])

    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert len(files) == 1, "content addressing did not dedup across projects"


async def test_a_missing_blob_reads_as_none():
    assert await store.get_blob("0" * 64) is None


async def test_missing_blobs_reports_only_what_is_absent():
    present = await store.put_blob(b"here")

    missing = await store.missing_blobs([present, "1" * 64])

    assert missing == {"1" * 64}


# --- trees --------------------------------------------------------------------------


async def test_a_commit_writes_the_tree_and_the_bytes():
    project_id = await _project()

    entries = await store.commit_tree(project_id, [_file("src/main.py", "print(1)"), _file("README.md", "hi")])

    assert [e.path for e in entries] == ["README.md", "src/main.py"]
    body = await store.get_blob(next(e.content_hash for e in entries if e.path == "README.md"))
    assert body == b"hi"
    assert next(e.size for e in entries if e.path == "src/main.py") == len("print(1)")


async def test_a_commit_replaces_the_tree_it_covers():
    project_id = await _project()
    await store.commit_tree(project_id, [_file("keep.txt", "1"), _file("drop.txt", "2")])

    await store.commit_tree(project_id, [_file("keep.txt", "1")])

    assert [e.path for e in await store.read_tree(project_id)] == ["keep.txt"]


async def test_a_commit_is_idempotent_on_retry():
    project_id = await _project()
    files = [_file("a.txt", "same"), _file("b.txt", "same too")]

    first = await store.commit_tree(project_id, files)
    second = await store.commit_tree(project_id, files)

    assert [(e.path, e.content_hash, e.size) for e in first] == [
        (e.path, e.content_hash, e.size) for e in second
    ]
    assert await pool.fetchval(
        "SELECT count(*) FROM project_files WHERE project_id = $1", uuid.UUID(project_id)
    ) == 2


async def test_a_commit_that_dies_before_the_rows_leaves_the_old_tree_whole(monkeypatch):
    """Blobs first, rows last: an interrupted commit is a no-op on the tree."""
    project_id = await _project()
    await store.commit_tree(project_id, [_file("original.txt", "before")])

    real_pool = pool.pool

    async def die(*a, **kw):
        raise RuntimeError("the process stopped here")

    monkeypatch.setattr(pool, "pool", die)
    with pytest.raises(RuntimeError):
        await store.commit_tree(project_id, [_file("replacement.txt", "after")])
    monkeypatch.setattr(pool, "pool", real_pool)

    tree = await store.read_tree(project_id)

    assert [e.path for e in tree] == ["original.txt"], "the tree was left half-written"
    assert await store.get_blob(next(e.content_hash for e in tree)) == b"before"
    # The blob for the abandoned commit is uploaded and orphaned, which is the
    # side the invariant errs on.
    assert await store.get_blob(store.sha256(b"after")) == b"after"


async def test_a_subpath_commit_leaves_the_rest_of_the_tree_alone():
    project_id = await _project()
    await store.commit_tree(project_id, [_file("src/a.py", "1"), _file("docs/b.md", "2")])

    await store.commit_tree(project_id, [_file("src/c.py", "3")], subpath="/src")

    assert [e.path for e in await store.read_tree(project_id)] == ["docs/b.md", "src/c.py"]


async def test_reading_a_subpath_returns_only_that_subtree():
    project_id = await _project()
    await store.commit_tree(project_id, [_file("src/a.py", "1"), _file("docs/b.md", "2")])

    assert [e.path for e in await store.read_tree(project_id, "/src")] == ["src/a.py"]


async def test_an_empty_commit_empties_the_tree():
    project_id = await _project()
    await store.commit_tree(project_id, [_file("a.txt", "1")])

    await store.commit_tree(project_id, [])

    assert await store.read_tree(project_id) == []


# --- diff ---------------------------------------------------------------------------


async def test_diff_reports_added_changed_and_removed_by_hash():
    project_id = await _project()
    before = await store.commit_tree(
        project_id, [_file("same.txt", "x"), _file("edit.txt", "1"), _file("gone.txt", "z")]
    )
    after = await store.commit_tree(
        project_id, [_file("same.txt", "x"), _file("edit.txt", "2"), _file("new.txt", "n")]
    )

    diff = store.diff_tree(before, after)

    assert diff.added == {"new.txt"}
    assert diff.changed == {"edit.txt"}
    assert diff.removed == {"gone.txt"}
    assert "same.txt" not in diff.paths


async def test_an_unchanged_tree_diffs_to_nothing():
    project_id = await _project()
    tree = await store.commit_tree(project_id, [_file("a.txt", "1")])

    assert not store.diff_tree(tree, tree)
