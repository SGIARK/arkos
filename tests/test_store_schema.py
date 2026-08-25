"""The store's schema: the tree is a map, a folder is derived, a link is a row.

Since 11.9 the tree is ONE flat namespace per user — `files (user_id, path)` —
and a folder is `path.split('/')[0]`, which is why nothing here creates one. A
project LINKS folders (`project_folders`) and owns none, so deleting a project
takes its links and leaves every file where it was.

Runs against a real Postgres with the migrations applied.
"""

from __future__ import annotations

import uuid

import asyncpg
import pytest
import pytest_asyncio

from db import pool
from tests.dbgate import require_db

pytestmark = pytest.mark.asyncio

_seeded: list[uuid.UUID] = []
_HASH = "0" * 64


@pytest_asyncio.fixture(autouse=True)
async def _db():
    await require_db()
    yield
    await pool.execute("DELETE FROM sessions WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM files WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM projects WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM users WHERE id = ANY($1::uuid[])", _seeded)
    _seeded.clear()
    await pool.close()


async def _project() -> tuple[uuid.UUID, uuid.UUID]:
    user_id = uuid.uuid4()
    await pool.execute("INSERT INTO users (id) VALUES ($1)", user_id)
    _seeded.append(user_id)
    project_id = await pool.fetchval(
        "INSERT INTO projects (user_id, title) VALUES ($1, 'p') RETURNING id", user_id
    )
    return user_id, project_id


async def _session(user_id: uuid.UUID) -> uuid.UUID:
    return await pool.fetchval(
        "INSERT INTO sessions (user_id, mode, status) VALUES ($1, 'attended', 'idle') RETURNING id",
        user_id,
    )


# --- the tree --------------------------------------------------------------------


async def test_a_path_appears_once_per_user():
    """The tree is a map from path to hash, not a log of writes."""
    user_id, _ = await _project()
    await pool.execute(
        "INSERT INTO files (user_id, path, content_hash, size) VALUES ($1, 'a/b.txt', $2, 3)",
        user_id,
        _HASH,
    )

    with pytest.raises(asyncpg.UniqueViolationError):
        await pool.execute(
            "INSERT INTO files (user_id, path, content_hash, size) VALUES ($1, 'a/b.txt', $2, 9)",
            user_id,
            "1" * 64,
        )


async def test_the_same_path_in_two_stores_is_two_rows():
    first, _ = await _project()
    second, _ = await _project()

    for user_id in (first, second):
        await pool.execute(
            "INSERT INTO files (user_id, path, content_hash, size) VALUES ($1, 'p/README.md', $2, 3)",
            user_id,
            _HASH,
        )

    assert await pool.fetchval("SELECT count(*) FROM files WHERE content_hash = $1", _HASH) >= 2


async def test_the_tree_holds_no_bytes():
    """Bytes live in the store, addressed by hash."""
    columns = {
        r["column_name"]
        for r in await pool.fetch(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'files'"
        )
    }

    assert {"user_id", "path", "content_hash", "size", "mtime"} <= columns
    assert not {"project_id", "storage_path", "name", "size_bytes"} & columns
    assert not {"content", "bytes", "data"} & columns


async def test_the_project_scoped_tree_is_gone():
    """One store, one table. A second one is a second path for code to grow into."""
    assert await pool.fetchval("SELECT to_regclass('public.project_files')") is None


async def test_no_table_holds_a_folder():
    """A folder is a path segment. There is nothing to insert, rename or orphan."""
    user_id, _ = await _project()
    await pool.execute(
        "INSERT INTO files (user_id, path, content_hash, size) VALUES ($1, 'triage/a.txt', $2, 3)",
        user_id,
        _HASH,
    )

    folders = await pool.fetch(
        "SELECT DISTINCT split_part(path, '/', 1) AS name FROM files WHERE user_id = $1", user_id
    )

    assert [r["name"] for r in folders] == ["triage"]
    assert await pool.fetchval("SELECT to_regclass('public.folders')") is None


# --- links -----------------------------------------------------------------------


async def test_a_project_links_a_folder_and_linking_twice_is_one_link():
    _, project_id = await _project()

    for _ in range(2):
        await pool.execute(
            "INSERT INTO project_folders (project_id, folder) VALUES ($1, 'triage') ON CONFLICT DO NOTHING",
            project_id,
        )

    assert await pool.fetchval(
        "SELECT count(*) FROM project_folders WHERE project_id = $1", project_id
    ) == 1


async def test_two_projects_may_link_the_same_folder():
    """Nothing owns a folder, so nothing has to be taken from anybody to share it."""
    user_id, first = await _project()
    second = await pool.fetchval(
        "INSERT INTO projects (user_id, title) VALUES ($1, 'other') RETURNING id", user_id
    )

    for project_id in (first, second):
        await pool.execute(
            "INSERT INTO project_folders (project_id, folder) VALUES ($1, 'triage')", project_id
        )

    assert await pool.fetchval("SELECT count(*) FROM project_folders WHERE folder = 'triage'") == 2


async def test_deleting_a_project_deletes_its_links_and_no_files():
    user_id, project_id = await _project()
    await pool.execute(
        "INSERT INTO files (user_id, path, content_hash, size) VALUES ($1, 'triage/a.txt', $2, 3)",
        user_id,
        _HASH,
    )
    await pool.execute("INSERT INTO project_folders (project_id, folder) VALUES ($1, 'triage')", project_id)

    await pool.execute("DELETE FROM projects WHERE id = $1", project_id)

    assert await pool.fetchval(
        "SELECT count(*) FROM project_folders WHERE project_id = $1", project_id
    ) == 0
    assert await pool.fetchval("SELECT count(*) FROM files WHERE user_id = $1", user_id) == 1


# --- claims ----------------------------------------------------------------------


async def test_a_claim_is_read_or_write_and_nothing_else():
    user_id, _ = await _project()
    session_id = await _session(user_id)

    with pytest.raises(asyncpg.CheckViolationError):
        await pool.execute(
            "INSERT INTO session_claims (session_id, folder, mode) VALUES ($1, 'triage', 'sideways')",
            session_id,
        )


async def test_a_claim_names_a_folder_and_defaults_to_all_of_it():
    user_id, _ = await _project()
    session_id = await _session(user_id)

    await pool.execute(
        "INSERT INTO session_claims (session_id, folder, mode) VALUES ($1, 'triage', 'write')",
        session_id,
    )

    row = await pool.fetchrow(
        "SELECT folder, subpath FROM session_claims WHERE session_id = $1", session_id
    )
    assert (row["folder"], row["subpath"]) == ("triage", "/")


async def test_one_session_may_claim_two_folders():
    """Which is the point: a project links several, and every one of them mounts."""
    user_id, _ = await _project()
    session_id = await _session(user_id)

    for folder, mode in (("triage", "write"), ("notes", "read")):
        await pool.execute(
            "INSERT INTO session_claims (session_id, folder, mode) VALUES ($1, $2, $3)",
            session_id,
            folder,
            mode,
        )

    assert await pool.fetchval("SELECT count(*) FROM session_claims WHERE session_id = $1", session_id) == 2


async def test_claims_go_with_their_session():
    user_id, _ = await _project()
    session_id = await _session(user_id)
    await pool.execute(
        "INSERT INTO session_claims (session_id, folder, mode) VALUES ($1, 'triage', 'write')",
        session_id,
    )

    await pool.execute("DELETE FROM sessions WHERE id = $1", session_id)

    assert await pool.fetchval("SELECT count(*) FROM session_claims WHERE session_id = $1", session_id) == 0


async def test_a_claim_outlives_the_project_that_caused_it():
    """It names a folder, and the folder is the store's. Nothing to cascade from."""
    user_id, project_id = await _project()
    session_id = await _session(user_id)
    await pool.execute(
        "INSERT INTO session_claims (session_id, folder, mode) VALUES ($1, 'triage', 'write')",
        session_id,
    )

    await pool.execute("DELETE FROM projects WHERE id = $1", project_id)

    assert await pool.fetchval("SELECT count(*) FROM session_claims WHERE session_id = $1", session_id) == 1
