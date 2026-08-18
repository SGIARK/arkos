"""The store's schema: the tree is a map, and a claim names what a session may touch.

Runs against a real Postgres with migration 0001 applied.
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


async def test_a_path_appears_once_per_project():
    """The tree is a map from path to hash, not a log of writes."""
    _, project_id = await _project()
    await pool.execute(
        "INSERT INTO project_files (project_id, path, content_hash, size) VALUES ($1, 'a/b.txt', $2, 3)",
        project_id,
        _HASH,
    )

    with pytest.raises(asyncpg.UniqueViolationError):
        await pool.execute(
            "INSERT INTO project_files (project_id, path, content_hash, size) VALUES ($1, 'a/b.txt', $2, 9)",
            project_id,
            "1" * 64,
        )


async def test_the_same_path_in_two_projects_is_two_rows():
    _, first = await _project()
    _, second = await _project()

    for project_id in (first, second):
        await pool.execute(
            "INSERT INTO project_files (project_id, path, content_hash, size) VALUES ($1, 'README.md', $2, 3)",
            project_id,
            _HASH,
        )

    assert await pool.fetchval("SELECT count(*) FROM project_files WHERE content_hash = $1", _HASH) >= 2


async def test_the_tree_holds_no_bytes():
    """Bytes live in the store, addressed by hash."""
    columns = {
        r["column_name"]
        for r in await pool.fetch(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'project_files'"
        )
    }

    assert {"path", "content_hash", "size", "mtime"} <= columns
    assert not {"storage_path", "name", "size_bytes"} & columns, "a superseded column survived the migration"
    assert not {"content", "bytes", "data"} & columns


# --- claims ----------------------------------------------------------------------


async def test_a_claim_is_read_or_write_and_nothing_else():
    user_id, project_id = await _project()
    session_id = await _session(user_id)

    with pytest.raises(asyncpg.CheckViolationError):
        await pool.execute(
            "INSERT INTO session_claims (session_id, project_id, mode) VALUES ($1, $2, 'sideways')",
            session_id,
            project_id,
        )


async def test_a_claim_defaults_to_the_whole_project():
    user_id, project_id = await _project()
    session_id = await _session(user_id)

    await pool.execute(
        "INSERT INTO session_claims (session_id, project_id, mode) VALUES ($1, $2, 'write')",
        session_id,
        project_id,
    )

    assert await pool.fetchval("SELECT subpath FROM session_claims WHERE session_id = $1", session_id) == "/"


async def test_one_session_may_claim_two_subpaths_of_one_project():
    user_id, project_id = await _project()
    session_id = await _session(user_id)

    for subpath, mode in (("/src", "write"), ("/docs", "read")):
        await pool.execute(
            "INSERT INTO session_claims (session_id, project_id, subpath, mode) VALUES ($1, $2, $3, $4)",
            session_id,
            project_id,
            subpath,
            mode,
        )

    assert await pool.fetchval("SELECT count(*) FROM session_claims WHERE session_id = $1", session_id) == 2


async def test_claims_go_with_their_session():
    user_id, project_id = await _project()
    session_id = await _session(user_id)
    await pool.execute(
        "INSERT INTO session_claims (session_id, project_id, mode) VALUES ($1, $2, 'write')",
        session_id,
        project_id,
    )

    await pool.execute("DELETE FROM sessions WHERE id = $1", session_id)

    assert await pool.fetchval("SELECT count(*) FROM session_claims WHERE session_id = $1", session_id) == 0


async def test_claims_go_with_their_project():
    user_id, project_id = await _project()
    session_id = await _session(user_id)
    await pool.execute(
        "INSERT INTO session_claims (session_id, project_id, mode) VALUES ($1, $2, 'write')",
        session_id,
        project_id,
    )

    await pool.execute("DELETE FROM projects WHERE id = $1", project_id)

    assert await pool.fetchval("SELECT count(*) FROM session_claims WHERE project_id = $1", project_id) == 0
