"""
The user's memory: a curated core and the notes appended to it.

Keyed by USER, not by any tree, and explicitly not a filesystem: nothing mounts it,
no claim can name it, and `workspace` does not import this module.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from db import pool
from db.ids import as_uuid as _uuid
from harness_module.blobs import put_blob

# Both paths are relative to the user's region.
MEMORY_CORE = "MEMORY.md"
NOTES_DIR = "notes"

# Advisory-lock namespace, so a memory lock cannot collide with any other lock here.
_MEMORY_LOCK = 8808

# One statement for both writers. `body` is stored alongside the blob hash because
# `search_memory` is a Postgres full-text query and the words must be where it runs.
_MEMORY_UPSERT = """
    INSERT INTO memory_files (user_id, path, content_hash, size, mtime, body)
    VALUES ($1, $2, $3, $4, $5, $6)
    ON CONFLICT (user_id, path) DO UPDATE
        SET content_hash = EXCLUDED.content_hash,
            size = EXCLUDED.size,
            mtime = EXCLUDED.mtime,
            body = EXCLUDED.body
"""


@dataclass(frozen=True, slots=True)
class Hit:
    """One search result, and how well it matched."""

    path: str
    text: str
    written_at: datetime
    rank: float

    @property
    def is_core(self) -> bool:
        return self.path == MEMORY_CORE


async def append_note(user_id: str, text: str) -> str:
    """
    Add one note to the user's memory, returning its path inside the region.

    Each note is a new file and nothing here reads a file to write it back, so
    concurrent appends cannot collide and no lock is needed.
    """
    now = datetime.now(UTC)
    path = f"{NOTES_DIR}/{now.strftime('%Y%m%dT%H%M%S%f')}-{uuid.uuid4().hex[:8]}.md"
    content = text.encode()
    await pool.execute(_MEMORY_UPSERT, _uuid(user_id), path, await put_blob(content), len(content), now, text)
    return path


async def update_memory(user_id: str, text: str) -> None:
    """
    Replace the curated core with `text`, one writer at a time.

    The core is the one memory file rewritten rather than appended, so it is gated by
    a transaction-scoped advisory lock on the user, released with the transaction.
    """
    # Blobs first, rows last, as everywhere in the store; it also keeps the lock off
    # an upload to another service.
    content = text.encode()
    content_hash = await put_blob(content)

    async with (await pool.pool()).acquire() as conn, conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock($1, hashtext($2))", _MEMORY_LOCK, str(user_id))
        await conn.execute(
            _MEMORY_UPSERT,
            _uuid(user_id),
            MEMORY_CORE,
            content_hash,
            len(content),
            datetime.now(UTC),
            text,
        )


async def read_memory(user_id: str) -> str:
    """The curated core, or '' when nothing has written one yet."""
    body = await pool.fetchval(
        "SELECT body FROM memory_files WHERE user_id = $1 AND path = $2",
        _uuid(user_id),
        MEMORY_CORE,
    )
    return body or ""


async def search_memory(user_id: str, query: str, limit: int = 10) -> list[Hit]:
    """
    Full-text search over one user's memory, core and notes alike.

    `websearch_to_tsquery` takes the query in the shape a person types — bare words,
    quoted phrases, `or` — and one that parses to nothing matches nothing.
    """
    rows = await pool.fetch(
        """
        SELECT path, body, mtime, ts_rank(tsv, q) AS rank
          FROM memory_files, websearch_to_tsquery('english', $2) AS q
         WHERE user_id = $1 AND tsv @@ q
         ORDER BY rank DESC, mtime DESC
         LIMIT $3
        """,
        _uuid(user_id),
        query,
        max(1, limit),
    )
    return [Hit(path=r["path"], text=r["body"], written_at=r["mtime"], rank=float(r["rank"])) for r in rows]
