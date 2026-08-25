"""
The user's memory: a curated core and the notes appended to it.

Keyed by USER, not by any tree, and it is explicitly not a filesystem — nothing
mounts it, no claim can name it, and `workspace` does not import this module.
Whether it should ever be mountable is D30, open; today the default posture is
that it is not, and nothing here is written to foreclose the answer.

Split out of `store.py` in 11.8.8. It shared a filename with the tree and shares
nothing else: different table, different key, different write discipline.

The discipline is the transcript's. A note is written once and never edited, so
concurrent sessions cannot collide. The core is the one file replaced whole, and
it is replaced under a transaction-scoped advisory lock on the user — the gate a
read-then-write compactor will need, held from the first version so it never has
to be retrofitted. The bytes go to a blob like everywhere else; the row also
carries the text, because `search_memory` is a Postgres full-text query and the
words have to be where the query runs.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from db import pool
from db.ids import as_uuid as _uuid
from harness_module.blobs import put_blob

# The curated core, and the directory one appended note lands in. Both are
# relative to the user's region.
MEMORY_CORE = "MEMORY.md"
NOTES_DIR = "notes"

# The advisory lock's namespace, so a memory lock cannot collide with any other
# advisory lock this database grows later.
_MEMORY_LOCK = 8808

# One statement for both writers: a note that has never been written before, and
# a core that is written over every time it is curated.
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
    Add one note to the user's memory. This is how a session records a fact.

    Each note is a file of its own, named for the moment it landed and a random
    suffix. Two sessions appending at once cannot collide or overwrite: nothing
    on this path reads a file in order to write it back, so no lock is needed.

    Returns:
        The note's path inside the region.
    """
    now = datetime.now(UTC)
    path = f"{NOTES_DIR}/{now.strftime('%Y%m%dT%H%M%S%f')}-{uuid.uuid4().hex[:8]}.md"
    content = text.encode()
    await pool.execute(
        _MEMORY_UPSERT, _uuid(user_id), path, await put_blob(content), len(content), now, text
    )
    return path


async def update_memory(user_id: str, text: str) -> None:
    """
    Replace the curated core with `text`, one writer at a time.

    The core is the one memory file that is rewritten rather than appended, so
    it is the one with a gate: the write takes a transaction-scoped advisory
    lock on the user, and a second curation waits for the first to finish. One
    upsert would be atomic without it; the lock is here because curation grows
    into read-then-write — the background compactor reading the notes before it
    replaces the core — and that is the version a gate has to already exist for.
    It is released with the transaction, including one that dies.

    Whoever calls this is the compactor. For now that is the model, reading the
    core and the notes and rewriting the core; the background job that will do
    it unattended is a later card and needs no other entry point than this.
    """
    # Blobs first, rows last, as everywhere in the store — and it keeps the lock
    # off an upload to another service.
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

    `websearch_to_tsquery` because the query is written by the model in the
    shape a person would type: bare words, quoted phrases, `or`. A query that
    parses to nothing matches nothing rather than everything.
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
    return [
        Hit(path=r["path"], text=r["body"], written_at=r["mtime"], rank=float(r["rank"])) for r in rows
    ]
