"""
The session transcript: one append-only table for chat, tasks and everything else.

What `append` writes is what SSE pushes and what the fold reads back. An append
that fails raises, and the caller halts the run.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import asyncpg

from agent_module.events import Event, ToolCallEvent, ToolResultEvent, parse_event
from db import pool

logger = logging.getLogger(__name__)


class TranscriptError(RuntimeError):
    """Raised when an append would break the transcript invariant."""


@dataclass(slots=True)
class StoredEvent:
    """An event with the two row columns that are not part of its payload."""

    seq: int
    ts: datetime
    event: Event


# Tool-argument keys whose values are replaced before the event is stored. The
# transcript is durable and rendered in the UI.
_SECRET_KEY = re.compile(
    r"token|secret|password|passwd|api[-_ ]?key|authorization"
    r"|credential|private[-_ ]?key|dsn|conn(ection)?[-_ ]?string",
    re.IGNORECASE,
)
_REDACTED = "[redacted]"

# The current run: everything after the last `done`. The invariant checks below
# read only this tail.
_RUN_START = """
    SELECT COALESCE(MAX(seq), 0) FROM session_events WHERE session_id = $1 AND kind = 'done'
"""

_OPEN_CALLS = f"""
    SELECT c.seq AS seq, c.payload ->> 'id' AS call_id, c.payload ->> 'name' AS name
      FROM session_events c
     WHERE c.session_id = $1
       AND c.kind = 'tool_call'
       AND c.seq > ({_RUN_START})
       AND NOT EXISTS (
           SELECT 1 FROM session_events r
            WHERE r.session_id = $1
              AND r.kind = 'tool_result'
              -- A result closes only a call that precedes it. Without this an
              -- earlier run's result vouches for a later call reusing its id.
              AND r.seq > c.seq
              AND r.payload ->> 'id' = c.payload ->> 'id')
     ORDER BY c.seq
"""


def _redact(value: Any, *, key_matched: bool = False) -> Any:
    """Replace values held under a secret-looking key, recursively."""
    # Whole, whatever the shape: {"authorization": {"header": "Bearer ..."}}
    # hides the secret one level below the key that names it.
    if key_matched:
        return _REDACTED
    if isinstance(value, dict):
        return {k: _redact(v, key_matched=bool(_SECRET_KEY.search(str(k)))) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def _to_row(event: Event) -> dict[str, Any]:
    """Return the row to insert, with tool arguments redacted."""
    row = event.to_row()
    if isinstance(event, ToolCallEvent):
        row["payload"] = {**row["payload"], "args": _redact(event.args)}
    return row


async def append(session_id: str, event: Event) -> StoredEvent:
    """
    Append one event and return it with its assigned seq.

    Raises:
        TranscriptError: the event would break the transcript invariant.
        asyncpg.PostgresError: the write failed. Both are fatal to the run.
    """
    async with (await pool.pool()).acquire() as conn, conn.transaction():
        return await append_tx(conn, session_id, event)


async def append_tx(conn: asyncpg.Connection, session_id: str, event: Event) -> StoredEvent:
    """Append inside a caller's transaction, so a status change and its event commit together."""
    # BIGSERIAL assigns seq before commit, so the lock is what makes commit order
    # equal seq order within a session. Without it a transaction holding seq 100
    # can commit after 101, and a reader that has sent 101 never emits 100.
    # The canonical UUID text, not the caller's spelling: Postgres compares uuids
    # case-insensitively but hashes text exactly, so an uppercased id would take
    # a different lock and serialize against nothing.
    await conn.execute("SELECT pg_advisory_xact_lock(hashtextextended($1::text, 0))", str(_uuid(session_id)))
    await _check_invariant(conn, session_id, event)

    row = _to_row(event)
    record = await conn.fetchrow(
        """
        INSERT INTO session_events (session_id, kind, version, payload)
        VALUES ($1, $2, $3, $4)
        RETURNING seq, ts
        """,
        _uuid(session_id),
        row["kind"],
        row["version"],
        row["payload"],
    )
    return StoredEvent(seq=record["seq"], ts=record["ts"], event=event)


async def _check_invariant(conn: asyncpg.Connection, session_id: str, event: Event) -> None:
    """
    Enforce that every tool_call.id is closed by exactly one tool_result.

    Checked on `tool_result` and `done` only. `content` is appended once per
    streamed chunk, and a query per chunk would put a round trip on the token
    path; the loop appends every result within its hop, so no content event
    falls between a call and its result.
    """
    if isinstance(event, ToolResultEvent):
        open_ids = {r["call_id"] for r in await conn.fetch(_OPEN_CALLS, _uuid(session_id))}
        if event.id not in open_ids:
            raise TranscriptError(
                f"tool_result {event.id!r} closes no open tool_call in session {session_id} "
                "(already closed, or the call was never appended)"
            )
        return

    if event.kind == "done":
        dangling = await conn.fetch(_OPEN_CALLS, _uuid(session_id))
        if dangling:
            names = ", ".join(f"{r['name']}({r['call_id']})" for r in dangling)
            raise TranscriptError(
                f"session {session_id} cannot end with open tool calls: {names}. "
                "Call close_dangling() first."
            )


async def close_dangling(session_id: str) -> list[StoredEvent]:
    """
    Close every tool_call the run left open, with an `interrupted` result.

    Used by the abort path inside a live run and by the startup sweep. The
    outcome of such a call is unknown, which is what the result says.
    """
    closed: list[StoredEvent] = []
    async with (await pool.pool()).acquire() as conn, conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock(hashtextextended($1::text, 0))", str(_uuid(session_id)))
        for row in await conn.fetch(_OPEN_CALLS, _uuid(session_id)):
            event = ToolResultEvent(
                id=row["call_id"],
                ok=False,
                content=INTERRUPTED,
                error_kind="interrupted",
            )
            logger.warning("closing dangling tool_call %s (%s) on session %s", row["call_id"], row["name"], session_id)
            closed.append(await append_tx(conn, session_id, event))
    return closed


INTERRUPTED = "Interrupted before this returned. The outcome is unknown; verify before retrying."


async def get_events(session_id: str, after_seq: int = 0, limit: int = 500) -> list[StoredEvent]:
    """Return up to `limit` events after `after_seq`, in seq order."""
    rows = await pool.fetch(
        """
        SELECT seq, ts, kind, version, payload
          FROM session_events
         WHERE session_id = $1 AND seq > $2
         ORDER BY seq
         LIMIT $3
        """,
        _uuid(session_id),
        after_seq,
        limit,
    )
    return [_stored(r) for r in rows]


async def recent_events(session_id: str, limit: int = 200) -> list[StoredEvent]:
    """Return the last `limit` events in seq order, for the just-opened snapshot."""
    rows = await pool.fetch(
        """
        SELECT seq, ts, kind, version, payload FROM (
            SELECT seq, ts, kind, version, payload
              FROM session_events
             WHERE session_id = $1
             ORDER BY seq DESC
             LIMIT $2
        ) tail ORDER BY seq
        """,
        _uuid(session_id),
        limit,
    )
    return [_stored(r) for r in rows]


def _stored(record: asyncpg.Record) -> StoredEvent:
    return StoredEvent(
        seq=record["seq"],
        ts=record["ts"],
        event=parse_event(record["kind"], record["payload"], record["version"]),
    )


async def save_blob(session_id: str, content: str) -> str:
    """Store an oversized tool result whole and return its ref."""
    ref = await pool.fetchval(
        "INSERT INTO result_blobs (session_id, content) VALUES ($1, $2) RETURNING ref",
        _uuid(session_id),
        content,
    )
    return str(ref)


async def read_blob(ref: str, offset: int = 0, limit: int = 2000, *, user_id: str | None = None) -> str | None:
    """
    Return a slice of a stored blob, or None if the caller may not read it.

    `user_id` scopes the read to the blob's owner; refs are unguessable UUIDs,
    but that is not access control on its own.
    """
    try:
        ref_uuid = uuid.UUID(str(ref))
    except (ValueError, AttributeError, TypeError):
        return None
    if user_id is None:
        return await pool.fetchval(
            "SELECT substr(content, $2, $3) FROM result_blobs WHERE ref = $1",
            ref_uuid,
            offset + 1,  # substr is 1-based, the argument is not
            max(0, limit),
        )
    return await pool.fetchval(
        """
        SELECT substr(b.content, $3, $4)
          FROM result_blobs b JOIN sessions s ON s.id = b.session_id
         WHERE b.ref = $1 AND s.user_id = $2
        """,
        ref_uuid,
        _uuid(user_id),
        offset + 1,
        max(0, limit),
    )


def _uuid(value: str) -> uuid.UUID:
    """Parse an id as a UUID, which every key in this schema is."""
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError) as e:
        raise ValueError(f"expected a UUID, got {value!r}") from e
