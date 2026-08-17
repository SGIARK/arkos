"""
The transcript: one append-only table for chat, tasks and everything else.

Store-shape equals wire-shape, so what `append` writes is what SSE pushes and
what the fold reads back. A failed append halts the run: nothing executes off
the record.
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
    """One row: the event plus the two columns the wire needs but the payload has no room for."""

    seq: int
    ts: datetime
    event: Event


# Values under these keys never reach the log. The transcript is durable and
# rendered in the UI, so a token pasted into tool args would outlive its rotation.
_SECRET_KEY = re.compile(
    r"token|secret|password|passwd|api[-_ ]?key|authorization|credential|private[-_ ]?key|dsn|conn(ection)?[-_ ]?string",
    re.IGNORECASE,
)
_REDACTED = "[redacted]"

# The tail of the log that belongs to the current run. Everything before the last
# `done` is a finished run whose calls are closed, so the invariant checks below
# never have to read it.
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
              AND r.payload ->> 'id' = c.payload ->> 'id')
     ORDER BY c.seq
"""


def _redact(value: Any, *, key_matched: bool = False) -> Any:
    """Recursively replace values held under a secret-looking key."""
    if key_matched and isinstance(value, str):
        return _REDACTED
    if isinstance(value, dict):
        return {k: _redact(v, key_matched=bool(_SECRET_KEY.search(str(k)))) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v, key_matched=key_matched) for v in value]
    return _REDACTED if key_matched else value


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
        TranscriptError: when the event would break the transcript invariant.
        asyncpg.PostgresError: on any database failure. Both are fatal to the
            run by design — a run that cannot record itself must not continue.
    """
    async with (await pool.pool()).acquire() as conn, conn.transaction():
        return await append_tx(conn, session_id, event)


async def append_tx(conn: asyncpg.Connection, session_id: str, event: Event) -> StoredEvent:
    """Append inside a caller's transaction, so a status change and its event commit together."""
    # BIGSERIAL hands out seq before commit, so without this an api transaction
    # holding seq 100 can commit after the loop's 101 and an SSE reader that
    # already sent 101 never emits 100. The lock makes commit order equal seq
    # order within a session.
    await conn.execute("SELECT pg_advisory_xact_lock(hashtextextended($1::text, 0))", session_id)
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
    Enforce: every tool_call.id is closed by exactly one tool_result before the run ends.

    Checked on the two kinds where it can be violated cheaply and meaningfully.
    `content` is deliberately exempt: it is appended once per streamed chunk, and
    a query per chunk would put a round trip on the token path. The loop appends
    every result within its hop, so no content event can fall between a call and
    its result.
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
    Close every tool_call this run left open, with the result nobody was alive to write.

    Used on two paths that look different and are the same: the abort inside a
    live run, and the startup sweep over a session the process died underneath.
    The outcome of the call is genuinely unknown, which is what the result says.
    """
    closed: list[StoredEvent] = []
    async with (await pool.pool()).acquire() as conn, conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock(hashtextextended($1::text, 0))", session_id)
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
    """Return events after `after_seq` in seq order. Every read is 'after N', never 'count'."""
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
    """Store an oversized tool result whole; the event carries a preview and this ref."""
    ref = await pool.fetchval(
        "INSERT INTO result_blobs (session_id, content) VALUES ($1, $2) RETURNING ref",
        _uuid(session_id),
        content,
    )
    return str(ref)


async def read_blob(ref: str, offset: int = 0, limit: int = 2000, *, user_id: str | None = None) -> str | None:
    """
    Return a slice of a stored blob, or None when there is none the caller may read.

    `user_id` scopes the read to its owner. Refs are unguessable UUIDs, but
    unguessable is not access control (auth.md), so every caller passes one.
    """
    try:
        ref_uuid = uuid.UUID(str(ref))
    except (ValueError, AttributeError, TypeError):
        return None
    if user_id is None:
        return await pool.fetchval(
            "SELECT substr(content, $2, $3) FROM result_blobs WHERE ref = $1",
            ref_uuid,
            offset + 1,  # substr is 1-based; the tool argument is not
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
    """Parse an id as the UUID every key in this schema is."""
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError) as e:
        raise ValueError(f"expected a UUID, got {value!r}") from e
