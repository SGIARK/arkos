"""Unanswered questions and consent requests raised by a running session.

A row is opened when a session parks and closed when a human answers. The
answer is delivered to the session as a `user` event; it is not returned to the
tool call that raised it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from db import pool

Kind = Literal["approval", "ask"]

_COLUMNS = "id, session_id, tool_call_id, kind, prompt, answer, created_at, answered_at"


@dataclass(slots=True)
class Approval:
    id: str
    session_id: str
    tool_call_id: str
    kind: str
    prompt: str
    answer: str | None
    created_at: datetime
    answered_at: datetime | None


def _row(record: Any) -> Approval:
    return Approval(
        id=str(record["id"]),
        session_id=str(record["session_id"]),
        tool_call_id=record["tool_call_id"],
        kind=record["kind"],
        prompt=record["prompt"],
        answer=record["answer"],
        created_at=record["created_at"],
        answered_at=record["answered_at"],
    )


async def create(session_id: str, tool_call_id: str, kind: Kind, prompt: str) -> Approval:
    """Open a question against a session.

    A partial unique index on the table permits at most one unanswered row per
    tool call, so a second call for a tool call that is already parked raises.
    """
    record = await pool.fetchrow(
        f"""
        INSERT INTO approvals (session_id, tool_call_id, kind, prompt)
        VALUES ($1, $2, $3, $4)
        RETURNING {_COLUMNS}
        """,
        _uuid(session_id),
        tool_call_id,
        kind,
        prompt,
    )
    return _row(record)


async def open_for(session_id: str) -> list[Approval]:
    """Return the session's unanswered questions, oldest first."""
    rows = await pool.fetch(
        f"SELECT {_COLUMNS} FROM approvals WHERE session_id = $1 AND answered_at IS NULL ORDER BY created_at",
        _uuid(session_id),
    )
    return [_row(r) for r in rows]


async def get(approval_id: str, user_id: str) -> Approval | None:
    """Return one approval the caller owns, or None."""
    record = await pool.fetchrow(
        """
        SELECT a.id, a.session_id, a.tool_call_id, a.kind, a.prompt, a.answer, a.created_at, a.answered_at
          FROM approvals a JOIN sessions s ON s.id = a.session_id
         WHERE a.id = $1 AND s.user_id = $2
        """,
        _uuid(approval_id),
        _uuid(user_id),
    )
    return _row(record) if record else None


async def answer(approval_id: str, text: str) -> Approval | None:
    """Record an answer to an unanswered question.

    The update matches only on `answered_at IS NULL`, so concurrent answers to
    the same question resolve to one update; the losers return None.
    """
    record = await pool.fetchrow(
        f"""
        UPDATE approvals SET answer = $2, answered_at = now()
         WHERE id = $1 AND answered_at IS NULL
        RETURNING {_COLUMNS}
        """,
        _uuid(approval_id),
        text,
    )
    return _row(record) if record else None


def _uuid(value: str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))
