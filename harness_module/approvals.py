"""Unanswered questions and consent requests raised by a running session.

A row is opened when a session parks and closed when a human answers.

Two shapes, and the difference matters. `ask` and `approval` are PROSE: the
model asks a question or describes an intention, and the answer comes back as a
`user` event for it to read. `call` is a GATED TOOL CALL: the session parked
with that call still open, the row carries the call that will actually run, and
answering does not talk to the model at all — it either runs that exact call or
closes it as declined. Consent binds to the call, never to a description of one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from db import pool
from db.ids import as_uuid as _uuid

Kind = Literal["approval", "ask", "call"]

_COLUMNS = (
    "id, session_id, tool_call_id, kind, prompt, answer, created_at, answered_at, "
    "tool_name, tool_args, consumed_at"
)

# What a human sends to resolve a gated call. Free text answers a question; a
# call is a decision, and it gets a vocabulary of exactly two words.
APPROVE = "approve"
DECLINE = "decline"


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
    # Set only on `call` rows: the tool that will run if this is approved.
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    # Claimed by the wake that executed it. See `consume`.
    consumed_at: datetime | None = None

    @property
    def gated_call(self) -> bool:
        """True for a parked tool call, whose answer runs code rather than being read."""
        return self.kind == "call"

    @property
    def approved(self) -> bool:
        """True when a human said yes. Anything that is not `approve` is not consent."""
        return (self.answer or "").strip().lower() == APPROVE


def _row(record: Any) -> Approval:
    args = record["tool_args"]
    return Approval(
        id=str(record["id"]),
        session_id=str(record["session_id"]),
        tool_call_id=record["tool_call_id"],
        kind=record["kind"],
        prompt=record["prompt"],
        answer=record["answer"],
        created_at=record["created_at"],
        answered_at=record["answered_at"],
        tool_name=record["tool_name"],
        # asyncpg hands back jsonb as text unless a codec is registered.
        tool_args=json.loads(args) if isinstance(args, str) else args,
        consumed_at=record["consumed_at"],
    )


async def create(
    session_id: str,
    tool_call_id: str,
    kind: Kind,
    prompt: str,
    *,
    tool_name: str | None = None,
    tool_args: dict[str, Any] | None = None,
) -> Approval:
    """Open a question against a session.

    A partial unique index on the table permits at most one unanswered row per
    tool call, so a second call for a tool call that is already parked raises.

    `tool_name` and `tool_args` are the gated call itself, stored so the human
    approves what will run and the resumed turn runs it without asking the model
    to describe its intention a second time.
    """
    record = await pool.fetchrow(
        f"""
        INSERT INTO approvals (session_id, tool_call_id, kind, prompt, tool_name, tool_args)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING {_COLUMNS}
        """,
        _uuid(session_id),
        tool_call_id,
        kind,
        prompt,
        tool_name,
        json.dumps(tool_args) if tool_args is not None else None,
    )
    return _row(record)


async def grantable(session_id: str) -> Approval | None:
    """Return the session's answered gated call, claimed or not, newest first.

    The caller decides what to do with it from `consumed_at`: unclaimed means run
    it, already claimed means a previous wake died mid-flight and the call needs
    repairing rather than repeating.
    """
    record = await pool.fetchrow(
        f"""
        SELECT {_COLUMNS} FROM approvals
         WHERE session_id = $1 AND kind = 'call' AND answered_at IS NOT NULL
         ORDER BY answered_at DESC
         LIMIT 1
        """,
        _uuid(session_id),
    )
    return _row(record) if record else None


async def consume(approval_id: str) -> Approval | None:
    """Claim a granted call for execution. Exactly one caller wins.

    The same conditional-update pattern as `answer`, and for the same reason: two
    wakes racing to resume one parked session must not both send the email. The
    losers get None and leave the call alone.
    """
    record = await pool.fetchrow(
        f"""
        UPDATE approvals SET consumed_at = now()
         WHERE id = $1 AND consumed_at IS NULL
        RETURNING {_COLUMNS}
        """,
        _uuid(approval_id),
    )
    return _row(record) if record else None


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
        SELECT a.id, a.session_id, a.tool_call_id, a.kind, a.prompt, a.answer,
               a.created_at, a.answered_at, a.tool_name, a.tool_args, a.consumed_at
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


