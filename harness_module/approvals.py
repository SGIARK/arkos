"""Unanswered questions and consent requests raised by a running session.

`ask`/`approval` are prose answered back to the model; `call` is a gated tool
call whose row carries the call that will actually run, so consent binds to the
call and never to a description of one; `plan` carries the proposed plan in
`tool_args` and each new proposal supersedes the open row rather than joining it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from db import pool
from db.ids import as_uuid as _uuid
from harness_module.stream import AttentionSignal, attention

logger = logging.getLogger(__name__)

Kind = Literal["approval", "ask", "call", "plan"]

_COLUMNS = (
    "id, session_id, tool_call_id, kind, prompt, answer, created_at, answered_at, "
    "tool_name, tool_args, consumed_at, answered_by"
)

# `answered_by` value for a harness-answered gate; NULL means a human.
AUTO = "auto"

# A gated call resolves on exactly these two words; other kinds take free text.
APPROVE = "approve"
DECLINE = "decline"

# Written to `answer` when a newer `propose_plan` replaces an open plan row: it
# closes the row without being the approve word, so `approved` stays False.
SUPERSEDED = "superseded"


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
    # NULL for a human, `auto` for autopilot answering its own gate.
    answered_by: str | None = None

    @property
    def auto_answered(self) -> bool:
        return self.answered_by == AUTO

    @property
    def gated_call(self) -> bool:
        """True for a parked tool call, whose answer runs code rather than being read."""
        return self.kind == "call"

    @property
    def is_plan(self) -> bool:
        """True for a proposed plan, whose `tool_args` are the plan itself."""
        return self.kind == "plan"

    @property
    def approved(self) -> bool:
        """True only for the exact approve word; anything else is not consent."""
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
        answered_by=record["answered_by"],
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

    Raises if the tool call already has an unanswered row: a partial unique index
    permits at most one per tool call.
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
    await _announce(session_id, "parked")
    return _row(record)


async def grantable(session_id: str) -> Approval | None:
    """Return the session's answered gated call, claimed or not, newest first.

    `consumed_at` tells the caller which: unclaimed means run it, already claimed
    means an earlier wake died mid-flight and the call needs repair, not a repeat.
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
    """Claim a granted call for execution; exactly one caller wins, losers get None."""
    record = await pool.fetchrow(
        f"""
        UPDATE approvals SET consumed_at = now()
         WHERE id = $1 AND consumed_at IS NULL
        RETURNING {_COLUMNS}
        """,
        _uuid(approval_id),
    )
    return _row(record) if record else None


async def supersede_plans(session_id: str) -> int:
    """Close any open plan row on this session and return how many were closed.

    Called before writing version n+1; the old row keeps its args for the diff.
    """
    rows = await pool.fetch(
        """
        UPDATE approvals SET answer = $2, answered_at = now()
         WHERE session_id = $1 AND kind = 'plan' AND answered_at IS NULL
        RETURNING id
        """,
        _uuid(session_id),
        SUPERSEDED,
    )
    return len(rows)


async def reopen(approval_id: str) -> Approval | None:
    """Un-answer a plan row whose authorised action did not happen.

    Compensating action only, never a way to reverse a human decision: a `call`
    row is latched by `consumed_at` and is never reopened, as the tool may have run.
    """
    record = await pool.fetchrow(
        f"""
        UPDATE approvals SET answer = NULL, answered_at = NULL
         WHERE id = $1 AND kind = 'plan' AND consumed_at IS NULL
        RETURNING {_COLUMNS}
        """,
        _uuid(approval_id),
    )
    return _row(record) if record else None


async def plan_history(session_id: str) -> list[Approval]:
    """Every plan this session has proposed, oldest first.

    A row's version is its 1-based position here; the row before the newest is
    what the card diffs against.
    """
    rows = await pool.fetch(
        f"SELECT {_COLUMNS} FROM approvals WHERE session_id = $1 AND kind = 'plan' ORDER BY created_at, id",
        _uuid(session_id),
    )
    return [_row(r) for r in rows]


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
               a.created_at, a.answered_at, a.tool_name, a.tool_args, a.consumed_at,
               a.answered_by
          FROM approvals a JOIN sessions s ON s.id = a.session_id
         WHERE a.id = $1 AND s.user_id = $2
        """,
        _uuid(approval_id),
        _uuid(user_id),
    )
    return _row(record) if record else None


async def _announce(session_id: str, reason: str) -> None:
    """Tell the session's owner their waiting list moved.

    Published from the write, not by the mounted surface, so a park reaches the
    user channel and not only a session stream; failures are swallowed on purpose.
    """
    try:
        user_id = await pool.fetchval("SELECT user_id FROM sessions WHERE id = $1", _uuid(session_id))
        if user_id:
            attention.publish(str(user_id), AttentionSignal(reason=reason, session_id=session_id))
    except Exception:  # noqa: BLE001 - a signal is never worth failing a write over
        logger.warning("could not announce attention for session %s", session_id, exc_info=True)


async def answer_auto(approval_id: str, text: str) -> Approval | None:
    """Answer a row as the harness rather than as a human, stamping `answered_by`.

    Otherwise identical to `answer`, announcement included.
    """
    record = await pool.fetchrow(
        f"""
        UPDATE approvals SET answer = $2, answered_at = now(), answered_by = $3
         WHERE id = $1 AND answered_at IS NULL
        RETURNING {_COLUMNS}
        """,
        _uuid(approval_id),
        text,
        AUTO,
    )
    if not record:
        return None
    approval = _row(record)
    await _announce(approval.session_id, "answered")
    return approval


async def answer(approval_id: str, text: str) -> Approval | None:
    """Record an answer to an unanswered question.

    Matches only on `answered_at IS NULL`, so concurrent answers resolve to one
    update and the losers return None.
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
    if not record:
        return None
    approval = _row(record)
    await _announce(approval.session_id, "answered")
    return approval
