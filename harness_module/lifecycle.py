"""
The state machine, and the sole writer of `sessions.status`.

Every move is a conditional UPDATE on the expected status, so a lost race
returns False rather than letting two writers disagree. No transition is
LLM-gated.
"""

from __future__ import annotations

import logging
import uuid
from typing import Literal

from agent_module.events import DoneEvent, LifecycleEvent
from db import pool
from harness_module import session_log

logger = logging.getLogger(__name__)

Status = Literal[
    "pending",
    "idle",
    "running",
    "awaiting_approval",
    "completed",
    "failed",
    "cancelled",
]

Mode = Literal["attended", "unattended"]

TERMINAL: frozenset[str] = frozenset({"completed", "failed", "cancelled"})

# Every legal move. Transitions not listed here raise; the triggers for each are
# in contracts.md.
ALLOWED: frozenset[tuple[str, str]] = frozenset(
    {
        ("pending", "running"),  # the runner claims the lease
        ("pending", "cancelled"),
        ("running", "idle"),  # done{turn_end}: attended, the model stopped calling tools
        ("running", "awaiting_approval"),  # a park tool
        ("running", "completed"),  # done{completed}
        ("running", "failed"),  # done{max_hops|wall_clock|model_error|context_overflow|interrupted}
        ("running", "cancelled"),
        ("idle", "running"),  # a human sends a message, or approves
        ("idle", "cancelled"),
        ("awaiting_approval", "running"),  # the respond endpoint wakes it
        ("awaiting_approval", "cancelled"),
        # A human restarts a finished session, or types into it. Nothing
        # auto-resumes: a blind retry re-executes an unclosed side effect.
        ("completed", "running"),
        ("failed", "running"),
        ("cancelled", "running"),
    }
)


class IllegalTransition(ValueError):
    """Raised for a move that is not in ALLOWED."""


async def transition(
    session_id: str,
    expected: Status,
    new: Status,
    reason: str,
    mode: Mode | None = None,
) -> bool:
    """
    Move a session from `expected` to `new`, atomically, appending a lifecycle event.

    Args:
        session_id: the session to move.
        expected: the status the UPDATE matches on.
        new: the status to move to.
        reason: recorded on the event, and as `terminal_reason` when `new` is terminal.
        mode: flipped in the same UPDATE when given, so a session is never
            unattended for budget accounting while still recorded attended.

    Returns:
        True if this call made the move, False if another writer got there first.

    Raises:
        IllegalTransition: the move is not in ALLOWED.
    """
    if (expected, new) not in ALLOWED:
        raise IllegalTransition(f"{expected} -> {new} is not a legal transition")

    terminal = new in TERMINAL
    async with (await pool.pool()).acquire() as conn, conn.transaction():
        moved = await conn.fetchval(
            """
            UPDATE sessions
               SET status          = $3,
                   mode            = COALESCE($4, mode),
                   terminal_reason = CASE WHEN $5 THEN $6 ELSE NULL END,
                   ended_at        = CASE WHEN $5 THEN now() ELSE NULL END
             WHERE id = $1 AND status = $2
            RETURNING id
            """,
            _uuid(session_id),
            expected,
            new,
            mode,
            terminal,
            reason,
        )
        if moved is None:
            logger.info("session %s: %s -> %s lost the race (not in %s)", session_id, expected, new, expected)
            return False
        # Same transaction as the UPDATE, so the status and its explanation
        # commit together.
        await session_log.append_tx(conn, session_id, LifecycleEvent(from_=expected, to=new, reason=reason))
    return True


def status_for(done: DoneEvent) -> Status | None:
    """
    Return the status a `done` event moves a running session to.

    `turn_end` is not terminal: it is the only trigger for running -> idle, and
    leaves `terminal_reason` and `ended_at` NULL. The failure reasons all bucket
    to `failed`, with the reason kept verbatim in `terminal_reason`.
    """
    if done.reason == "turn_end":
        return "idle"
    if done.reason == "completed":
        return "completed"
    if done.reason == "cancelled":
        return "cancelled"
    return "failed"


async def sweep_interrupted(reason: str = "interrupted") -> int:
    """
    Fail every session still marked `running` at startup, recording why.

    Nothing is requeued: a blind retry would re-execute a side effect nobody
    closed. Restarting is a human act, over the terminal -> running move.
    """
    rows = await pool.fetch("SELECT id FROM sessions WHERE status = 'running'")
    swept = 0
    for row in rows:
        session_id = str(row["id"])
        try:
            # The dangling call is closed first, or the session cannot be
            # folded back into messages.
            await session_log.close_dangling(session_id)
            await session_log.append(session_id, DoneEvent(reason=reason))
            if await transition(session_id, "running", "failed", reason):
                swept += 1
        except Exception:
            logger.exception("startup sweep could not fail session %s", session_id)
    if swept:
        logger.warning("startup sweep failed %d session(s) the process died underneath", swept)
    return swept


def _uuid(value: str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError) as e:
        raise ValueError(f"expected a UUID, got {value!r}") from e
