"""
The only state machine, and the sole writer of `sessions.status`.

It sits next to the agent, never inside it: no transition is ever LLM-gated.
Every move is a conditional UPDATE, so a lost race is a `False` return rather
than two writers disagreeing, and cancel wins by construction.
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

# Every legal move, and nothing else. The trigger for each is in contracts.md;
# what this map exists for is to make an illegal one impossible to write by
# accident, from any of the several call sites that move a session.
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
    """Raised for a move that is not in ALLOWED. A bug, not a race."""


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
        expected: the status the caller believes it is in; the UPDATE matches on it.
        new: the status to move to.
        reason: recorded on the event, and as `terminal_reason` when `new` is terminal.
        mode: flipped in the SAME update when given. A mode flip without a status
            change is illegal — both real flips (approve, unattended done) change
            status anyway — and doing it separately opens a window where a session
            is unattended for budget accounting but still recorded attended.

    Returns:
        True if this call made the move; False if it lost the race to another
        writer, which is how cancel wins.

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
        # Same transaction as the UPDATE: a status the transcript cannot explain
        # is exactly the opacity this table exists to remove.
        await session_log.append_tx(conn, session_id, LifecycleEvent(from_=expected, to=new, reason=reason))
    return True


def status_for(done: DoneEvent) -> Status | None:
    """
    Return the status a `done` event moves a running session to.

    `turn_end` is the attended "I have said my piece" and is NOT terminal: it is
    the one trigger for running -> idle, and leaves terminal_reason and ended_at
    NULL. The four failure reasons all bucket to `failed`, with the reason kept
    verbatim in `terminal_reason` so no second vocabulary appears.
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
    Fail every session still marked `running` at startup, and say why in its transcript.

    The process died underneath them. Nothing is requeued: a blind retry
    re-executes a side effect nobody closed. Restarting is a human act, and it
    uses the terminal -> running row above.
    """
    rows = await pool.fetch("SELECT id FROM sessions WHERE status = 'running'")
    swept = 0
    for row in rows:
        session_id = str(row["id"])
        try:
            # The transcript says why it stopped before the status does, and the
            # dangling call is closed first or the session cannot be folded again.
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
