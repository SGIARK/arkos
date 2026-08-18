"""The session state machine, and the sole writer of `sessions.status`.

Every move is a conditional UPDATE on the expected status: a caller that loses the race
gets None back, and each move that lands appends a lifecycle event in the same
transaction and publishes it once that transaction has committed.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Literal

from agent_module.events import DoneEvent, LifecycleEvent
from db import pool
from harness_module import session_log
from harness_module.stream import stream

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

# Every value `sessions.status` may hold, matching migration 0's CHECK. Read by
# callers that validate a status coming in from outside.
ALL_STATUSES: frozenset[str] = frozenset(
    {"pending", "idle", "running", "awaiting_approval", "completed", "failed", "cancelled"}
)

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
        # A human restarts a finished session, or types into it. Nothing auto-resumes.
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
) -> session_log.StoredEvent | None:
    """Moves a session from `expected` to `new` atomically, appending a lifecycle event.

    The event is published to the session's subscribers AFTER the transaction
    commits, and publishing lives here rather than in each caller: a status that
    moves without saying so is a pill that only updates when someone reconnects,
    and every caller having to remember the publish is how that happened once
    already. Publishing inside the transaction would be its own bug — a
    subscriber would be handed a seq that a `Last-Event-ID` reader cannot fetch
    yet.

    Args:
        session_id: the session to move.
        expected: the status the UPDATE matches on.
        new: the status to move to.
        reason: recorded on the event, and as `terminal_reason` when `new` is terminal.
        mode: set in the same UPDATE when given, so status and mode change together.

    Returns:
        The lifecycle event this call appended — truthy, so `if await
        transition(...)` still reads as "did I make the move" — or None if
        another writer got there first. It is already published; the return
        value is for a caller that needs the seq, not for publishing again.

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
            return None
        # Same transaction as the UPDATE, so the status and its explanation commit
        # together.
        stored = await session_log.append_tx(
            conn, session_id, LifecycleEvent(from_=expected, to=new, reason=reason)
        )
        await touch_project(conn, session_id)

    # Outside the block, so the seq being announced is one the log can already serve.
    stream.publish(session_id, stored)
    return stored


async def touch_project(conn: Any, session_id: str) -> None:
    """Mark the session's project as updated. A no-op for a session with no project.

    `projects.updated_at` has no trigger, so it moves only where code writes it.
    `GET /projects` and `list_projects` order by it.
    """
    await conn.execute(
        "UPDATE projects SET updated_at = now() WHERE id = (SELECT project_id FROM sessions WHERE id = $1)",
        _uuid(session_id),
    )


def status_for(done: DoneEvent) -> Status:
    """Returns the status a `done` event moves a running session to.

    `turn_end` is the only trigger for running -> idle, and is not terminal, so it leaves
    `terminal_reason` and `ended_at` NULL. `completed` and `cancelled` map to statuses of
    the same name; every other reason maps to `failed`, with the reason itself kept in
    `terminal_reason`.
    """
    if done.reason == "turn_end":
        return "idle"
    if done.reason == "completed":
        return "completed"
    if done.reason == "cancelled":
        return "cancelled"
    return "failed"


async def sweep_interrupted(reason: str = "interrupted") -> int:
    """Fails every session still marked `running` at startup, recording why.

    Nothing is requeued; a swept session restarts on a human's terminal -> running move.
    """
    rows = await pool.fetch("SELECT id FROM sessions WHERE status = 'running'")
    swept = 0
    for row in rows:
        session_id = str(row["id"])
        try:
            # Dangling calls close first: a session holding one cannot be folded back
            # into messages.
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
