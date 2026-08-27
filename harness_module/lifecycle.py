"""The session state machine, and the sole writer of `sessions.status`."""

from __future__ import annotations

import logging
from typing import Any, Literal

from agent_module.events import DoneEvent, LifecycleEvent
from db import pool
from db.ids import as_uuid as _uuid
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

# Mirrors migration 0's CHECK on `sessions.status`.
ALL_STATUSES: frozenset[str] = frozenset(
    {"pending", "idle", "running", "awaiting_approval", "completed", "failed", "cancelled"}
)

# Every legal move; the trigger for each is in contracts.md.
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
        # A declined plan: the only answer that ends a park without waking the session.
        ("awaiting_approval", "idle"),
        ("awaiting_approval", "cancelled"),
        # Terminal -> running is a human restarting the session; nothing auto-resumes.
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

    `reason` is also recorded as `terminal_reason` when `new` is terminal, and `mode`
    is set in the same UPDATE. Returns None when another writer got there first; the
    returned event has already been published.

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
        # Same transaction as the UPDATE, so the status and its explanation commit together.
        stored = await session_log.append_tx(conn, session_id, LifecycleEvent(from_=expected, to=new, reason=reason))
        await touch_project(conn, session_id)

    # Outside the block, so the seq being announced is one the log can already serve.
    stream.publish(session_id, stored)
    return stored


async def touch_project(conn: Any, session_id: str) -> None:
    """Mark the session's project as updated. A no-op for a session with no project.

    `projects.updated_at` has no trigger, so it moves only where code writes it.
    """
    await conn.execute(
        "UPDATE projects SET updated_at = now() WHERE id = (SELECT project_id FROM sessions WHERE id = $1)",
        _uuid(session_id),
    )


def status_for(done: DoneEvent) -> Status:
    """Returns the status a `done` event moves a running session to.

    `turn_end` and `stopped` are non-terminal, so both leave `terminal_reason` and
    `ended_at` NULL.
    """
    if done.reason in ("turn_end", "stopped"):
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
            stream.publish_all(session_id, await session_log.close_dangling(session_id))
            stream.publish(session_id, await session_log.append(session_id, DoneEvent(reason=reason)))
            if await transition(session_id, "running", "failed", reason):
                swept += 1
        except Exception:
            logger.exception("startup sweep could not fail session %s", session_id)
    if swept:
        logger.warning("startup sweep failed %d session(s) the process died underneath", swept)
    return swept
