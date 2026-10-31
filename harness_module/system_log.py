"""
The operational log: retries, lease churn, timing internals.

Separate from `session_events` by audience: that table is the transcript and a failed
append halts the run; this one is diagnostics, batched, and a failed write loses a line.

Two things feed it: explicit `record` calls from the runner, and every logging record
at WARNING or above from any module, forwarded by `_LoggingForwarder` while the log
is started. Without the forwarder, a tool or library error went only to stdout.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import deque
from typing import Any, Literal

from config_module.loader import cfg as _cfg
from db import pool
from db.ids import as_uuid_or_none as _uuid

logger = logging.getLogger(__name__)

Level = Literal["info", "warn", "error"]

# Pending rows, bounded: a write outage drops the oldest diagnostics rather than
# growing until the process dies.
_queue: deque[tuple[str, str, Any, Any, dict[str, Any]]] = deque(maxlen=10_000)
_flusher: asyncio.Task[None] | None = None
_stopping: asyncio.Event | None = None


# Seconds a shutdown waits for an in-flight write before giving up on it.
_SHUTDOWN_GRACE_S = 10.0

# The event name every forwarded logging record lands under.
LOGGING_EVENT = "log"
# Below this, the browser agent alone emits hundreds of lines per task.
_FORWARD_FROM = logging.WARNING


class _LoggingForwarder(logging.Handler):
    """Forwards warnings and errors from the standard logger into this log."""

    def __init__(self) -> None:
        super().__init__(level=_FORWARD_FROM)

    def emit(self, rec: logging.LogRecord) -> None:
        # This module warns when a write fails; forwarding that would queue a row
        # whose write fails, which warns again, forever.
        if rec.name == __name__:
            return
        fields: dict[str, Any] = {"logger": rec.name, "message": rec.getMessage()}
        if rec.exc_info and rec.exc_info[0] is not None:
            fields["traceback"] = logging.Formatter().formatException(rec.exc_info)
        record(LOGGING_EVENT, level="error" if rec.levelno >= logging.ERROR else "warn", **fields)


_forwarder: _LoggingForwarder | None = None


def record(
    event: str,
    *,
    level: Level = "info",
    session_id: str | None = None,
    user_id: str | None = None,
    **fields: Any,
) -> None:
    """Queue one operational record. Synchronous, and never raises."""
    try:
        # Round-tripped here rather than at flush: the pool's jsonb codec encodes with a
        # plain json.dumps, so a value it cannot handle would take down the whole batch.
        plain = json.loads(json.dumps(fields, default=str))
        _queue.append((level, event, _uuid(session_id), _uuid(user_id), plain))
    except Exception:  # noqa: BLE001 - a diagnostic must never break its caller
        logger.debug("could not queue system event %s", event, exc_info=True)


async def flush() -> int:
    """Write everything queued. Returns the number of rows written."""
    if not _queue:
        return 0
    batch = [_queue.popleft() for _ in range(len(_queue))]
    try:
        await pool.execute(
            """
            INSERT INTO system_events (level, event, session_id, user_id, fields)
            SELECT * FROM unnest($1::text[], $2::text[], $3::uuid[], $4::uuid[], $5::jsonb[])
            """,
            [r[0] for r in batch],
            [r[1] for r in batch],
            [r[2] for r in batch],
            [r[3] for r in batch],
            [r[4] for r in batch],
        )
    except Exception:
        # Dropped rather than re-queued: re-queueing risks looping on a poison batch.
        logger.warning("dropped %d system event(s): the write failed", len(batch), exc_info=True)
        return 0
    return len(batch)


async def start() -> None:
    """Begin flushing in the background, and prune what has aged out."""
    global _flusher, _stopping
    if _flusher is not None and not _flusher.done():
        return
    try:
        await prune()
    except Exception:
        logger.warning("could not prune system_events", exc_info=True)
    _stopping = asyncio.Event()
    _flusher = asyncio.create_task(_flush_loop(_stopping), name="system_log")
    _attach_forwarder()


async def stop() -> None:
    """Stop flushing and write what is left.

    The loop is asked to finish rather than cancelled: a cancel landing inside an
    in-flight write loses that batch, which `flush` has already taken off the queue.
    """
    global _flusher, _stopping
    _detach_forwarder()
    if _stopping is not None:
        _stopping.set()
    if _flusher is not None:
        try:
            await asyncio.wait_for(asyncio.shield(_flusher), timeout=_SHUTDOWN_GRACE_S)
        except (TimeoutError, asyncio.CancelledError):
            logger.warning("system_log did not stop in time; cancelling it")
            _flusher.cancel()
        except Exception:  # noqa: BLE001 - shutdown continues regardless
            logger.warning("system_log stopped with an error", exc_info=True)
        _flusher = None
    _stopping = None
    await flush()


def _attach_forwarder() -> None:
    global _forwarder
    if _forwarder is None:
        _forwarder = _LoggingForwarder()
        logging.getLogger().addHandler(_forwarder)


def _detach_forwarder() -> None:
    global _forwarder
    if _forwarder is not None:
        logging.getLogger().removeHandler(_forwarder)
        _forwarder = None


async def _flush_loop(stopping: asyncio.Event) -> None:
    """Flush on an interval until asked to stop, then once more."""
    interval = float(_cfg("system_log.flush_interval_s", 5))
    while not stopping.is_set():
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stopping.wait(), timeout=interval)
        try:
            await flush()
        except Exception:  # noqa: BLE001 - the loop outlives any single failure
            logger.warning("system_log flush failed", exc_info=True)


async def prune() -> int:
    """Delete records older than the retention window."""
    days = int(_cfg("system_log.retain_days", 30))
    result = await pool.execute(
        "DELETE FROM system_events WHERE ts < now() - make_interval(days => $1)",
        days,
    )
    return int(result.rsplit(" ", 1)[-1] or 0)
