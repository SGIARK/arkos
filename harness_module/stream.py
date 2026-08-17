"""
Live fan-out of session events to whoever is watching.

The transcript in Postgres is the truth; this is only the "no polling" half. A
subscriber that falls behind is told so rather than silently missing an event —
it re-reads from the log after its last seq and rejoins. That makes a dropped
queue slot a latency problem instead of a correctness one.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from harness_module.session_log import StoredEvent

logger = logging.getLogger(__name__)


class Lagged:
    """Sentinel: this subscriber's queue overflowed and must re-read from the log."""


LAGGED = Lagged()

Item = StoredEvent | Lagged


class SessionStream:
    """In-memory fan-out, one publisher per session and any number of subscribers."""

    def __init__(self, queue_size: int = 256):
        self._queue_size = queue_size
        self._subscribers: dict[str, set[asyncio.Queue[Item]]] = {}

    def publish(self, session_id: str, event: StoredEvent) -> None:
        """Hand one appended event to every live subscriber. Never blocks, never raises."""
        for queue in list(self._subscribers.get(session_id, ())):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drain to the sentinel: the subscriber is going back to the log
                # anyway, so holding stale events would only delay the catch-up.
                _drain(queue)
                queue.put_nowait(LAGGED)

    @asynccontextmanager
    async def subscribe(self, session_id: str) -> AsyncIterator[asyncio.Queue[Item]]:
        """
        Attach to a session's live events for the life of the context.

        Subscribe BEFORE reading the backlog, or an event appended between the
        read and the attach is delivered to nobody. The reader de-duplicates on
        seq, which costs nothing and closes that window.
        """
        queue: asyncio.Queue[Item] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.setdefault(session_id, set()).add(queue)
        try:
            yield queue
        finally:
            subscribers = self._subscribers.get(session_id)
            if subscribers is not None:
                subscribers.discard(queue)
                if not subscribers:
                    del self._subscribers[session_id]

    def subscriber_count(self, session_id: str) -> int:
        return len(self._subscribers.get(session_id, ()))


def _drain(queue: asyncio.Queue[Item]) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return


stream = SessionStream()
