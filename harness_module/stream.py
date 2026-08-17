"""Live fan-out of session events to subscribers.

The log in Postgres is the record; this pushes only. A subscriber whose queue overflows
receives the LAGGED sentinel and re-reads from the log after its last seq.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from harness_module.session_log import StoredEvent

logger = logging.getLogger(__name__)


class Lagged:
    """Sentinel telling a subscriber its queue overflowed and it re-reads from the log."""


LAGGED = Lagged()

Item = StoredEvent | Lagged


class SessionStream:
    """In-memory fan-out: one publisher per session, any number of subscribers."""

    def __init__(self, queue_size: int = 256):
        self._queue_size = queue_size
        self._subscribers: dict[str, set[asyncio.Queue[Item]]] = {}

    def publish(self, session_id: str, event: StoredEvent) -> None:
        """Hands one appended event to every subscriber. Never blocks, never raises."""
        for queue in list(self._subscribers.get(session_id, ())):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # The subscriber catches up from the log, so its queued events are
                # dropped and replaced by the sentinel.
                _drain(queue)
                queue.put_nowait(LAGGED)

    @asynccontextmanager
    async def subscribe(self, session_id: str) -> AsyncIterator[asyncio.Queue[Item]]:
        """Attaches to a session's live events for the life of the context.

        Callers subscribe before reading the backlog, so an event appended between the two
        still arrives; readers de-duplicate on seq.
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
