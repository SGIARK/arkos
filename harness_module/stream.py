"""Live fan-out to subscribers: session events, and per-user attention signals.

The log in Postgres is the record; this pushes only. A subscriber whose queue overflows
receives the LAGGED sentinel and re-reads from the log after its last seq. Both channels
are published from the code that writes the row (contracts: ANNOUNCE FROM WHERE IT IS WRITTEN).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Any

from harness_module.session_log import StoredEvent

logger = logging.getLogger(__name__)


class Lagged:
    """Sentinel telling a subscriber its queue overflowed and it re-reads from the log."""


class Closed:
    """Sentinel telling a subscriber the server is going down.

    A stream generator that sees this yields nothing further and RETURNS, so the client
    reads a clean end of stream rather than an aborted connection.
    """


LAGGED = Lagged()
CLOSED = Closed()

Item = StoredEvent | Lagged | Closed


class _Fanout:
    """The subscriber machinery both channels share.

    An SSE response is an in-flight request that never finishes, so a graceful shutdown
    hangs unless the streams end themselves: `shutdown` does that for every channel at once.
    """

    def __init__(self, queue_size: int):
        self._queue_size = queue_size
        self._subscribers: dict[str, set[asyncio.Queue[Any]]] = {}

    @asynccontextmanager
    async def _subscribe(self, key: str) -> AsyncIterator[asyncio.Queue[Any]]:
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.setdefault(key, set()).add(queue)
        try:
            yield queue
        finally:
            subscribers = self._subscribers.get(key)
            if subscribers is not None:
                subscribers.discard(queue)
                if not subscribers:
                    del self._subscribers[key]

    def shutdown(self) -> int:
        """Wake every subscriber with CLOSED so its generator can return.

        Force-put past a full queue: a subscriber that is behind still has to learn the
        server is leaving. Returns how many were told.
        """
        told = 0
        for queues in list(self._subscribers.values()):
            for queue in list(queues):
                _drain(queue)
                queue.put_nowait(CLOSED)
                told += 1
        return told

    def subscriber_count(self, key: str | None = None) -> int:
        """For tests, and for a log line that answers "is anyone listening"."""
        if key is not None:
            return len(self._subscribers.get(key, ()))
        return sum(len(q) for q in self._subscribers.values())


class SessionStream(_Fanout):
    """In-memory fan-out: one publisher per session, any number of subscribers."""

    def __init__(self, queue_size: int = 256):
        super().__init__(queue_size)

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

    def publish_all(self, session_id: str, events: Iterable[StoredEvent]) -> None:
        """Publish a batch, in order."""
        for event in events:
            self.publish(session_id, event)

    def subscribe(self, session_id: str) -> AbstractAsyncContextManager[asyncio.Queue[Item]]:
        """Attaches to a session's live events for the life of the context.

        Callers subscribe before reading the backlog, so an event appended between the two
        still arrives; readers de-duplicate on seq.
        """
        return self._subscribe(session_id)


@dataclass(frozen=True, slots=True)
class AttentionSignal:
    """A nudge, not a payload: "your waiting list changed, read it again".

    Deliberately carries no approval row; the client refetches the list itself.
    """

    reason: str
    session_id: str


class UserStream(_Fanout):
    """In-memory fan-out keyed by user: one publisher per user, any subscribers."""

    def __init__(self, queue_size: int = 64):
        super().__init__(queue_size)

    def publish(self, user_id: str, signal: AttentionSignal) -> None:
        """Nudge every subscriber of one user. Never blocks, never raises.

        A full queue is DROPPED rather than sentinelled: an older copy of "read the list
        again" is worth nothing, and the reader is about to refetch anyway.
        """
        if not user_id:
            return
        for queue in list(self._subscribers.get(user_id, ())):
            try:
                queue.put_nowait(signal)
            except asyncio.QueueFull:
                logger.debug("attention queue full for %s; dropping a nudge", user_id)

    def subscribe(self, user_id: str) -> AbstractAsyncContextManager[asyncio.Queue[AttentionSignal | Closed]]:
        """Attach to one user's attention signals for the life of the context."""
        return self._subscribe(user_id)


def _drain(queue: asyncio.Queue[Item]) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return


stream = SessionStream()
attention = UserStream()


def shutdown_streams() -> int:
    """End every open stream, on every channel, cleanly. Called once, from the lifespan's teardown."""
    told = stream.shutdown() + attention.shutdown()
    if told:
        logger.info("shutdown: ended %d open stream(s)", told)
    return told
