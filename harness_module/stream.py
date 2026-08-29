"""Live fan-out of session events to subscribers.

The log in Postgres is the record; this pushes only. A subscriber whose queue overflows
receives the LAGGED sentinel and re-reads from the log after its last seq.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from typing import TypeVar

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

    def publish_all(self, session_id: str, events: Iterable[StoredEvent]) -> None:
        """Publish a batch, in order.

        `close_dangling` returns events that were appended without being
        published, and the append-then-publish pair was copy-pasted at five
        call sites — where the fifth (`lifecycle.sweep_interrupted`) forgot the
        publish, so a watcher of a swept session saw the calls hang open
        forever. One helper is one place to forget it.
        """
        for event in events:
            self.publish(session_id, event)

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


T = TypeVar("T")


def _drain(queue: asyncio.Queue[T]) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return


stream = SessionStream()


class AttentionStream:
    """User-scoped invalidations for the account attention list."""

    def __init__(self, queue_size: int = 16, channel_limit: int = 1024):
        self._queue_size = queue_size
        self._channel_limit = channel_limit
        self._last_seq = 0
        self._latest: OrderedDict[str, int] = OrderedDict()
        self._subscribers: dict[str, set[asyncio.Queue[int]]] = {}

    def last_seq(self, user_id: str) -> int:
        return self._latest.get(user_id, 0)

    def publish(self, user_id: str) -> int:
        """Notify one user's subscribers that their attention list changed."""
        self._last_seq = max(time.time_ns(), self._last_seq + 1)
        self._latest[user_id] = self._last_seq
        self._latest.move_to_end(user_id)
        while len(self._latest) > self._channel_limit:
            self._latest.popitem(last=False)
        for queue in list(self._subscribers.get(user_id, ())):
            try:
                queue.put_nowait(self._last_seq)
            except asyncio.QueueFull:
                _drain(queue)
                queue.put_nowait(self._last_seq)
        return self._last_seq

    @asynccontextmanager
    async def subscribe(self, user_id: str) -> AsyncIterator[asyncio.Queue[int]]:
        queue: asyncio.Queue[int] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.setdefault(user_id, set()).add(queue)
        try:
            yield queue
        finally:
            subscribers = self._subscribers.get(user_id)
            if subscribers is not None:
                subscribers.discard(queue)
                if not subscribers:
                    del self._subscribers[user_id]


attention_stream = AttentionStream()
