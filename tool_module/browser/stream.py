"""The frame side-channel: what a user's browser is showing, for the app's browser pane.

Frames are not events: never appended, never replayed past the latest one, no seq.
Keyed by user, because each user has one browser that all their sessions share.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Backlog for a subscriber that is not keeping up; small so a slow reader gets
# the newest frame rather than stale ones.
_QUEUE_SIZE = 4


class FrameBroker:
    """In-memory fan-out of JPEG frames, per user."""

    def __init__(self, queue_size: int = _QUEUE_SIZE):
        self._subscribers: dict[str, set[asyncio.Queue[str]]] = {}
        # A page that is not changing sends no frames, so a new viewer starts from the last one.
        self._latest: dict[str, str] = {}
        self._queue_size = queue_size

    def publish(self, user_id: str, frame: str) -> None:
        """Hand one base64 JPEG to every viewer. Never blocks, never raises."""
        self._latest[str(user_id)] = frame
        for queue in list(self._subscribers.get(str(user_id), ())):
            _put_newest(queue, frame)

    @asynccontextmanager
    async def subscribe(self, user_id: str) -> AsyncIterator[asyncio.Queue[str]]:
        """Attach to a user's frames for the life of the context."""
        key = str(user_id)
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=self._queue_size)
        if key in self._latest:
            queue.put_nowait(self._latest[key])
        self._subscribers.setdefault(key, set()).add(queue)
        try:
            yield queue
        finally:
            watchers = self._subscribers.get(key)
            if watchers is not None:
                watchers.discard(queue)
                if not watchers:
                    del self._subscribers[key]


def _put_newest(queue: asyncio.Queue[str], frame: str) -> None:
    """Queue a frame, dropping the oldest ones if the reader is behind."""
    while queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:  # pragma: no cover - another reader drained it
            break
    with contextlib.suppress(asyncio.QueueFull):  # a reader filled it back up
        queue.put_nowait(frame)


broker = FrameBroker()
