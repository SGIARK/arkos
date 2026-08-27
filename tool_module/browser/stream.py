"""The frame side-channel: what the browser is looking at, while it looks.

Frames are not events: never appended, never replayed, no seq. Keyed by
`(user_id, session_id)`, so a user's concurrent sessions never share a queue.
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

Key = tuple[str, str]


# Nothing here is appended to the log: a `status` event carries this stream's
# URL, and that event is what mounts the UI pane.
class FrameBroker:
    """In-memory fan-out of JPEG frames, per (user, session)."""

    def __init__(self, queue_size: int = _QUEUE_SIZE):
        self._subscribers: dict[Key, set[asyncio.Queue[str]]] = {}
        self._queue_size = queue_size

    def publish(self, user_id: str, session_id: str, frame: str) -> None:
        """Hand one base64 JPEG to every viewer. Never blocks, never raises."""
        for queue in list(self._subscribers.get((str(user_id), str(session_id)), ())):
            while queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:  # pragma: no cover - another reader drained it
                    break
            with contextlib.suppress(asyncio.QueueFull):  # a reader filled it back up
                queue.put_nowait(frame)

    @asynccontextmanager
    async def subscribe(self, user_id: str, session_id: str) -> AsyncIterator[asyncio.Queue[str]]:
        """Attach to a session's frames for the life of the context."""
        key = (str(user_id), str(session_id))
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.setdefault(key, set()).add(queue)
        try:
            yield queue
        finally:
            watchers = self._subscribers.get(key)
            if watchers is not None:
                watchers.discard(queue)
                if not watchers:
                    del self._subscribers[key]


broker = FrameBroker()
