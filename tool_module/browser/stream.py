"""The frame side-channel: what the browser is looking at, while it looks.

Frames are NOT events. They are never appended, never replayed, and carry no
seq: a video of a run that finished is worth nothing, and putting JPEGs in the
transcript would make the log unreadable and enormous. What reaches the log is a
`status` event carrying this stream's URL, so the UI mounts the pane from the
event stream and drops it when the run ends.

Keyed by `(user_id, session_id)`. The user half is the ownership check's
business; the session half is what the old implementation got wrong — two of a
user's sessions driving browsers at once shared one queue and clobbered each
other's frames.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Frames held for a subscriber that is not keeping up. Small on purpose: video
# is only worth watching live, so a slow reader should see the newest frame
# rather than a backlog of stale ones.
_QUEUE_SIZE = 4

Key = tuple[str, str]


class FrameBroker:
    """In-memory fan-out of JPEG frames, per (user, session)."""

    def __init__(self, queue_size: int = _QUEUE_SIZE):
        self._subscribers: dict[Key, set[asyncio.Queue[str]]] = {}
        self._queue_size = queue_size

    def publish(self, user_id: str, session_id: str, frame: str) -> None:
        """Hand one base64 JPEG to every viewer. Never blocks, never raises.

        A full queue drops its oldest frame rather than the new one: the newest
        frame is the only one anybody wants.
        """
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

    def watching(self, user_id: str, session_id: str) -> bool:
        """Whether anyone is looking, so a run does not pay for frames nobody sees."""
        return bool(self._subscribers.get((str(user_id), str(session_id))))


broker = FrameBroker()
