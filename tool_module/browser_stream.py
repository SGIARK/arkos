"""
Per-user broker for live browser screencast frames.

The browser tool pushes JPEG frames in; an SSE endpoint iterates `subscribe()`.
Queues are bounded and drop the oldest frame, so a slow client cannot block the
agent. Starting a session while one is running ends the old one first.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

logger = logging.getLogger(__name__)


_FrameEvent = dict[str, Any]


class BrowserStreamBroker:
    """In-memory fan-out from one producer per user to one consumer per user."""

    def __init__(self, queue_size: int = 16):
        self._queue_size = queue_size
        self._queues: dict[str, asyncio.Queue[_FrameEvent]] = {}
        self._active: set[str] = set()

    def _queue(self, user_id: str) -> asyncio.Queue[_FrameEvent]:
        q = self._queues.get(user_id)
        if q is None:
            q = asyncio.Queue(maxsize=self._queue_size)
            self._queues[user_id] = q
        return q

    def _put(self, user_id: str, event: _FrameEvent) -> None:
        q = self._queue(user_id)
        # Drop oldest on overflow so a slow consumer cannot stall the agent.
        while True:
            try:
                q.put_nowait(event)
                return
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    return

    def start_session(self, user_id: str) -> None:
        """Begin a screencast session for `user_id`, ending any active one first."""
        if user_id in self._active:
            self._put(user_id, {"type": "ended"})
        self._active.add(user_id)
        self._put(user_id, {"type": "started"})

    def push_frame(self, user_id: str, jpeg_b64: str) -> None:
        """Push one base64-encoded JPEG frame; a no-op if no session is active."""
        if user_id not in self._active:
            return
        self._put(user_id, {"type": "frame", "jpeg_b64": jpeg_b64})

    def end_session(self, user_id: str) -> None:
        """End the active screencast session; idempotent."""
        if user_id not in self._active:
            return
        self._active.discard(user_id)
        self._put(user_id, {"type": "ended"})

    async def subscribe(self, user_id: str) -> AsyncIterator[_FrameEvent]:
        """Yield events for `user_id` until the consumer stops iterating; no replay of old frames."""
        q = self._queue(user_id)
        while True:
            event = await q.get()
            yield event


broker = BrowserStreamBroker()
