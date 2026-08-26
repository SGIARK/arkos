"""Live fan-out to subscribers: session events, and per-user attention signals.

The log in Postgres is the record; this pushes only. A subscriber whose queue overflows
receives the LAGGED sentinel and re-reads from the log after its last seq.

TWO CHANNELS, one rule. A session's events fan out per session, and attention
fans out per USER — and both are published from the code that writes the row,
never relayed by a bystander. That rule is the whole point of the second
channel: account-level attention used to be driven by a `pulse` that only a
MOUNTED session window could bump, so parking a call while sitting on the desk
published to a stream nobody was reading and the waiting list stayed frozen.
State has to announce itself from where it is written.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class AttentionSignal:
    """A nudge, not a payload: "your waiting list changed, read it again".

    Deliberately carries no approval row. The list is a query at three scopes
    and the client already knows how to ask; shipping the row here would mean
    two ways to learn the same fact, which drift. `reason` and `session_id` are
    for the log and for a client that wants to know whether the change was in
    the window it is looking at.
    """

    reason: str
    session_id: str


class UserStream:
    """In-memory fan-out keyed by user: one publisher per user, any subscribers.

    Separate from `SessionStream` rather than a mode of it, because the two
    carry different things — an ordered log with a seq a reader resumes from,
    versus a signal with no history worth replaying. A missed nudge costs one
    stale list until the next one; a missed event costs a hole in a transcript.
    """

    def __init__(self, queue_size: int = 64):
        self._queue_size = queue_size
        self._subscribers: dict[str, set[asyncio.Queue[AttentionSignal]]] = {}

    def publish(self, user_id: str, signal: AttentionSignal) -> None:
        """Nudge every subscriber of one user. Never blocks, never raises.

        A full queue is DROPPED rather than sentinelled: the message is "read
        the list again", so an older copy of the same instruction is worth
        nothing, and the reader is about to refetch anyway.
        """
        if not user_id:
            return
        for queue in list(self._subscribers.get(user_id, ())):
            try:
                queue.put_nowait(signal)
            except asyncio.QueueFull:
                logger.debug("attention queue full for %s; dropping a nudge", user_id)

    @asynccontextmanager
    async def subscribe(self, user_id: str) -> AsyncIterator[asyncio.Queue[AttentionSignal]]:
        """Attach to one user's attention signals for the life of the context."""
        queue: asyncio.Queue[AttentionSignal] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.setdefault(user_id, set()).add(queue)
        try:
            yield queue
        finally:
            subscribers = self._subscribers.get(user_id)
            if subscribers is not None:
                subscribers.discard(queue)
                if not subscribers:
                    del self._subscribers[user_id]

    def subscriber_count(self, user_id: str) -> int:
        """For tests, and for a log line that answers "is anyone listening"."""
        return len(self._subscribers.get(user_id, ()))


def _drain(queue: asyncio.Queue[Item]) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return


stream = SessionStream()
attention = UserStream()
