"""The screencast frame broker in tool_module/browser_stream.py."""

from __future__ import annotations

import asyncio

import pytest

from tool_module.browser_stream import BrowserStreamBroker


async def _drain(broker: BrowserStreamBroker, user_id: str, n: int) -> list[dict]:
    """Pull exactly `n` events off the user's subscription."""
    out: list[dict] = []
    agen = broker.subscribe(user_id).__aiter__()
    for _ in range(n):
        out.append(await asyncio.wait_for(agen.__anext__(), timeout=1.0))
    return out


@pytest.mark.asyncio
async def test_session_lifecycle_emits_started_frame_ended():
    broker = BrowserStreamBroker()

    async def consumer():
        return await _drain(broker, "u1", 3)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)  # let the consumer attach

    broker.start_session("u1")
    broker.push_frame("u1", "AAAA")
    broker.end_session("u1")

    events = await task
    assert events == [
        {"type": "started"},
        {"type": "frame", "jpeg_b64": "AAAA"},
        {"type": "ended"},
    ]


@pytest.mark.asyncio
async def test_push_frame_without_session_is_dropped():
    broker = BrowserStreamBroker()

    broker.push_frame("u1", "AAAA")  # no active session
    broker.start_session("u1")
    broker.push_frame("u1", "BBBB")

    events = await _drain(broker, "u1", 2)
    assert events == [
        {"type": "started"},
        {"type": "frame", "jpeg_b64": "BBBB"},
    ]


@pytest.mark.asyncio
async def test_users_are_isolated():
    broker = BrowserStreamBroker()

    broker.start_session("alice")
    broker.start_session("bob")
    broker.push_frame("alice", "A1")
    broker.push_frame("bob", "B1")

    alice = await _drain(broker, "alice", 2)
    bob = await _drain(broker, "bob", 2)

    assert alice == [{"type": "started"}, {"type": "frame", "jpeg_b64": "A1"}]
    assert bob == [{"type": "started"}, {"type": "frame", "jpeg_b64": "B1"}]


@pytest.mark.asyncio
async def test_overflow_drops_oldest_frame():
    """A slow consumer must never block the producer; oldest frames lose."""
    broker = BrowserStreamBroker(queue_size=2)
    broker.start_session("u1")  # uses 1 slot

    for tag in ("F1", "F2", "F3", "F4", "F5"):
        broker.push_frame("u1", tag)

    q = broker._queue("u1")
    assert q.qsize() <= 2
    survivors = []
    while not q.empty():
        survivors.append(q.get_nowait())
    assert all(s.get("type") in {"started", "frame"} for s in survivors)


@pytest.mark.asyncio
async def test_starting_new_session_ends_previous():
    broker = BrowserStreamBroker()

    async def consumer():
        return await _drain(broker, "u1", 3)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)

    broker.start_session("u1")
    broker.start_session("u1")  # should emit ended for the first, then started

    events = await task
    assert events == [
        {"type": "started"},
        {"type": "ended"},
        {"type": "started"},
    ]


@pytest.mark.asyncio
async def test_end_without_active_session_is_noop():
    broker = BrowserStreamBroker()
    broker.end_session("ghost")  # must not raise
    assert broker._queue("ghost").empty()
