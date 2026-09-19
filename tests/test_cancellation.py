"""`run_to_completion`, the one wait-through-cancellation the runner uses."""

from __future__ import annotations

import asyncio

import pytest

from cancellation import run_to_completion

pytestmark = pytest.mark.asyncio


async def test_the_work_finishes_through_a_cancel_and_the_cancel_is_reported():
    gate = asyncio.Event()
    finished: list[str] = []

    async def work():
        await gate.wait()
        finished.append("done")

    waiter = asyncio.create_task(run_to_completion(work()))
    await asyncio.sleep(0.01)
    waiter.cancel()
    await asyncio.sleep(0.01)
    assert not waiter.done(), "the wait gave up on the work at the first cancel"
    gate.set()

    assert await waiter is True
    assert finished == ["done"]


async def test_an_uncancelled_wait_reports_no_cancel():
    """The control: True means a cancel was absorbed, not merely that the work finished."""

    async def work():
        await asyncio.sleep(0)

    assert await run_to_completion(work()) is False


async def test_a_failing_work_ends_the_wait_and_keeps_its_exception():
    async def work():
        raise RuntimeError("it broke")

    task = asyncio.ensure_future(work())
    assert await run_to_completion(task) is False
    assert isinstance(task.exception(), RuntimeError)


async def test_a_turn_that_fails_and_is_cancelled_while_recording_it_ends_cancelled(monkeypatch):
    """The runner half: a cancel absorbed while an internal error is recorded is put back.

    `_shielded` used to swallow it, so a turn torn down by a process coming down during
    that recording ended as though nobody had cancelled it.
    """
    from harness_module import runner

    gate = asyncio.Event()

    async def load(session_id):
        raise RuntimeError("the turn failed outside the loop")

    async def ending(session_id, sink, reason, **kw):
        await gate.wait()
        return True

    monkeypatch.setattr(runner, "load", load)
    monkeypatch.setattr(runner, "_ending", ending)
    turn = asyncio.create_task(runner._drive("s-1"))
    await asyncio.sleep(0.02)
    turn.cancel()
    await asyncio.sleep(0.02)
    gate.set()
    await asyncio.wait({turn})

    assert turn.cancelled(), "the cancel was swallowed and the turn ended as if nobody had asked"
