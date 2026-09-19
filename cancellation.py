"""Waiting for work to finish however often the waiter is cancelled. One implementation, one caller here."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable


async def run_to_completion(work: Awaitable[object]) -> bool:
    """Run `work` to its end even if this coroutine is cancelled meanwhile. True if a cancel was absorbed.

    `asyncio.shield` protects the TASK and not the await, so one cancel raises out of the
    await while the work carries on unobserved; looping on the shield is what makes the
    result always seen. An exception from the work ends the wait and stays on the task.

    THE CALLER MUST RE-RAISE ON TRUE. Absorbing a cancel and returning normally makes the
    rest of a teardown run as though nobody had cancelled it.
    """
    task = asyncio.ensure_future(work)
    absorbed = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            absorbed = True
        except Exception:  # noqa: BLE001 - the task keeps its exception for the caller to read
            break
    return absorbed
