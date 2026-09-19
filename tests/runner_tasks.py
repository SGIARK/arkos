"""Clearing the runner's global task registries between tests.

`runner._running` and `runner._reapers` are process-global; a test's event loop
is not. So a task a previous test created outlives the loop that could cancel
it, and calling `cancel()` on one raises "Event loop is closed" out of
`call_soon`, surfacing as a teardown error in whatever test happens to run next.
That is the second half of F-1, the half the per-loop pool fix did not reach.

A task whose loop is gone is not running. It needs forgetting, not cancelling.
"""

from __future__ import annotations

import asyncio
import contextlib

from harness_module import runner


def cancel_and_forget() -> None:
    """Cancel what this loop can, forget the rest, and always clear."""
    for task in list(runner._reapers) + list(runner._running.values()):
        loop = None
        with contextlib.suppress(Exception):  # too dead to answer needs no cancelling
            loop = task.get_loop()
        # `is_closed` rather than "is it my loop": a task on another LIVE loop
        # is somebody else's to cancel, and cancelling it from here would be the
        # cross-loop call this function exists to prevent.
        if loop is not None and not loop.is_closed() and loop is _running_loop():
            task.cancel()
    runner._running.clear()
    runner._reapers.clear()
    runner._teardown.clear()


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None
