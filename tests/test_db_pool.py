"""The pool is per EVENT LOOP, not per process.

This is F-1, which errored intermittently for days and whose traceback took
around ten attempts to catch. An asyncpg pool binds its connections' futures to
the loop that made it, so reusing one from a second loop raises "got Future
attached to a different loop" out of the protocol, nowhere near the caller. In
production there is one loop and it never shows; under pytest there is a loop per
test, so a pool that outlives its loop poisons whatever runs next.

These need no database: the failure is about which loop owns an object, and that
is decided before a connection is ever made.
"""

from __future__ import annotations

import asyncio

from db import pool as db_pool


def test_a_pool_from_a_dead_loop_is_not_handed_out_again() -> None:
    """The whole bug in one assertion."""
    sentinel = object()

    async def first() -> None:
        db_pool._pool = sentinel  # type: ignore[assignment]
        db_pool._pool_loop = asyncio.get_running_loop()

    asyncio.run(first())
    assert db_pool._pool is sentinel, "the setup did not take; this would pass vacuously"

    async def second() -> object | None:
        db_pool._abandon_if_from_another_loop()
        return db_pool._pool

    try:
        assert asyncio.run(second()) is None
    finally:
        db_pool._pool = db_pool._pool_loop = None


def test_a_pool_from_this_loop_is_kept() -> None:
    """The other half: abandoning unconditionally would make the cache pointless."""
    sentinel = object()

    async def same_loop() -> object | None:
        db_pool._pool = sentinel  # type: ignore[assignment]
        db_pool._pool_loop = asyncio.get_running_loop()
        db_pool._abandon_if_from_another_loop()
        return db_pool._pool

    try:
        assert asyncio.run(same_loop()) is sentinel
    finally:
        db_pool._pool = db_pool._pool_loop = None


def test_close_clears_the_reference_even_when_closing_fails() -> None:
    """The second half of F-1.

    A close that raises and leaves the pool in place hands callers a pool
    somebody has already decided is finished with, and a recovery path that
    raises from inside an `except` turns a retryable blip into a fixture error.
    """

    class Stubborn:
        async def close(self) -> None:
            raise RuntimeError("Event loop is closed")

    async def go() -> None:
        db_pool._pool = Stubborn()  # type: ignore[assignment]
        db_pool._pool_loop = asyncio.get_running_loop()
        await db_pool.close()  # must not raise

    try:
        asyncio.run(go())
        assert db_pool._pool is None
    finally:
        db_pool._pool = db_pool._pool_loop = None
