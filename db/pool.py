"""Async Postgres access: one asyncpg pool per EVENT LOOP, shared by every session.

Per loop, not per process, and the distinction is the whole of this module's one
subtlety. An asyncpg pool binds its connections' futures to the loop that created
it, so the same pool used from a second loop raises "got Future attached to a
different loop" from inside the protocol, nowhere near the code that did it. In
production there is one loop and the distinction never shows. Under pytest there
is a fresh loop per test, and a pool surviving into the next one is F-1.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

import asyncpg

from config_module.loader import cfg as _cfg
from config_module.loader import config

_pool: asyncpg.Pool | None = None
_pool_loop: asyncio.AbstractEventLoop | None = None
_lock = asyncio.Lock()


def _abandon_if_from_another_loop() -> None:
    """Drop a pool built on a loop that is not the one running now.

    ABANDONED, NOT CLOSED. Closing it would await on the dead loop's transports
    and raise "Event loop is closed" from inside whatever was unlucky enough to
    ask for a connection. The sockets go when the loop that owns them is
    collected; what matters here is that nobody hands out its connections again.
    """
    global _pool, _pool_loop
    if _pool is None:
        return
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:  # no loop at all; nothing to compare against
        return
    if _pool_loop is not None and _pool_loop is not running:
        _pool = None
        _pool_loop = None


async def pool() -> asyncpg.Pool:
    """Return this loop's pool, creating it on first use."""
    global _pool, _pool_loop
    _abandon_if_from_another_loop()
    if _pool is None:
        # Double-checked: two coroutines creating two pools leaks the loser's connections.
        async with _lock:
            _abandon_if_from_another_loop()
            if _pool is None:
                _pool = await asyncpg.create_pool(
                    dsn=config.get("database.url"),
                    min_size=int(_cfg("database.pool_min_size", 1)),
                    max_size=int(_cfg("database.pool_max_size", 10)),
                    # Required: Supabase's transaction pooler breaks prepared statements.
                    statement_cache_size=0,
                    init=_register_json,
                )
                _pool_loop = asyncio.get_running_loop()
    return _pool


async def _register_json(conn: asyncpg.Connection) -> None:
    """Decode json and jsonb columns to Python objects instead of raw text."""
    for typename in ("json", "jsonb"):
        await conn.set_type_codec(
            typename,
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )


async def close() -> None:
    """Close the pool and clear it.

    THE REFERENCE IS CLEARED EVEN IF THE CLOSE FAILS. A close that raises and
    leaves the pool in place is the worst of both: callers keep being handed a
    pool somebody has already decided is finished with. That happened, and it is
    the second half of F-1: the recovery path raised from inside an `except`,
    so a retry that should have built a fresh pool never ran.
    """
    global _pool, _pool_loop
    doomed, _pool, _pool_loop = _pool, None, None
    if doomed is not None:
        with contextlib.suppress(Exception):
            await doomed.close()


async def fetch(query: str, *args: Any) -> list[asyncpg.Record]:
    return await (await pool()).fetch(query, *args)


async def fetchrow(query: str, *args: Any) -> asyncpg.Record | None:
    return await (await pool()).fetchrow(query, *args)


async def fetchval(query: str, *args: Any) -> Any:
    return await (await pool()).fetchval(query, *args)


async def execute(query: str, *args: Any) -> str:
    return await (await pool()).execute(query, *args)
