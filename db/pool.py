"""
The asyncpg pool: one per process, shared by every session.

The psycopg2 helpers elsewhere in the repo are synchronous, and one sync DB call
in an async path freezes every user on the single event loop. New code uses this
and nothing else.

`statement_cache_size=0` is not optional against Supabase's transaction pooler:
prepared statements do not survive a connection being handed to another client.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import asyncpg

from config_module.loader import config

_pool: asyncpg.Pool | None = None
_lock = asyncio.Lock()


def _cfg(key: str, default: Any) -> Any:
    value = config.get(key)
    return default if value is None else value


async def pool() -> asyncpg.Pool:
    """The pool, created on first use. Safe to call from anywhere."""
    global _pool
    if _pool is None:
        # Double-checked: two coroutines can both miss the first read, and
        # creating two pools leaks every connection in the loser.
        async with _lock:
            if _pool is None:
                _pool = await asyncpg.create_pool(
                    dsn=config.get("database.url"),
                    min_size=int(_cfg("database.pool_min_size", 1)),
                    max_size=int(_cfg("database.pool_max_size", 10)),
                    statement_cache_size=0,
                    init=_register_json,
                )
    return _pool


async def _register_json(conn: asyncpg.Connection) -> None:
    """Hand back dicts/lists for json columns instead of raw text."""
    for typename in ("json", "jsonb"):
        await conn.set_type_codec(
            typename,
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )


async def close() -> None:
    """Drop the pool. Called at shutdown, and by tests between cases."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def fetch(query: str, *args: Any) -> list[asyncpg.Record]:
    return await (await pool()).fetch(query, *args)


async def fetchrow(query: str, *args: Any) -> asyncpg.Record | None:
    return await (await pool()).fetchrow(query, *args)


async def fetchval(query: str, *args: Any) -> Any:
    return await (await pool()).fetchval(query, *args)


async def execute(query: str, *args: Any) -> str:
    return await (await pool()).execute(query, *args)
