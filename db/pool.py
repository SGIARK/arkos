"""Async Postgres access: one asyncpg pool per process, shared by every session."""

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
    """Return the process-wide pool, creating it on first use."""
    global _pool
    if _pool is None:
        # Double-checked: two coroutines creating two pools leaks the loser's connections.
        async with _lock:
            if _pool is None:
                _pool = await asyncpg.create_pool(
                    dsn=config.get("database.url"),
                    min_size=int(_cfg("database.pool_min_size", 1)),
                    max_size=int(_cfg("database.pool_max_size", 10)),
                    # Required: Supabase's transaction pooler breaks prepared statements.
                    statement_cache_size=0,
                    init=_register_json,
                )
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
    """Close the pool and clear it."""
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
