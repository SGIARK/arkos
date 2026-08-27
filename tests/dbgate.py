"""Gate for tests that need the live database."""

from __future__ import annotations

import asyncio

import pytest

from db import pool


async def require_db(attempts: int = 3, delay: float = 0.5) -> None:
    """Skip the calling module unless the database answers."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            await pool.fetchval("SELECT 1")
            return
        except Exception as e:  # noqa: BLE001 - any failure is worth one more try
            last = e
            await pool.close()
            if attempt + 1 < attempts:
                await asyncio.sleep(delay)
    pytest.skip(f"needs the arkos database (migrations applied): {last}")
