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
            # `close` swallows its own failures and clears the reference either
            # way, which it did not always do: a recovery that raises from
            # inside an `except` turns a retryable blip into a fixture error,
            # and that was half of F-1.
            await pool.close()
            if attempt + 1 < attempts:
                await asyncio.sleep(delay)
    pytest.skip(f"needs the buddy database (migrations applied): {last}")
