"""Gate for tests that need the live database."""

from __future__ import annotations

import asyncio

import pytest

from db import pool


async def require_db(attempts: int = 3, delay: float = 0.5) -> None:
    """Skip when there is no database; FAIL when there is one and it has no schema.

    THE MESSAGE USED TO PROMISE WHAT THE CHECK DID NOT DO. It said "needs the
    buddy database (migrations applied)" and tested `SELECT 1`, which is
    connectivity. In CI the service container answered, the gate passed, and
    ~500 tests ran against an empty schema and failed one by one instead of
    saying why once.

    REACHABLE-BUT-EMPTY FAILS RATHER THAN SKIPS, deliberately. Skipping would
    turn a misconfigured run GREEN with everything skipped, and a false green is
    the one outcome worse than no signal. An absent database is a machine without
    one; an empty database is a machine that was supposed to have a schema.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            await pool.fetchval("SELECT 1")
            if await pool.fetchval("SELECT to_regclass('public.sessions') IS NULL"):
                pytest.fail(
                    "the database is reachable and has NO SCHEMA. Run `python db/migrate.py` "
                    "against DB_URL. Failing rather than skipping, because a suite that skips "
                    "itself green is worse than one that goes red."
                )
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
    pytest.skip(f"needs a database: {last}")
