"""The operational log: batched, best-effort, and never in the caller's way.

Requires a real Postgres with migration 0 applied.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

import pytest
import pytest_asyncio

from db import pool
from harness_module import system_log
from tests.dbgate import require_db

pytestmark = pytest.mark.asyncio

_seeded: list[uuid.UUID] = []


@pytest_asyncio.fixture(autouse=True)
async def _db():
    await require_db()
    system_log._queue.clear()
    yield
    await system_log.stop()
    system_log._queue.clear()
    await pool.execute("DELETE FROM system_events WHERE session_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM sessions WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM users WHERE id = ANY($1::uuid[])", _seeded)
    _seeded.clear()
    await pool.close()


async def _session() -> str:
    user_id = uuid.uuid4()
    await pool.execute("INSERT INTO users (id) VALUES ($1)", user_id)
    _seeded.append(user_id)
    session_id = await pool.fetchval(
        "INSERT INTO sessions (user_id, mode, status) VALUES ($1, 'attended', 'idle') RETURNING id",
        user_id,
    )
    _seeded.append(session_id)
    return str(session_id)


async def _rows(session_id: str) -> list:
    return await pool.fetch(
        "SELECT level, event, fields FROM system_events WHERE session_id = $1 ORDER BY id",
        uuid.UUID(session_id),
    )


async def test_a_record_reaches_the_table_with_its_fields():
    session_id = await _session()

    system_log.record("fold", session_id=session_id, ms=12, messages=4)
    written = await system_log.flush()
    rows = await _rows(session_id)

    assert written == 1
    assert (rows[0]["level"], rows[0]["event"]) == ("info", "fold")
    assert rows[0]["fields"] == {"ms": 12, "messages": 4}


async def test_recording_does_not_touch_the_database():
    """It is queued, so nothing on a request or token path waits for it."""
    session_id = await _session()
    queries = 0
    real_execute = pool.execute

    async def counting(*a, **kw):
        nonlocal queries
        queries += 1
        return await real_execute(*a, **kw)

    pool.execute = counting
    try:
        for _ in range(50):
            system_log.record("lease_wait", session_id=session_id, waited_ms=1)
    finally:
        pool.execute = real_execute

    assert queries == 0
    assert len(system_log._queue) == 50


async def test_one_write_carries_the_whole_batch():
    session_id = await _session()
    for i in range(25):
        system_log.record("terminal_retry", level="warn", session_id=session_id, attempt=i)

    written = await system_log.flush()

    assert written == 25
    assert len(await _rows(session_id)) == 25


async def test_a_failed_write_loses_the_line_and_nothing_else(monkeypatch):
    """A lost diagnostic is not a stopped run."""
    session_id = await _session()
    system_log.record("fold", session_id=session_id)

    async def boom(*a, **kw):
        raise RuntimeError("the database went away")

    monkeypatch.setattr(pool, "execute", boom)
    written = await system_log.flush()

    assert written == 0
    assert not system_log._queue, "a poison batch must not be retried forever"


async def test_a_record_with_no_session_is_kept():
    system_log.record("startup", level="info", detail="swept 0 sessions")

    written = await system_log.flush()

    assert written == 1
    await pool.execute("DELETE FROM system_events WHERE event = 'startup' AND session_id IS NULL")


async def test_an_unparseable_id_does_not_lose_the_record():
    system_log.record("fold", session_id="not-a-uuid")

    assert await system_log.flush() == 1
    await pool.execute("DELETE FROM system_events WHERE event = 'fold' AND session_id IS NULL")


async def test_recording_never_raises():
    class Awkward:
        def __repr__(self):
            raise ValueError("not even repr")

    system_log.record("odd", session_id=None, thing=Awkward())

    assert True, "record() returned rather than raising"


async def test_the_flusher_writes_without_being_asked(monkeypatch):
    session_id = await _session()
    monkeypatch.setattr(system_log, "_cfg", lambda key, default: 0.05 if "flush" in key else default)

    await system_log.start()
    system_log.record("fold", session_id=session_id, ms=1)
    await asyncio.sleep(0.4)

    # Written by the loop, without stop() having been called.
    assert len(await _rows(session_id)) == 1
    await system_log.stop()


async def test_stopping_writes_what_is_still_queued():
    """A shutdown must not drop the batch it is holding."""
    session_id = await _session()
    await system_log.start()
    system_log.record("fold", session_id=session_id, ms=2)

    await system_log.stop()

    assert len(await _rows(session_id)) == 1


async def test_a_write_in_flight_survives_the_shutdown(monkeypatch):
    """stop() asks the loop to finish rather than cancelling it mid-write."""
    session_id = await _session()
    monkeypatch.setattr(system_log, "_cfg", lambda key, default: 0.01 if "flush" in key else default)
    real_execute = pool.execute

    async def slow_execute(*a, **kw):
        await asyncio.sleep(0.3)
        return await real_execute(*a, **kw)

    await system_log.start()
    system_log.record("fold", session_id=session_id, ms=3)
    await asyncio.sleep(0.05)  # the loop has taken the batch and is writing

    monkeypatch.setattr(pool, "execute", slow_execute)
    await system_log.stop()
    monkeypatch.undo()

    assert len(await _rows(session_id)) == 1, "the in-flight batch was lost at shutdown"


async def test_prune_removes_what_has_aged_out():
    session_id = await _session()
    await pool.execute(
        """
        INSERT INTO system_events (ts, level, event, session_id, fields)
        VALUES (now() - interval '40 days', 'info', 'fold', $1, '{}')
        """,
        uuid.UUID(session_id),
    )
    system_log.record("fold", session_id=session_id)
    await system_log.flush()

    await system_log.prune()

    assert len(await _rows(session_id)) == 1, "prune took the recent record too"


async def test_a_logged_error_from_any_module_reaches_the_table():
    """Tool and library errors used to go only to stdout; now the table hears them."""
    await system_log.start()
    try:
        logging.getLogger("tool_module.envelope").error("grep failed", exc_info=RuntimeError("refused"))
    finally:
        await system_log.stop()

    rows = await pool.fetch(
        "SELECT level, event, fields FROM system_events WHERE event = $1 AND fields->>'logger' = $2",
        system_log.LOGGING_EVENT,
        "tool_module.envelope",
    )
    await pool.execute("DELETE FROM system_events WHERE event = $1 AND session_id IS NULL", system_log.LOGGING_EVENT)
    assert len(rows) == 1
    assert rows[0]["level"] == "error"
    assert rows[0]["fields"]["message"] == "grep failed"
    assert "RuntimeError: refused" in rows[0]["fields"]["traceback"]


async def test_a_warning_is_forwarded_and_info_is_not():
    await system_log.start()
    try:
        logging.getLogger("tool_module.browser").warning("lease took %dms", 90_000)
        logging.getLogger("tool_module.browser").info("hundreds of these per task")
    finally:
        levels = [(level, fields["message"]) for level, _, _, _, fields in system_log._queue]
        await system_log.stop()
    await pool.execute("DELETE FROM system_events WHERE event = $1 AND session_id IS NULL", system_log.LOGGING_EVENT)

    assert levels == [("warn", "lease took 90000ms")]


async def test_nothing_is_forwarded_once_stopped():
    await system_log.start()
    await system_log.stop()

    logging.getLogger("tool_module.envelope").error("after shutdown")

    assert not system_log._queue


async def test_a_failed_write_does_not_feed_itself(monkeypatch):
    """The module's own 'dropped' warning must not become a row whose write fails again."""
    await system_log.start()
    try:
        system_log.record("fold")

        async def boom(*a, **kw):
            raise RuntimeError("the database went away")

        monkeypatch.setattr(pool, "execute", boom)
        assert await system_log.flush() == 0
        assert not system_log._queue
    finally:
        monkeypatch.undo()
        await system_log.stop()
