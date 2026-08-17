"""The transcript: seq ordering under concurrency, the tool-call invariant, blobs, redaction.

Runs against a real Postgres with migration 0 applied; each case owns its rows.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
import pytest_asyncio

from agent_module.events import ContentEvent, DoneEvent, ToolCallEvent, ToolResultEvent, UserEvent
from db import pool
from harness_module import session_log as slog

pytestmark = pytest.mark.asyncio


_seeded: list[uuid.UUID] = []


@pytest_asyncio.fixture(autouse=True)
async def _db():
    """Skip the module unless the real schema is reachable."""
    try:
        await pool.fetchval("SELECT 1")
    except Exception as e:  # noqa: BLE001 - any connection failure means skip
        await pool.close()
        pytest.skip(f"needs the arkos database (migration 0 applied): {e}")
    yield
    # A session left `running` is one the startup sweep will fail, here and in
    # any other process sharing this database.
    await pool.execute("DELETE FROM sessions WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM users WHERE id = ANY($1::uuid[])", _seeded)
    _seeded.clear()
    # Each test gets its own event loop, and a pool outliving its loop is dead.
    await pool.close()


async def _session(mode: str = "attended", status: str = "running") -> str:
    """Seed a user and a session, and return the session id."""
    user_id = uuid.uuid4()
    await pool.execute("INSERT INTO users (id) VALUES ($1)", user_id)
    _seeded.append(user_id)
    session_id = await pool.fetchval(
        "INSERT INTO sessions (user_id, mode, status) VALUES ($1, $2, $3) RETURNING id",
        user_id,
        mode,
        status,
    )
    return str(session_id)


# --- ordering ----------------------------------------------------------------


async def test_append_returns_the_seq_and_reads_back_the_same_event():
    session_id = await _session()

    stored = await slog.append(session_id, ContentEvent(text="hello"))
    events = await slog.get_events(session_id)

    assert stored.seq > 0
    assert [e.seq for e in events] == [stored.seq]
    assert events[0].event == ContentEvent(text="hello")


async def test_append_ordering_under_concurrency():
    """A slow writer commits first, so a live reader can never skip its seq.

    BIGSERIAL assigns before commit. Without the advisory lock the fast writer
    takes the higher seq and commits while the slow one is still open, and a
    reader that already sent the higher one never emits the lower.
    """
    session_id = await _session()
    seen: list[int] = []
    reading = True

    async def slow_writer() -> int:
        async with (await pool.pool()).acquire() as conn, conn.transaction():
            stored = await slog.append_tx(conn, session_id, UserEvent(text="slow"))
            await asyncio.sleep(0.3)
            return stored.seq

    async def fast_writer() -> int:
        await asyncio.sleep(0.05)
        return (await slog.append(session_id, UserEvent(text="fast"))).seq

    async def reader() -> None:
        cursor = 0
        while reading:
            for stored in await slog.get_events(session_id, after_seq=cursor):
                seen.append(stored.seq)
                cursor = stored.seq
            await asyncio.sleep(0.02)

    reader_task = asyncio.create_task(reader())
    slow_seq, fast_seq = await asyncio.gather(slow_writer(), fast_writer())
    await asyncio.sleep(0.1)
    reading = False
    await reader_task

    assert slow_seq < fast_seq, "the blocked writer must not be handed the lower seq"
    assert seen == sorted(seen), "the reader saw events out of order"
    assert set(seen) == {slow_seq, fast_seq}, f"the reader skipped a seq: saw {seen}"


async def test_get_events_reads_after_a_cursor_and_recent_reads_the_tail():
    session_id = await _session()
    seqs = [(await slog.append(session_id, ContentEvent(text=str(i)))).seq for i in range(5)]

    after = await slog.get_events(session_id, after_seq=seqs[2])
    tail = await slog.recent_events(session_id, limit=2)

    assert [e.seq for e in after] == seqs[3:]
    assert [e.seq for e in tail] == seqs[-2:], "recent_events must return the tail in seq order"


# --- the transcript invariant -------------------------------------------------


async def test_transcript_invariant():
    """Every tool_call.id is closed by exactly one tool_result, before the run ends."""
    session_id = await _session()
    await slog.append(session_id, ToolCallEvent(id="c1", name="grep", args={}))

    with pytest.raises(slog.TranscriptError):
        await slog.append(session_id, DoneEvent(reason="turn_end"))

    await slog.append(session_id, ToolResultEvent(id="c1", ok=True, content="found"))
    with pytest.raises(slog.TranscriptError):
        await slog.append(session_id, ToolResultEvent(id="c1", ok=True, content="found twice"))

    # Closed, so the run may now end.
    await slog.append(session_id, DoneEvent(reason="turn_end"))


async def test_a_result_for_a_call_that_was_never_appended_is_refused():
    session_id = await _session()

    with pytest.raises(slog.TranscriptError):
        await slog.append(session_id, ToolResultEvent(id="ghost", ok=True, content="x"))


async def test_close_dangling_synthesizes_the_result_nobody_was_alive_to_write():
    session_id = await _session()
    await slog.append(session_id, ToolCallEvent(id="c1", name="run_command", args={}))
    await slog.append(session_id, ToolCallEvent(id="c2", name="read_file", args={}))
    await slog.append(session_id, ToolResultEvent(id="c2", ok=True, content="done"))

    closed = await slog.close_dangling(session_id)

    assert [c.event.id for c in closed] == ["c1"], "only the open call is closed"
    assert closed[0].event.error_kind == "interrupted"
    assert not closed[0].event.ok
    # And the run can now end.
    await slog.append(session_id, DoneEvent(reason="cancelled"))


async def test_a_finished_run_does_not_constrain_the_next_one():
    """The invariant checks read from the last `done`, not from the start of time."""
    session_id = await _session()
    await slog.append(session_id, ToolCallEvent(id="c1", name="grep", args={}))
    await slog.append(session_id, ToolResultEvent(id="c1", ok=True, content="x"))
    await slog.append(session_id, DoneEvent(reason="turn_end"))

    await slog.append(session_id, UserEvent(text="again"))
    await slog.append(session_id, DoneEvent(reason="turn_end"))


# --- redaction ----------------------------------------------------------------


async def test_secrets_in_tool_args_never_reach_the_log():
    session_id = await _session()
    args = {
        "url": "https://example.com",
        "headers": {"Authorization": "Bearer sk-live-123"},
        "api_key": "sk-live-456",
        "nested": [{"password": "hunter2"}],
    }

    stored = await slog.append(session_id, ToolCallEvent(id="c1", name="http", args=args))
    payload = await pool.fetchval("SELECT payload FROM session_events WHERE seq = $1", stored.seq)

    assert payload["args"]["url"] == "https://example.com"
    assert payload["args"]["headers"]["Authorization"] == "[redacted]"
    assert payload["args"]["api_key"] == "[redacted]"
    assert payload["args"]["nested"][0]["password"] == "[redacted]"
    assert "sk-live-123" not in str(payload)
    assert "hunter2" not in str(payload)


# --- blobs --------------------------------------------------------------------


async def test_a_blob_round_trips_and_slices():
    session_id = await _session()
    ref = await slog.save_blob(session_id, "abcdefghij")

    assert await slog.read_blob(ref) == "abcdefghij"
    assert await slog.read_blob(ref, offset=3, limit=4) == "defg"


async def test_a_blob_is_unreadable_by_another_user():
    """Unguessable is not access control: refs are ownership-checked too."""
    mine = await _session()
    ref = await slog.save_blob(mine, "secret")
    other_user = await pool.fetchval("SELECT user_id FROM sessions WHERE id = $1", uuid.UUID(await _session()))

    assert await slog.read_blob(ref, user_id=str(other_user)) is None


async def test_an_unparseable_ref_is_a_miss_not_a_crash():
    assert await slog.read_blob("not-a-uuid") is None


# --- id reuse across runs -----------------------------------------------------


async def test_an_earlier_runs_result_does_not_close_a_later_call_with_the_same_id():
    """A model that repeats a tool_call id must not brick the session."""
    session_id = await _session()
    await slog.append(session_id, ToolCallEvent(id="call_1", name="grep", args={}))
    await slog.append(session_id, ToolResultEvent(id="call_1", ok=True, content="first"))
    await slog.append(session_id, DoneEvent(reason="turn_end"))

    # A second run reuses the id. The result must still close it.
    await slog.append(session_id, ToolCallEvent(id="call_1", name="grep", args={}))
    stored = await slog.append(session_id, ToolResultEvent(id="call_1", ok=True, content="second"))

    assert stored.seq > 0
    await slog.append(session_id, DoneEvent(reason="turn_end"))


async def test_a_reused_id_is_still_closable_by_the_sweep():
    session_id = await _session()
    await slog.append(session_id, ToolCallEvent(id="call_1", name="grep", args={}))
    await slog.append(session_id, ToolResultEvent(id="call_1", ok=True, content="first"))
    await slog.append(session_id, DoneEvent(reason="turn_end"))
    await slog.append(session_id, ToolCallEvent(id="call_1", name="grep", args={}))

    closed = await slog.close_dangling(session_id)

    assert [c.event.id for c in closed] == ["call_1"]


async def test_the_advisory_lock_keys_off_one_canonical_id():
    """Postgres compares uuids case-insensitively but hashes text exactly."""
    session_id = await _session()
    upper = session_id.upper()

    async def append_as(spelling: str, text: str) -> int:
        return (await slog.append(spelling, ContentEvent(text=text))).seq

    first, second = await asyncio.gather(append_as(session_id, "lower"), append_as(upper, "upper"))
    events = await slog.get_events(session_id)

    assert {first, second} == {e.seq for e in events}, "both spellings must write to one session"
    assert [e.seq for e in events] == sorted(e.seq for e in events)
