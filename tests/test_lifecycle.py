"""The one state machine: legal moves only, cancel wins, and every move leaves a row.

Runs against a real Postgres with migration 0 applied; each case owns its rows.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
import pytest_asyncio

from agent_module.events import DoneEvent, ToolCallEvent
from db import pool
from harness_module import lifecycle as lc
from harness_module import session_log as slog

pytestmark = pytest.mark.asyncio


_seeded: list[uuid.UUID] = []


@pytest_asyncio.fixture(autouse=True)
async def _db():
    try:
        await pool.fetchval("SELECT 1")
    except Exception as e:  # noqa: BLE001 - any connection failure means skip
        await pool.close()
        pytest.skip(f"needs the arkos database (migration 0 applied): {e}")
    yield
    # The sweep is global, so a session left `running` is swept by the next test
    # and by anything else sharing this database.
    await pool.execute("DELETE FROM sessions WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM users WHERE id = ANY($1::uuid[])", _seeded)
    _seeded.clear()
    await pool.close()


async def _session(status: str = "running", mode: str = "attended") -> str:
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


async def _row(session_id: str):
    return await pool.fetchrow(
        "SELECT status, mode, terminal_reason, ended_at FROM sessions WHERE id = $1",
        uuid.UUID(session_id),
    )


# --- the map ------------------------------------------------------------------


async def test_lifecycle_transitions():
    """Only moves in the ALLOWED map are legal, and a move from the wrong status changes nothing."""
    session_id = await _session(status="pending")

    assert await lc.transition(session_id, "pending", "running", "claimed")
    assert (await _row(session_id))["status"] == "running"

    # Not in ALLOWED: pending is not reachable from running.
    with pytest.raises(lc.IllegalTransition):
        await lc.transition(session_id, "running", "pending", "nope")

    # In ALLOWED, but the session is not in `expected`, so the UPDATE matches nothing.
    assert not await lc.transition(session_id, "idle", "running", "stale")
    assert (await _row(session_id))["status"] == "running"


async def test_cancel_wins_a_live_race():
    """Of two transitions out of one status, exactly one returns True."""
    session_id = await _session(status="running")

    cancelled, completed = await asyncio.gather(
        lc.transition(session_id, "running", "cancelled", "cancelled"),
        lc.transition(session_id, "running", "completed", "completed"),
    )

    assert cancelled != completed, "exactly one writer may win"
    assert (await _row(session_id))["status"] in ("cancelled", "completed")


# --- terminal bookkeeping -----------------------------------------------------


async def test_a_terminal_move_records_the_reason_verbatim():
    session_id = await _session(status="running")

    await lc.transition(session_id, "running", "failed", "max_hops")
    row = await _row(session_id)

    assert row["status"] == "failed"
    assert row["terminal_reason"] == "max_hops", "terminal_reason is the reason it was given"
    assert row["ended_at"] is not None


async def test_turn_end_leaves_no_terminal_marks():
    """A move to `idle` leaves terminal_reason and ended_at unset."""
    session_id = await _session(status="running")

    await lc.transition(session_id, "running", "idle", "turn_end")
    row = await _row(session_id)

    assert row["status"] == "idle"
    assert row["terminal_reason"] is None
    assert row["ended_at"] is None


async def test_reopening_a_terminal_session_clears_the_terminal_marks():
    session_id = await _session(status="running")
    await lc.transition(session_id, "running", "completed", "completed")

    assert await lc.transition(session_id, "completed", "running", "restarted by the human")
    row = await _row(session_id)

    assert row["terminal_reason"] is None
    assert row["ended_at"] is None


async def test_mode_flips_in_the_same_update_as_the_status():
    """A transition sets mode in the same update as status."""
    session_id = await _session(status="idle", mode="attended")

    await lc.transition(session_id, "idle", "running", "approved", mode="unattended")
    row = await _row(session_id)

    assert (row["status"], row["mode"]) == ("running", "unattended")


# --- the event ----------------------------------------------------------------


async def test_every_transition_appends_a_lifecycle_event():
    session_id = await _session(status="running")

    await lc.transition(session_id, "running", "idle", "turn_end")
    events = [e.event for e in await slog.get_events(session_id)]

    assert len(events) == 1
    assert (events[0].kind, events[0].from_, events[0].to, events[0].reason) == (
        "lifecycle",
        "running",
        "idle",
        "turn_end",
    )


async def test_a_lost_race_appends_nothing():
    session_id = await _session(status="idle")

    assert not await lc.transition(session_id, "running", "completed", "completed")
    assert await slog.get_events(session_id) == []


# --- done -> status -----------------------------------------------------------


async def test_status_for_buckets_every_done_reason():
    assert lc.status_for(DoneEvent(reason="turn_end")) == "idle"
    assert lc.status_for(DoneEvent(reason="completed")) == "completed"
    assert lc.status_for(DoneEvent(reason="cancelled")) == "cancelled"
    for reason in ("max_hops", "wall_clock", "model_error", "context_overflow", "interrupted"):
        assert lc.status_for(DoneEvent(reason=reason)) == "failed"


# --- the startup sweep --------------------------------------------------------


async def test_the_sweep_fails_a_running_session_and_says_why_in_the_transcript():
    session_id = await _session(status="running")
    await slog.append(session_id, ToolCallEvent(id="c1", name="run_command", args={}))

    await lc.sweep_interrupted()
    row = await _row(session_id)
    kinds = [e.event.kind for e in await slog.get_events(session_id)]

    assert (row["status"], row["terminal_reason"]) == ("failed", "interrupted")
    # The dangling call is closed ahead of the done event.
    assert kinds == ["tool_call", "tool_result", "done", "lifecycle"]


async def test_the_sweep_leaves_idle_conversations_alone():
    """The sweep leaves an `idle` session at `idle`."""
    session_id = await _session(status="idle")

    await lc.sweep_interrupted()

    assert (await _row(session_id))["status"] == "idle"
