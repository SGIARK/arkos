"""Stop is not cancel: a held run keeps its plan (11.8.6).

The rule this file exists to pin: the run control has two faces. Stop closes the
calls in flight and holds the turn at its next hop boundary — no `done`, no
terminal, and NO MODE FLIP — so the plan the run was approved from is still
approved and resuming costs nothing. Cancel is the second press.

What it replaced: `POST /cancel` was the only control, and it is `task.cancel()`
on the whole turn. Stopping one slow browser step therefore wrote
`done{cancelled}`, flipped the mode back to attended, and spent an approved plan
— which happened on the plan gate's first day of use, 2026-08-20.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from agent_module import loop as lp
from db import pool
from harness_module import api, approvals, lifecycle, runner, store
from harness_module import session_log as slog
from tests.dbgate import require_db
from tests.test_api import _supabase_token
from tool_module.envelope import ResultEnvelope, ToolSpec

_seeded: list[uuid.UUID] = []


@pytest_asyncio.fixture(autouse=True)
async def _db(tmp_path):
    await require_db()
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    yield
    for task in list(runner._reapers) + list(runner._running.values()) + list(runner._stop_backstops):
        task.cancel()
    runner._running.clear()
    runner._reapers.clear()
    runner._sinks.clear()
    runner._stop_backstops.clear()
    runner._cancelling.clear()
    await asyncio.sleep(0)
    store.use_blobs(None)
    await pool.execute("DELETE FROM sessions WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM projects WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM users WHERE id = ANY($1::uuid[])", _seeded)
    _seeded.clear()
    await pool.close()


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=api.app)
    async with AsyncClient(transport=transport, base_url="https://testserver") as c:
        yield c


async def _user(client: AsyncClient) -> str:
    user_id = str(uuid.uuid4())
    _seeded.append(uuid.UUID(user_id))
    response = await client.post(
        "/auth/session", headers={"Authorization": f"Bearer {_supabase_token(user_id)}"}
    )
    assert response.status_code == 204
    return user_id


async def _session(user_id: str, *, status: str = "running", mode: str = "unattended") -> str:
    return str(
        await pool.fetchval(
            "INSERT INTO sessions (user_id, mode, status, title) VALUES ($1, $2, $3, 'a run') RETURNING id",
            uuid.UUID(user_id),
            mode,
            status,
        )
    )


async def _held(session_id: str) -> approvals.Approval:
    """Put a session where a stop leaves it: parked on a `resume` row, mode intact."""
    row = await approvals.create(session_id, f"stop_{uuid.uuid4().hex[:12]}", "resume", runner._STOP_PROMPT)
    await pool.execute(
        "UPDATE sessions SET status = 'awaiting_approval' WHERE id = $1", uuid.UUID(session_id)
    )
    return row


# --- the loop half: a stopped call is not a failed tool -------------------------


def test_a_stopped_call_does_not_spend_the_tool_s_attempts():
    """Stopping the browser three times must not close the browser to the run."""
    state = lp._State(budgets=lp.Budgets.load("unattended"))
    call = lp._PartialCall(id="c1", name="browser_task")
    stopped = runner._stopped_envelope("browser_task")

    for _ in range(5):
        asyncio.run(_settle(call, stopped, state))

    assert state.failures.get("browser_task", 0) == 0
    assert stopped.error_kind == lp.CANCELLED_BY_USER


def test_a_genuinely_failed_call_still_spends_them():
    state = lp._State(budgets=lp.Budgets.load("unattended"))
    call = lp._PartialCall(id="c1", name="browser_task")
    failed = ResultEnvelope(ok=False, content="boom", error_kind="upstream_error")

    asyncio.run(_settle(call, failed, state))

    assert state.failures["browser_task"] == 1


async def _settle(call, envelope, state):
    return await lp._settle(call, envelope, state, [])


# --- the runner half: stop closes calls and refuses the rest of the hop ---------


@pytest.mark.asyncio
async def test_stop_closes_the_call_in_flight_and_refuses_the_next(monkeypatch):
    """The two halves of "stop": what was running, and what would have run next."""
    user_id = str(uuid.uuid4())
    _seeded.append(uuid.UUID(user_id))
    await pool.execute("INSERT INTO users (id) VALUES ($1)", uuid.UUID(user_id))
    session_id = await _session(user_id)
    sink = runner._Sink(await runner.load(session_id))
    running = asyncio.Event()

    async def slow(name: str, args: dict) -> ResultEnvelope:
        running.set()
        await asyncio.sleep(30)
        raise AssertionError("the stop did not reach the call")

    dispatch = sink.write_ahead(slow, [ToolSpec(name="browser_task", readonly=True)])
    task = asyncio.create_task(dispatch("browser_task", {}))
    await asyncio.wait_for(running.wait(), timeout=5)

    assert sink.request_stop() == 1
    envelope = await asyncio.wait_for(task, timeout=5)

    assert envelope.ok is False
    assert envelope.error_kind == lp.CANCELLED_BY_USER
    assert "stopped the run" in envelope.content
    assert sink.stopping is True

    # Anything the hop had queued behind it never runs.
    after = await dispatch("browser_task", {})
    assert after.error_kind == lp.CANCELLED_BY_USER

    await sink.close(runner.DoneEvent(reason="cancelled"))


@pytest.mark.asyncio
async def test_the_hold_keeps_the_mode_and_writes_no_terminal():
    """A park is not a terminal. That is the whole of why the plan survives."""
    user_id = str(uuid.uuid4())
    _seeded.append(uuid.UUID(user_id))
    await pool.execute("INSERT INTO users (id) VALUES ($1)", uuid.UUID(user_id))
    session_id = await _session(user_id)
    sink = runner._Sink(await runner.load(session_id))
    sink.request_stop()

    assert await sink.park_stopped() is True

    row = await pool.fetchrow(
        "SELECT status, mode, terminal_reason FROM sessions WHERE id = $1", uuid.UUID(session_id)
    )
    assert (row["status"], row["mode"]) == ("awaiting_approval", "unattended")
    assert row["terminal_reason"] is None
    events = [e.event for e in await slog.get_events(session_id)]
    assert [e.kind for e in events if e.kind == "done"] == [], "a hold ends nothing"

    open_rows = await approvals.open_for(session_id)
    assert [r.kind for r in open_rows] == ["resume"]
    assert open_rows[0].is_resume and open_rows[0].tool_name is None


@pytest.mark.asyncio
async def test_stop_refuses_a_session_that_is_not_running(client):
    user_id = await _user(client)
    session_id = await _session(user_id, status="idle")

    response = await client.post(f"/sessions/{session_id}/stop")

    assert response.status_code == 409
    assert response.json()["code"] == "not_running"


@pytest.mark.asyncio
async def test_stop_reports_when_there_is_no_live_turn_to_hold(client):
    """`stopped: false` is the caller's cue to cancel instead."""
    user_id = await _user(client)
    session_id = await _session(user_id, status="running")

    response = await client.post(f"/sessions/{session_id}/stop")

    assert response.status_code == 202
    assert response.json()["stopped"] is False


# --- answering a held run --------------------------------------------------------


@pytest.mark.asyncio
async def test_the_approve_word_resumes_without_touching_the_mode(client, monkeypatch):
    user_id = await _user(client)
    session_id = await _session(user_id)
    row = await _held(session_id)

    calls: list[dict] = []

    async def fake_start(session_id: str, **kw) -> bool:
        calls.append({"session_id": session_id, **kw})
        return True

    monkeypatch.setattr(runner, "start", fake_start)
    response = await client.post(f"/approvals/{row.id}/respond", json={"answer": "approve"})

    assert response.status_code == 202
    assert calls[-1]["reason"] == "resumed"
    assert "mode" not in calls[-1], "the hold never moved it, so nothing moves it back"
    assert await approvals.open_for(session_id) == []
    # Bare consent to carry on says nothing to the model.
    assert [e.event.kind for e in await slog.get_events(session_id)] == []


@pytest.mark.asyncio
async def test_prose_resumes_the_run_with_the_message_appended(client, monkeypatch):
    """"skip that step, do X instead" IS the resume."""
    user_id = await _user(client)
    session_id = await _session(user_id)
    row = await _held(session_id)

    calls: list[dict] = []

    async def fake_start(session_id: str, **kw) -> bool:
        calls.append({"session_id": session_id, **kw})
        return True

    monkeypatch.setattr(runner, "start", fake_start)
    response = await client.post(
        f"/approvals/{row.id}/respond", json={"answer": "skip the browser, do it another way"}
    )

    assert response.status_code == 202
    assert calls[-1]["reason"] == "resumed"
    events = [e.event for e in await slog.get_events(session_id)]
    assert [(e.kind, e.source) for e in events] == [("user", "human")]
    assert events[0].text == "skip the browser, do it another way"


@pytest.mark.asyncio
async def test_a_composer_message_resumes_a_stopped_run(client, monkeypatch):
    """The one park a typed message answers: its consent was given already.

    `call` and `plan` still 409 there — those wait on consent nobody has given
    yet, and prose beside them would be read as giving it.
    """
    user_id = await _user(client)
    session_id = await _session(user_id)
    await _held(session_id)

    calls: list[dict] = []

    async def fake_start(session_id: str, **kw) -> bool:
        calls.append({"session_id": session_id, **kw})
        return True

    monkeypatch.setattr(runner, "start", fake_start)
    response = await client.post(
        f"/sessions/{session_id}/messages", json={"text": "skip the browser, do it another way"}
    )

    assert response.status_code == 202
    assert calls, "the run woke"
    assert await approvals.open_for(session_id) == []
    events = [e.event for e in await slog.get_events(session_id)]
    assert events[-1].text == "skip the browser, do it another way"


@pytest.mark.asyncio
async def test_prose_that_contains_the_decline_word_still_resumes(client, monkeypatch):
    """Cancelling is a card action. The word is not a trapdoor in the composer."""
    user_id = await _user(client)
    session_id = await _session(user_id)
    await _held(session_id)

    calls: list[str] = []

    async def fake_start(session_id: str, **kw) -> bool:
        calls.append(session_id)
        return True

    monkeypatch.setattr(runner, "start", fake_start)
    response = await client.post(
        f"/sessions/{session_id}/messages", json={"text": "decline the invite, then carry on"}
    )

    assert response.status_code == 202
    assert calls == [session_id], "it woke the run rather than ending it"
    row = await pool.fetchrow(
        "SELECT status, terminal_reason FROM sessions WHERE id = $1", uuid.UUID(session_id)
    )
    assert row["status"] != "cancelled" and row["terminal_reason"] is None
    events = [e.event for e in await slog.get_events(session_id)]
    assert events[-1].text == "decline the invite, then carry on", "the words went to the model"


@pytest.mark.asyncio
async def test_a_plan_park_still_refuses_a_composer_message(client):
    """The exemption is `resume` alone."""
    user_id = await _user(client)
    session_id = await _session(user_id, mode="attended")
    await approvals.create(session_id, "c1", "plan", "a goal", tool_name="propose_plan", tool_args={})
    await pool.execute(
        "UPDATE sessions SET status = 'awaiting_approval' WHERE id = $1", uuid.UUID(session_id)
    )

    response = await client.post(f"/sessions/{session_id}/messages", json={"text": "sounds good"})

    assert response.status_code == 409
    assert response.json()["code"] == "awaiting_approval"


@pytest.mark.asyncio
async def test_the_decline_word_cancels_for_real_and_spends_the_plan(client):
    user_id = await _user(client)
    session_id = await _session(user_id)
    row = await _held(session_id)

    response = await client.post(f"/approvals/{row.id}/respond", json={"answer": "decline"})

    assert response.status_code == 202
    assert response.json()["cancelled"] is True
    session = await pool.fetchrow(
        "SELECT status, mode, terminal_reason FROM sessions WHERE id = $1", uuid.UUID(session_id)
    )
    assert session["status"] == "cancelled"
    assert session["mode"] == "attended", "the plan's approval is spent"
    assert session["terminal_reason"] == "cancelled"
    assert await approvals.open_for(session_id) == []


@pytest.mark.asyncio
async def test_cancelling_a_stopped_session_closes_its_row(client):
    """A resume card offering to restart a terminal session is a lie on screen."""
    user_id = await _user(client)
    session_id = await _session(user_id)
    await _held(session_id)

    response = await client.post(f"/sessions/{session_id}/cancel")

    assert response.status_code == 202
    assert await approvals.open_for(session_id) == []
    row = await pool.fetchrow("SELECT status, mode FROM sessions WHERE id = $1", uuid.UUID(session_id))
    assert (row["status"], row["mode"]) == ("cancelled", "attended")


# --- the shape of the thing ------------------------------------------------------


def test_a_hold_is_a_park_and_a_cancel_is_a_terminal():
    assert ("running", "awaiting_approval") in lifecycle.ALLOWED
    assert ("awaiting_approval", "cancelled") in lifecycle.ALLOWED


def test_the_stop_grace_is_configured():
    """The backstop: a hop that never reaches a boundary is still killable."""
    assert float(runner._cfg("harness.stop_grace_s", 0)) > 0

