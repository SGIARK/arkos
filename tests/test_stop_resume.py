"""One teardown, two landings: stop and cancel are the same `task.cancel()` path.

Cancel is terminal (mode handed back to attended); stop is not (`idle`, mode kept),
so resuming is just the ordinary start of an idle session.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from agent_module import loop as lp
from agent_module.events import DoneEvent, UserEvent
from db import pool
from harness_module import api, approvals, lifecycle, runner, store
from harness_module import session_log as slog
from tests.dbgate import require_db
from tests.runner_tasks import cancel_and_forget
from tests.test_api import _supabase_token
from tool_module.envelope import ResultEnvelope

_seeded: list[uuid.UUID] = []


@pytest_asyncio.fixture(autouse=True)
async def _db(tmp_path):
    await require_db()
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    yield
    cancel_and_forget()
    await asyncio.sleep(0)
    store.use_blobs(None)
    await pool.execute("DELETE FROM sessions WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM files WHERE user_id = ANY($1::uuid[])", _seeded)
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
    response = await client.post("/auth/session", headers={"Authorization": f"Bearer {_supabase_token(user_id)}"})
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


async def _stopped(user_id: str) -> str:
    """A session where a stop leaves it: idle, mode kept, nothing to answer."""
    session_id = await _session(user_id, status="running")
    sink = runner._Sink(await runner.load(session_id))
    await sink.abort("stopped")
    return session_id


# --- the landing -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_stop_lands_idle_with_the_mode_kept_and_no_terminal():
    """The whole of why the plan survives: `idle` is not terminal, and mode did not move."""
    user_id = str(uuid.uuid4())
    _seeded.append(uuid.UUID(user_id))
    await pool.execute("INSERT INTO users (id) VALUES ($1)", uuid.UUID(user_id))
    session_id = await _session(user_id)
    sink = runner._Sink(await runner.load(session_id))

    assert await sink.abort("stopped") is True

    row = await pool.fetchrow(
        "SELECT status, mode, terminal_reason, ended_at FROM sessions WHERE id = $1",
        uuid.UUID(session_id),
    )
    assert (row["status"], row["mode"]) == ("idle", "unattended")
    assert row["terminal_reason"] is None and row["ended_at"] is None
    done = [e.event for e in await slog.get_events(session_id) if e.event.kind == "done"]
    assert [e.reason for e in done] == ["stopped"]
    assert await approvals.open_for(session_id) == []


@pytest.mark.asyncio
async def test_a_cancel_lands_terminal_and_hands_the_mode_back():
    user_id = str(uuid.uuid4())
    _seeded.append(uuid.UUID(user_id))
    await pool.execute("INSERT INTO users (id) VALUES ($1)", uuid.UUID(user_id))
    session_id = await _session(user_id)
    sink = runner._Sink(await runner.load(session_id))

    await sink.abort("cancelled")

    row = await pool.fetchrow("SELECT status, mode, terminal_reason FROM sessions WHERE id = $1", uuid.UUID(session_id))
    assert (row["status"], row["mode"]) == ("cancelled", "attended")
    assert row["terminal_reason"] == "cancelled", "the plan's approval is spent"


@pytest.mark.asyncio
async def test_the_press_decides_the_landing_of_one_teardown(monkeypatch):
    """Both faces signal the turn identically; the intent says where it lands."""
    user_id = str(uuid.uuid4())
    _seeded.append(uuid.UUID(user_id))
    await pool.execute("INSERT INTO users (id) VALUES ($1)", uuid.UUID(user_id))

    for intent, status, mode in (("stopped", "idle", "unattended"), ("cancelled", "cancelled", "attended")):
        session_id = await _session(user_id)
        landed = asyncio.Event()

        async def forever(sid: str = session_id, done: asyncio.Event = landed) -> None:
            # Mirrors `_drive`'s cancellation path: the intent decides the reason.
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                sink = runner._Sink(await runner.load(sid))
                await runner._ending(sid, sink, runner._teardown.get(sid, "cancelled"))
                done.set()
                raise

        task = asyncio.create_task(forever())
        runner._running[session_id] = task
        await asyncio.sleep(0)

        assert await (runner.stop if intent == "stopped" else runner.cancel)(session_id) is True
        await asyncio.wait_for(landed.wait(), timeout=5)
        runner._running.pop(session_id, None)
        runner._teardown.pop(session_id, None)

        row = await pool.fetchrow("SELECT status, mode FROM sessions WHERE id = $1", uuid.UUID(session_id))
        assert (row["status"], row["mode"]) == (status, mode), intent


@pytest.mark.asyncio
async def test_both_landings_give_every_lease_back(monkeypatch):
    """Neither a stop nor a cancel is acting, so neither holds a shared resource."""
    user_id = str(uuid.uuid4())
    _seeded.append(uuid.UUID(user_id))
    await pool.execute("INSERT INTO users (id) VALUES ($1)", uuid.UUID(user_id))

    released: list[str] = []
    for reason in ("stopped", "cancelled"):
        session_id = await _session(user_id)
        sink = runner._Sink(await runner.load(session_id))

        async def spy(reason=reason) -> None:
            released.append(reason)

        monkeypatch.setattr(sink, "_release_leases", spy)
        await sink.abort(reason)

    assert released == ["stopped", "cancelled"]


@pytest.mark.asyncio
async def test_a_live_stop_lands_stopped_and_not_the_loops_cancelled(monkeypatch):
    """A real turn, cancelled mid-hop, lands `stopped`, not `cancelled`.

    A task cancelled while awaiting `__anext__` never receives `run_turn`'s parting
    `DoneEvent(reason="cancelled")`; every other test here calls `abort` directly.
    """
    from model_module import client as mc

    user_id = str(uuid.uuid4())
    _seeded.append(uuid.UUID(user_id))
    await pool.execute("INSERT INTO users (id) VALUES ($1)", uuid.UUID(user_id))
    session_id = await _session(user_id, status="idle")
    await slog.append(session_id, UserEvent(text="go"))

    def slow(messages, tools=None, **kw):
        async def gen():
            yield mc.TextDelta(text="starting")
            await asyncio.sleep(30)
            yield mc.Finish(reason="stop")

        return gen()

    monkeypatch.setattr(mc, "generate", slow)
    assert await runner.start(session_id) is True
    await asyncio.sleep(0.3)

    assert await runner.stop(session_id) is True

    row = await pool.fetchrow("SELECT status, mode, terminal_reason FROM sessions WHERE id = $1", uuid.UUID(session_id))
    dones = [e.event for e in await slog.get_events(session_id) if e.event.kind == "done"]
    assert [d.reason for d in dones] == ["stopped"], "the loop's own done{cancelled} was consumed"
    assert (row["status"], row["mode"]) == ("idle", "unattended")
    assert row["terminal_reason"] is None, "a stop wrote a terminal"


@pytest.mark.asyncio
async def test_a_live_cancel_still_lands_cancelled(monkeypatch):
    """The other landing of the same teardown, through the same live path."""
    from model_module import client as mc

    user_id = str(uuid.uuid4())
    _seeded.append(uuid.UUID(user_id))
    await pool.execute("INSERT INTO users (id) VALUES ($1)", uuid.UUID(user_id))
    session_id = await _session(user_id, status="idle")
    await slog.append(session_id, UserEvent(text="go"))

    def slow(messages, tools=None, **kw):
        async def gen():
            yield mc.TextDelta(text="starting")
            await asyncio.sleep(30)
            yield mc.Finish(reason="stop")

        return gen()

    monkeypatch.setattr(mc, "generate", slow)
    await runner.start(session_id)
    await asyncio.sleep(0.3)

    assert await runner.cancel(session_id) is True

    row = await pool.fetchrow("SELECT status, mode, terminal_reason FROM sessions WHERE id = $1", uuid.UUID(session_id))
    dones = [e.event for e in await slog.get_events(session_id) if e.event.kind == "done"]
    assert [d.reason for d in dones] == ["cancelled"]
    assert (row["status"], row["mode"], row["terminal_reason"]) == ("cancelled", "attended", "cancelled")


# --- resuming is code that already exists ----------------------------------------


@pytest.mark.asyncio
async def test_a_message_resumes_a_stopped_run_unattended(client, monkeypatch):
    """The ordinary message path: the mode was kept, so an idle session starts unattended."""
    user_id = await _user(client)
    session_id = await _stopped(user_id)

    calls: list[dict] = []

    async def fake_start(sid: str, **kw) -> bool:
        calls.append({"session_id": sid, **kw})
        return True

    monkeypatch.setattr(runner, "start", fake_start)
    response = await client.post(
        f"/sessions/{session_id}/messages", json={"text": "skip the browser, do it another way"}
    )

    assert response.status_code == 202
    assert calls and "mode" not in calls[-1], "nothing moved the mode, so nothing moves it back"
    events = [e.event for e in await slog.get_events(session_id)]
    assert events[-1].text == "skip the browser, do it another way"
    assert await pool.fetchval("SELECT mode FROM sessions WHERE id = $1", uuid.UUID(session_id)) == "unattended"


@pytest.mark.asyncio
async def test_a_plain_start_resumes_it_with_nothing_added(client, monkeypatch):
    user_id = await _user(client)
    session_id = await _stopped(user_id)
    before = len(await slog.get_events(session_id))

    calls: list[dict] = []

    async def fake_start(sid: str, **kw) -> bool:
        calls.append({"session_id": sid, **kw})
        return True

    monkeypatch.setattr(runner, "start", fake_start)
    response = await client.post(f"/sessions/{session_id}/resume")

    assert response.status_code == 202
    assert response.json()["mode"] == "unattended"
    assert calls[-1]["reason"] == "resumed"
    assert "mode" not in calls[-1]
    assert len(await slog.get_events(session_id)) == before, "resuming injected something"


@pytest.mark.asyncio
async def test_resume_refuses_a_session_that_is_not_waiting(client):
    user_id = await _user(client)
    running = await _session(user_id, status="running")

    response = await client.post(f"/sessions/{running}/resume")

    assert response.status_code == 409
    assert response.json()["code"] == "not_idle"


@pytest.mark.asyncio
async def test_cancelling_a_stopped_run_spends_the_plan(client):
    """The second press. No live turn, so the terminal is written directly."""
    user_id = await _user(client)
    session_id = await _stopped(user_id)

    response = await client.post(f"/sessions/{session_id}/cancel")

    assert response.status_code == 202
    row = await pool.fetchrow("SELECT status, mode, terminal_reason FROM sessions WHERE id = $1", uuid.UUID(session_id))
    assert (row["status"], row["mode"]) == ("cancelled", "attended")
    assert row["terminal_reason"] == "cancelled"


@pytest.mark.asyncio
async def test_every_direct_write_terminal_hands_the_mode_back(client):
    """The no-sink path hands the mode back for EVERY terminal, not just its one caller."""
    user_id = await _user(client)
    session_id = await _session(user_id, status="idle", mode="unattended")

    await client.post(f"/sessions/{session_id}/cancel")

    row = await pool.fetchrow("SELECT status, mode FROM sessions WHERE id = $1", uuid.UUID(session_id))
    assert (row["status"], row["mode"]) == ("cancelled", "attended")

    # The no-sink path directly: what a cancel landing before `_drive` builds a sink hits.
    other = await _session(user_id, status="running", mode="unattended")
    assert await runner._ending(other, None, "cancelled") is True

    row = await pool.fetchrow("SELECT status, mode FROM sessions WHERE id = $1", uuid.UUID(other))
    assert (row["status"], row["mode"]) == ("cancelled", "attended")


@pytest.mark.asyncio
async def test_a_stop_through_the_no_sink_path_keeps_the_mode(client):
    """`stopped` is not terminal, so it must NOT be caught by the hand-back."""
    user_id = await _user(client)
    session_id = await _session(user_id, status="running", mode="unattended")

    assert await runner._ending(session_id, None, "stopped") is True

    row = await pool.fetchrow("SELECT status, mode, terminal_reason FROM sessions WHERE id = $1", uuid.UUID(session_id))
    assert (row["status"], row["mode"]) == ("idle", "unattended")
    assert row["terminal_reason"] is None


@pytest.mark.asyncio
async def test_stop_refuses_a_session_that_is_not_running(client):
    user_id = await _user(client)
    session_id = await _session(user_id, status="idle")

    response = await client.post(f"/sessions/{session_id}/stop")

    assert response.status_code == 409
    assert response.json()["code"] == "not_running"


@pytest.mark.asyncio
async def test_a_plan_park_still_refuses_a_composer_message(client):
    """Consent nobody has given yet is still answered on its own card."""
    user_id = await _user(client)
    session_id = await _session(user_id, mode="attended")
    await approvals.create(session_id, "c1", "plan", "a goal", tool_name="propose_plan", tool_args={})
    await pool.execute("UPDATE sessions SET status = 'awaiting_approval' WHERE id = $1", uuid.UUID(session_id))

    response = await client.post(f"/sessions/{session_id}/messages", json={"text": "sounds good"})

    assert response.status_code == 409
    assert response.json()["code"] == "awaiting_approval"


# --- what the deletion bought ----------------------------------------------------


def test_a_stop_is_not_terminal_and_not_a_failure():
    assert lifecycle.status_for(DoneEvent(reason="stopped")) == "idle"
    assert not DoneEvent(reason="stopped").is_terminal()
    assert ("running", "idle") in lifecycle.ALLOWED


def test_every_failed_call_spends_the_tools_attempts():
    """The failure streak is per-turn state, so every failed call spends an attempt."""
    state = lp._State(budgets=lp.Budgets.load("unattended"))
    call = lp._PartialCall(id="c1", name="browser_task")
    failed = ResultEnvelope(ok=False, content="boom", error_kind="upstream_error")

    asyncio.run(lp._settle(call, failed, state, []))

    assert state.failures["browser_task"] == 1


def test_the_stop_machinery_is_gone():
    """The stop machinery stays deleted."""
    for name in ("request_stop", "park_stopped", "_force_stop", "_stopped_envelope", "_stop_backstops"):
        assert not hasattr(runner, name), name
    assert not hasattr(runner._Sink, "stopping")
    assert "resume" not in approvals.Kind.__args__
