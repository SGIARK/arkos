"""Parking on a question, and waking when it is answered.

Runs against a real Postgres with migration 0 applied; the model is mocked.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from agent_module.events import UserEvent
from db import pool
from harness_module import api, approvals, runner
from harness_module import session_log as slog
from model_module import client as mc
from tests.dbgate import require_db

pytestmark = pytest.mark.asyncio

_seeded: list[uuid.UUID] = []


@pytest_asyncio.fixture(autouse=True)
async def _db():
    await require_db()
    yield
    for task in list(runner._reapers) + list(runner._running.values()):
        task.cancel()
    runner._running.clear()
    runner._reapers.clear()
    runner._teardown.clear()
    await asyncio.sleep(0)
    await pool.execute("DELETE FROM sessions WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM users WHERE id = ANY($1::uuid[])", _seeded)
    _seeded.clear()
    await pool.close()


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=api.app)
    async with AsyncClient(transport=transport, base_url="https://testserver") as c:
        yield c


async def _user() -> str:
    user_id = uuid.uuid4()
    await pool.execute("INSERT INTO users (id) VALUES ($1)", user_id)
    _seeded.append(user_id)
    return str(user_id)


async def _session(user_id: str, status: str = "idle", mode: str = "unattended") -> str:
    return str(
        await pool.fetchval(
            "INSERT INTO sessions (user_id, mode, status, goal) VALUES ($1, $2, $3, 'do it') RETURNING id",
            uuid.UUID(user_id),
            mode,
            status,
        )
    )


async def _sign_in(client: AsyncClient, user_id: str) -> None:
    token = jwt.encode(
        {
            "sub": user_id,
            "email": "a@example.com",
            "aud": "authenticated",
            "exp": datetime.now(UTC) + timedelta(hours=1),
        },
        "test-supabase-secret-at-least-32-chars",
        algorithm="HS256",
    )
    assert (await client.post("/auth/session", headers={"Authorization": f"Bearer {token}"})).status_code == 204


def _calls(name: str, args: str, call_id: str = "p1"):
    return [
        mc.ToolCallDelta(index=0, id=call_id, name=name, arguments=args),
        mc.Finish(reason="tool_calls"),
    ]


@pytest.fixture
def model(monkeypatch):
    """Arm the model with one delta list per hop."""

    def arm(*hops):
        remaining = list(hops)

        def generate(messages, tools=None, **kw):
            deltas = remaining.pop(0) if remaining else [mc.TextDelta(text="ok"), mc.Finish(reason="stop")]

            async def gen():
                for d in deltas:
                    await asyncio.sleep(0)
                    yield d

            return gen()

        monkeypatch.setattr(mc, "generate", generate)

    return arm


async def _park_on(session_id: str, model, tool: str, args: str) -> None:
    """Drive one turn that calls a park tool, and wait for the session to settle."""
    await slog.append(session_id, UserEvent(text="go"))
    model(_calls(tool, args))
    await runner.start(session_id)
    task = runner._running.get(session_id)
    if task is not None:
        await asyncio.wait_for(asyncio.shield(task), timeout=45)


# --- parking --------------------------------------------------------------------


async def test_a_park_leaves_no_open_tool_call(model):
    """The transcript a park leaves behind must fold back into messages."""
    user_id = await _user()
    session_id = await _session(user_id)

    await _park_on(session_id, model, "ask", '{"question": "which invoice?"}')

    kinds = [e.event.kind for e in await slog.get_events(session_id)]
    row = await pool.fetchrow("SELECT status FROM sessions WHERE id = $1", uuid.UUID(session_id))

    assert row["status"] == "awaiting_approval"
    assert kinds.count("tool_call") == kinds.count("tool_result") == 1
    assert kinds.index("tool_call") < kinds.index("tool_result")
    assert "done" not in kinds, "a park is not a terminal; the run is not over"


async def test_a_parked_session_folds_cleanly(model):
    user_id = await _user()
    session_id = await _session(user_id)
    await _park_on(session_id, model, "ask", '{"question": "which invoice?"}')

    folded = await runner.fold(await runner.load(session_id))

    open_ids: set[str] = set()
    for message in folded.messages:
        if message.get("role") == "assistant":
            open_ids = {c["id"] for c in message.get("tool_calls") or []}
        elif message.get("role") == "tool":
            assert message["tool_call_id"] in open_ids


async def test_the_question_is_recorded_for_a_human_to_answer(model):
    user_id = await _user()
    session_id = await _session(user_id)

    await _park_on(session_id, model, "ask", '{"question": "which invoice?"}')
    open_questions = await approvals.open_for(session_id)

    assert len(open_questions) == 1
    assert open_questions[0].kind == "ask"
    assert open_questions[0].prompt == "which invoice?"
    assert open_questions[0].answered_at is None


async def test_request_approval_records_the_action_and_its_detail(model):
    user_id = await _user()
    session_id = await _session(user_id)

    await _park_on(
        session_id,
        model,
        "request_approval",
        '{"action": "send the invoice", "detail": "to a@b.c, 400 GBP"}',
    )
    open_questions = await approvals.open_for(session_id)

    assert open_questions[0].kind == "approval"
    assert "send the invoice" in open_questions[0].prompt
    assert "400 GBP" in open_questions[0].prompt


async def test_a_parked_session_queries_nothing(model):
    """No poll and no timer: the wait costs one row and no work."""
    user_id = await _user()
    session_id = await _session(user_id)
    await _park_on(session_id, model, "ask", '{"question": "which invoice?"}')

    queries = 0
    real_fetch, real_execute = pool.fetch, pool.execute

    async def counting_fetch(*a, **kw):
        nonlocal queries
        queries += 1
        return await real_fetch(*a, **kw)

    async def counting_execute(*a, **kw):
        nonlocal queries
        queries += 1
        return await real_execute(*a, **kw)

    pool.fetch, pool.execute = counting_fetch, counting_execute
    try:
        await asyncio.sleep(0.5)
    finally:
        pool.fetch, pool.execute = real_fetch, real_execute

    assert queries == 0
    assert not runner.is_running(session_id)


async def test_the_pending_question_survives_a_restart(model):
    """Nothing is held in memory, so a new process finds the same open row."""
    user_id = await _user()
    session_id = await _session(user_id)
    await _park_on(session_id, model, "ask", '{"question": "which invoice?"}')

    runner._running.clear()
    await pool.close()

    still_open = await approvals.open_for(session_id)
    row = await pool.fetchrow("SELECT status FROM sessions WHERE id = $1", uuid.UUID(session_id))

    assert [q.prompt for q in still_open] == ["which invoice?"]
    assert row["status"] == "awaiting_approval"


# --- answering ------------------------------------------------------------------


async def test_responding_answers_the_row_appends_the_answer_and_wakes(client, model, monkeypatch):
    user_id = await _user()
    session_id = await _session(user_id)
    await _park_on(session_id, model, "ask", '{"question": "which invoice?"}')
    question = (await approvals.open_for(session_id))[0]
    await _sign_in(client, user_id)

    woken: list[str] = []

    async def fake_start(sid, **kw):
        woken.append(sid)
        return True

    monkeypatch.setattr(runner, "start", fake_start)
    response = await client.post(f"/approvals/{question.id}/respond", json={"answer": "the March one"})

    assert response.status_code == 202
    assert woken == [session_id]
    assert not await approvals.open_for(session_id)
    events = [e.event for e in await slog.get_events(session_id)]
    assert events[-1].kind == "user" and events[-1].text == "the March one"
    # The answer is a message, not the return value of the call that asked.
    assert events[-1].source == "human"


async def test_answering_twice_is_refused(client, model, monkeypatch):
    user_id = await _user()
    session_id = await _session(user_id)
    await _park_on(session_id, model, "ask", '{"question": "which one?"}')
    question = (await approvals.open_for(session_id))[0]
    await _sign_in(client, user_id)
    monkeypatch.setattr(runner, "start", lambda sid, **kw: _true())

    first = await client.post(f"/approvals/{question.id}/respond", json={"answer": "this one"})
    second = await client.post(f"/approvals/{question.id}/respond", json={"answer": "no, that one"})

    assert first.status_code == 202
    assert second.status_code == 409
    assert second.json()["code"] == "already_answered"


async def test_an_empty_answer_is_refused(client, model):
    user_id = await _user()
    session_id = await _session(user_id)
    await _park_on(session_id, model, "ask", '{"question": "which one?"}')
    question = (await approvals.open_for(session_id))[0]
    await _sign_in(client, user_id)

    response = await client.post(f"/approvals/{question.id}/respond", json={"answer": "   "})

    assert response.status_code == 400
    assert (await approvals.open_for(session_id))[0].answered_at is None


async def test_another_users_question_cannot_be_answered(client, model):
    theirs = await _user()
    their_session = await _session(theirs)
    await _park_on(their_session, model, "ask", '{"question": "secret?"}')
    question = (await approvals.open_for(their_session))[0]

    await _sign_in(client, await _user())
    response = await client.post(f"/approvals/{question.id}/respond", json={"answer": "hello"})

    assert response.status_code == 404
    assert (await approvals.open_for(their_session))[0].answered_at is None


# --- answering from the composer -------------------------------------------------


async def test_a_message_answers_an_ask_and_wakes_the_session(client, model, monkeypatch):
    user_id = await _user()
    session_id = await _session(user_id)
    await _park_on(session_id, model, "ask", '{"question": "which invoice?"}')
    await _sign_in(client, user_id)

    woken: list[str] = []

    async def fake_start(sid, **kw):
        woken.append(sid)
        return True

    monkeypatch.setattr(runner, "start", fake_start)
    response = await client.post(f"/sessions/{session_id}/messages", json={"text": "the March one"})

    assert response.status_code == 202
    assert woken == [session_id]
    assert not await approvals.open_for(session_id)


async def test_a_message_never_answers_a_request_for_approval(client, model):
    """An ambiguous reply typed into a chat box is not consent."""
    user_id = await _user()
    session_id = await _session(user_id)
    await _park_on(session_id, model, "request_approval", '{"action": "send the invoice"}')
    await _sign_in(client, user_id)
    before = len(await slog.get_events(session_id))

    response = await client.post(f"/sessions/{session_id}/messages", json={"text": "sure"})

    assert response.status_code == 409
    assert response.json()["code"] == "awaiting_approval"
    assert (await approvals.open_for(session_id))[0].answered_at is None
    assert len(await slog.get_events(session_id)) == before, "the refused message left no trace"
    row = await pool.fetchrow("SELECT status FROM sessions WHERE id = $1", uuid.UUID(session_id))
    assert row["status"] == "awaiting_approval"


async def _true() -> bool:
    return True
