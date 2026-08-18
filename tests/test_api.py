"""The HTTP surface: the cookie is the only way in, and every read is scoped to its owner.

Runs against a real Postgres with migration 0 applied. The SSE generator is driven
directly, because httpx's ASGITransport buffers streamed responses.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from agent_module.events import ContentEvent, DoneEvent, ToolCallEvent, UserEvent
from db import pool
from harness_module import api, runner
from harness_module import session_log as slog
from harness_module.stream import SessionStream, stream
from tests.dbgate import require_db

pytestmark = pytest.mark.asyncio

_seeded: list[uuid.UUID] = []


@pytest_asyncio.fixture(autouse=True)
async def _db(monkeypatch):
    await require_db()

    # The loop is out of scope here, so start is stubbed and no turn runs.
    started: list[str] = []
    start_calls: list[dict] = []

    async def fake_start(session_id: str, **kw) -> bool:
        started.append(session_id)
        start_calls.append({"session_id": session_id, **kw})
        return True

    monkeypatch.setattr(runner, "start", fake_start)
    api.started = started
    api.start_calls = start_calls
    yield
    await pool.execute("DELETE FROM sessions WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM projects WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM users WHERE id = ANY($1::uuid[])", _seeded)
    _seeded.clear()
    await pool.close()


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=api.app)
    # https, because a Secure cookie is not sent over http even to a fake host.
    async with AsyncClient(transport=transport, base_url="https://testserver") as c:
        yield c


def _supabase_token(user_id: str, email: str = "a@example.com", **over) -> str:
    """A token shaped like the one supabase-js sends, signed with the project secret."""
    claims = {
        "sub": user_id,
        "email": email,
        "aud": "authenticated",
        "exp": datetime.now(UTC) + timedelta(hours=1),
        **over,
    }
    return jwt.encode(claims, "test-supabase-secret-at-least-32-chars", algorithm="HS256")


async def _signed_in(client: AsyncClient) -> str:
    """Sign a fresh user in and leave the cookie on the client."""
    user_id = str(uuid.uuid4())
    _seeded.append(uuid.UUID(user_id))
    response = await client.post("/auth/session", headers={"Authorization": f"Bearer {_supabase_token(user_id)}"})
    assert response.status_code == 204
    return user_id


async def _session_for(user_id: str, **cols) -> str:
    session_id = await pool.fetchval(
        "INSERT INTO sessions (user_id, mode, status, title, goal) VALUES ($1, $2, $3, $4, $5) RETURNING id",
        uuid.UUID(user_id),
        cols.get("mode", "attended"),
        cols.get("status", "idle"),
        cols.get("title", "A session"),
        cols.get("goal", "do the thing"),
    )
    return str(session_id)


# --- auth ----------------------------------------------------------------------


async def test_a_verified_supabase_token_is_the_only_way_to_a_cookie(client):
    user_id = str(uuid.uuid4())
    _seeded.append(uuid.UUID(user_id))

    response = await client.post("/auth/session", headers={"Authorization": f"Bearer {_supabase_token(user_id)}"})

    assert response.status_code == 204
    assert "ark_session" in response.cookies
    row = await pool.fetchrow("SELECT id, email FROM users WHERE id = $1", uuid.UUID(user_id))
    assert row["email"] == "a@example.com", "sub -> users, created on first login"


async def test_a_token_we_did_not_verify_buys_nothing(client):
    forged = jwt.encode(
        {"sub": str(uuid.uuid4()), "aud": "authenticated", "exp": datetime.now(UTC) + timedelta(hours=1)},
        "not-the-supabase-secret",
        algorithm="HS256",
    )

    response = await client.post("/auth/session", headers={"Authorization": f"Bearer {forged}"})

    assert response.status_code == 401
    assert response.json()["code"] == "unauthenticated"


async def test_an_expired_token_is_refused(client):
    expired = _supabase_token(str(uuid.uuid4()), exp=datetime.now(UTC) - timedelta(minutes=1))

    response = await client.post("/auth/session", headers={"Authorization": f"Bearer {expired}"})

    assert response.status_code == 401


async def test_test_cookie_session(client):
    """No cookie and a foreign cookie are both rejected, on every endpoint."""
    forged = jwt.encode({"sub": str(uuid.uuid4()), "iss": "arkos"}, "wrong-secret", algorithm="HS256")

    for method, path in (
        ("get", "/auth/me"),
        ("get", "/projects"),
        ("get", f"/sessions/{uuid.uuid4()}"),
        ("get", f"/sessions/{uuid.uuid4()}/events"),
        ("post", "/sessions"),
        ("post", f"/sessions/{uuid.uuid4()}/messages"),
        ("post", f"/sessions/{uuid.uuid4()}/cancel"),
        ("get", f"/results/{uuid.uuid4()}"),
        ("get", "/connections"),
    ):
        body = {"json": {}} if method == "post" else {}
        bare = await getattr(client, method)(path, **body)
        assert bare.status_code == 401, f"{method} {path} let a caller in with no cookie"

        foreign = await getattr(client, method)(path, cookies={"ark_session": forged}, **body)
        assert foreign.status_code == 401, f"{method} {path} accepted a cookie we did not sign"


async def test_a_cross_origin_mutation_is_refused(client, monkeypatch):
    """A mutation carrying an Origin other than the configured one is refused."""
    await _signed_in(client)
    monkeypatch.setattr(api, "_origin", "https://app.example.com")

    same_site = await client.post("/sessions", json={"goal": "ours"}, headers={"Origin": "https://app.example.com"})
    cross_site = await client.post("/sessions", json={"goal": "theirs"}, headers={"Origin": "https://evil.example"})

    assert same_site.status_code == 201
    assert cross_site.status_code == 403
    assert cross_site.json()["code"] == "bad_origin"


async def test_an_unset_public_url_refuses_browser_mutations_rather_than_all_of_them(client, monkeypatch):
    """With no configured origin, a mutation carrying any Origin is refused."""
    await _signed_in(client)
    monkeypatch.setattr(api, "_origin", "")

    from_browser = await client.post("/sessions", json={"goal": "x"}, headers={"Origin": "https://anywhere"})
    no_origin = await client.post("/sessions", json={"goal": "y"})

    assert from_browser.status_code == 403
    assert no_origin.status_code == 201, "a non-browser client sends no Origin at all"


async def test_a_read_is_not_origin_checked(client, monkeypatch):
    await _signed_in(client)
    monkeypatch.setattr(api, "_origin", "https://app.example.com")

    response = await client.get("/auth/me", headers={"Origin": "https://evil.example"})

    assert response.status_code == 200


async def test_auth_me_reports_the_signed_in_user(client):
    user_id = await _signed_in(client)

    body = (await client.get("/auth/me")).json()

    assert body == {"user_id": user_id, "email": "a@example.com"}


async def test_logout_clears_the_cookie(client):
    await _signed_in(client)

    response = await client.delete("/auth/session")

    assert response.status_code == 204
    assert response.cookies.get("ark_session") in (None, "")


async def test_health_needs_no_cookie(client):
    body = (await client.get("/health")).json()

    assert body["status"] in ("ok", "degraded")


# --- sessions -------------------------------------------------------------------


async def test_creating_a_session_opens_a_project_and_starts_the_turn(client):
    await _signed_in(client)

    response = await client.post("/sessions", json={"goal": "file my taxes", "steps": ["gather", "file"]})
    body = response.json()

    assert response.status_code == 201
    row = await pool.fetchrow(
        "SELECT status, mode, goal, project_id FROM sessions WHERE id = $1",
        uuid.UUID(body["session_id"]),
    )
    assert (row["status"], row["mode"]) == ("pending", "attended"), "a new session is created attended"
    assert str(row["project_id"]) == body["project_id"]

    kinds = [e.event.kind for e in await slog.get_events(body["session_id"])]
    assert kinds == ["user", "todo"], "the goal is the first message; steps seed the todo list"
    assert body["session_id"] in api.started, "the turn was started"


async def test_a_session_without_a_goal_is_refused(client):
    await _signed_in(client)

    response = await client.post("/sessions", json={"goal": "   "})

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_request"


async def test_the_new_session_rate_quota_binds_before_anything_is_written(client, monkeypatch):
    user_id = await _signed_in(client)
    monkeypatch.setattr(api, "_cfg", lambda key, default: 1 if key == "quotas.new_sessions_per_hour" else default)

    first = await client.post("/sessions", json={"goal": "one"})
    second = await client.post("/sessions", json={"goal": "two"})

    assert first.status_code == 201
    assert second.status_code == 429
    assert second.json()["retryable"] is True
    count = await pool.fetchval("SELECT count(*) FROM sessions WHERE user_id = $1", uuid.UUID(user_id))
    assert count == 1, "the refused session left no row"


async def test_the_snapshot_carries_the_mode_and_the_recent_events(client):
    user_id = await _signed_in(client)
    session_id = await _session_for(user_id)
    await slog.append(session_id, UserEvent(text="hello"))
    await slog.append(session_id, ContentEvent(text="hi"))

    body = (await client.get(f"/sessions/{session_id}")).json()

    assert body["status"] == "idle"
    assert body["mode"] == "attended", "the snapshot dropped the mode"
    assert body["hops_max"] == 6
    assert [e["kind"] for e in body["recent_events"]] == ["user", "content"]


async def test_a_message_appends_and_starts(client):
    user_id = await _signed_in(client)
    session_id = await _session_for(user_id)

    response = await client.post(f"/sessions/{session_id}/messages", json={"text": "and now this"})

    assert response.status_code == 202
    events = [e.event for e in await slog.get_events(session_id)]
    assert events[-1].text == "and now this"
    assert events[-1].source == "human"
    assert session_id in api.started


async def test_a_message_closes_a_dead_runs_open_call_before_it_lands(client):
    """A message closes an open tool call before the user event is appended."""
    user_id = await _signed_in(client)
    session_id = await _session_for(user_id)
    await slog.append(session_id, ToolCallEvent(id="c1", name="run_command", args={}))

    response = await client.post(f"/sessions/{session_id}/messages", json={"text": "any update?"})

    assert response.status_code == 202
    kinds = [e.event.kind for e in await slog.get_events(session_id)]
    assert kinds == ["tool_call", "tool_result", "user"]


async def test_a_composer_message_to_a_park_with_no_open_question_wakes_it(client):
    """A session parked with nothing to answer is unstuck by a message.

    Which park kinds accept a message, and which refuse, is covered in
    tests/test_approvals.py against a real park.
    """
    user_id = await _signed_in(client)
    session_id = await _session_for(user_id, status="awaiting_approval")

    response = await client.post(f"/sessions/{session_id}/messages", json={"text": "still there?"})

    assert response.status_code == 202
    events = [e.event for e in await slog.get_events(session_id)]
    assert [e.kind for e in events] == ["user"]
    assert session_id in api.started


# --- the handoff ----------------------------------------------------------------


async def test_approving_flips_the_mode_and_starts_the_run(client):
    """Approving asks for the unattended mode and starts the run on the same session."""
    user_id = await _signed_in(client)
    session_id = await _session_for(user_id, status="idle", mode="attended")

    response = await client.post(f"/sessions/{session_id}/approve")

    assert response.status_code == 202
    assert response.json()["mode"] == "unattended"
    assert session_id in api.started
    # The endpoint asks for the flip; test_runner pins that start applies it atomically.
    assert api.start_calls[-1]["mode"] == "unattended"
    assert api.start_calls[-1]["reason"] == "approved"


async def test_a_running_session_cannot_be_handed_over(client):
    user_id = await _signed_in(client)
    session_id = await _session_for(user_id, status="running")

    response = await client.post(f"/sessions/{session_id}/approve")

    assert response.status_code == 409
    assert response.json()["code"] == "not_idle"


async def test_approving_twice_is_refused(client):
    user_id = await _signed_in(client)
    session_id = await _session_for(user_id, status="idle", mode="unattended")

    response = await client.post(f"/sessions/{session_id}/approve")

    assert response.status_code == 409
    assert response.json()["code"] == "already_unattended"


async def test_the_unattended_quota_binds_where_it_can_fire(client, monkeypatch):
    """The unattended quota is counted at approval, and a refusal leaves the row unchanged."""
    user_id = await _signed_in(client)
    monkeypatch.setattr(api, "_cfg", lambda key, default: 1 if key == "quotas.max_unattended_sessions" else default)
    await _session_for(user_id, status="running", mode="unattended")
    blocked = await _session_for(user_id, status="idle", mode="attended")

    response = await client.post(f"/sessions/{blocked}/approve")

    assert response.status_code == 429
    assert response.json()["code"] == "quota_exceeded"
    row = await pool.fetchrow("SELECT mode, status FROM sessions WHERE id = $1", uuid.UUID(blocked))
    assert (row["mode"], row["status"]) == ("attended", "idle"), "the refusal changed nothing"


async def test_an_attended_session_does_not_consume_the_unattended_quota(client, monkeypatch):
    user_id = await _signed_in(client)
    monkeypatch.setattr(api, "_cfg", lambda key, default: 1 if key == "quotas.max_unattended_sessions" else default)
    await _session_for(user_id, status="running", mode="attended")
    mine = await _session_for(user_id, status="idle", mode="attended")

    response = await client.post(f"/sessions/{mine}/approve")

    assert response.status_code == 202


async def test_another_users_session_cannot_be_approved(client):
    theirs_user = str(uuid.uuid4())
    _seeded.append(uuid.UUID(theirs_user))
    await pool.execute("INSERT INTO users (id) VALUES ($1)", uuid.UUID(theirs_user))
    theirs = await _session_for(theirs_user, status="idle")
    await _signed_in(client)

    assert (await client.post(f"/sessions/{theirs}/approve")).status_code == 404


async def test_projects_roll_up_to_one_dot_each(client):
    user_id = await _signed_in(client)
    project_id = await pool.fetchval(
        "INSERT INTO projects (user_id, title) VALUES ($1, 'Taxes') RETURNING id", uuid.UUID(user_id)
    )
    await pool.execute(
        "INSERT INTO sessions (user_id, project_id, mode, status) VALUES ($1, $2, 'attended', 'awaiting_approval')",
        uuid.UUID(user_id),
        project_id,
    )

    body = (await client.get("/projects")).json()

    assert [p["title"] for p in body] == ["Taxes"]
    assert body[0]["status_rollup"] == "awaiting_approval", "the most urgent thing in the project wins the dot"


# --- ownership ------------------------------------------------------------------


async def test_authz_scoping(client):
    """User A cannot read or steer user B's sessions, results, or projects."""
    theirs = str(uuid.uuid4())
    _seeded.append(uuid.UUID(theirs))
    await pool.execute("INSERT INTO users (id) VALUES ($1)", uuid.UUID(theirs))
    their_session = await _session_for(theirs)
    their_ref = await slog.save_blob(their_session, "their secret")
    await pool.execute("INSERT INTO projects (user_id, title) VALUES ($1, 'Theirs')", uuid.UUID(theirs))

    await _signed_in(client)

    assert (await client.get(f"/sessions/{their_session}")).status_code == 404
    assert (await client.get(f"/sessions/{their_session}/events")).status_code == 404
    assert (await client.post(f"/sessions/{their_session}/messages", json={"text": "hi"})).status_code == 404
    assert (await client.post(f"/sessions/{their_session}/cancel")).status_code == 404
    assert (await client.get(f"/results/{their_ref}")).status_code == 404
    assert (await client.get("/projects")).json() == []


async def test_a_result_is_readable_by_its_owner_and_pages(client):
    user_id = await _signed_in(client)
    session_id = await _session_for(user_id)
    ref = await slog.save_blob(session_id, "abcdefghij")

    whole = (await client.get(f"/results/{ref}")).json()
    slice_ = (await client.get(f"/results/{ref}", params={"offset": 3, "limit": 4})).json()

    assert whole["content"] == "abcdefghij"
    assert slice_["content"] == "defg"


async def test_a_malformed_id_is_a_miss_not_a_500(client):
    await _signed_in(client)

    assert (await client.get("/sessions/not-a-uuid")).status_code == 404
    assert (await client.get("/results/not-a-uuid")).status_code == 404


# --- the stream -----------------------------------------------------------------


async def test_the_stream_replays_after_last_event_id_then_goes_live():
    session_id = await _new_session()
    first = await slog.append(session_id, UserEvent(text="one"))
    second = await slog.append(session_id, ContentEvent(text="two"))

    frames = []
    gen = api._event_stream(session_id, after_seq=first.seq).__aiter__()
    frames.append(await gen.__anext__())

    third = await slog.append(session_id, ContentEvent(text="three"))
    stream.publish(session_id, third)
    frames.append(await gen.__anext__())
    await gen.aclose()

    assert f"id: {second.seq}" in frames[0], "replay starts after the cursor, not before it"
    assert "one" not in frames[0]
    assert f"id: {third.seq}" in frames[1] and "three" in frames[1]


async def test_streaming_first_token():
    """The first content event reaches the wire before the model has finished."""
    session_id = await _new_session()
    gen = api._event_stream(session_id, after_seq=0).__aiter__()
    finished = asyncio.Event()

    async def slow_model() -> None:
        stored = await slog.append(session_id, ContentEvent(text="Hel"))
        stream.publish(session_id, stored)
        await asyncio.sleep(0.5)
        stored = await slog.append(session_id, ContentEvent(text="lo"))
        stream.publish(session_id, stored)
        finished.set()

    task = asyncio.create_task(slow_model())
    frame = await asyncio.wait_for(gen.__anext__(), timeout=5)

    assert "Hel" in frame
    assert not finished.is_set(), "the first token waited for the whole completion"
    await task
    await gen.aclose()


async def test_a_mid_stream_failure_sends_an_error_chunk_not_a_truncation(monkeypatch):
    """A stream that fails mid-flight ends with an error frame."""
    session_id = await _new_session()

    async def boom(*a, **kw):
        raise RuntimeError("the database went away")

    monkeypatch.setattr(slog, "get_events", boom)
    frames = [frame async for frame in api._event_stream(session_id, after_seq=0)]

    assert frames, "a failed stream must say so"
    assert frames[-1].startswith("event: error")
    body = json.loads(frames[-1].split("data: ", 1)[1])
    assert body["code"] == "stream_failed"
    assert body["retryable"] is True


async def test_a_lagging_subscriber_rejoins_from_the_log_instead_of_losing_events(monkeypatch):
    """A subscriber whose queue overflows catches up from the log."""
    session_id = await _new_session()
    # A two-slot queue, so three appends overflow it.
    monkeypatch.setattr(api, "stream", SessionStream(queue_size=2))

    gen = api._event_stream(session_id, after_seq=0).__aiter__()
    await asyncio.sleep(0)  # let it subscribe before anything is published

    for _ in range(3):
        api.stream.publish(session_id, await slog.append(session_id, ContentEvent(text="x")))
    last = await slog.append(session_id, DoneEvent(reason="turn_end"))
    api.stream.publish(session_id, last)

    seen = []
    for _ in range(10):
        frame = await asyncio.wait_for(gen.__anext__(), timeout=5)
        seen.append(frame)
        if f"id: {last.seq}" in frame:
            break
    await gen.aclose()

    assert any(f"id: {last.seq}" in f for f in seen), "the lagging subscriber never caught up"


async def _new_session() -> str:
    user_id = uuid.uuid4()
    await pool.execute("INSERT INTO users (id) VALUES ($1)", user_id)
    _seeded.append(user_id)
    session_id = await pool.fetchval(
        "INSERT INTO sessions (user_id, mode, status) VALUES ($1, 'attended', 'idle') RETURNING id",
        user_id,
    )
    return str(session_id)
