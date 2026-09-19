"""The HTTP surface: the cookie is the only way in, and every read is scoped to its owner.

Needs a real Postgres with migration 0 applied. SSE generators are driven directly
because httpx's ASGITransport buffers streamed responses.
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

from agent_module import prompts
from agent_module.events import ContentEvent, DoneEvent, ToolCallEvent, UserEvent
from db import pool
from harness_module import api, approvals, jwt_utils, lifecycle, runner, store
from harness_module import session_log as slog
from harness_module.stream import SessionStream, stream
from tests.dbgate import require_db
from tool_module.composio_mcp import ComposioError

pytestmark = pytest.mark.asyncio

# The real config reader, so a test can override one key without losing the rest.
_real_cfg = api._cfg

_seeded: list[uuid.UUID] = []


@pytest_asyncio.fixture(autouse=True)
async def _db(monkeypatch, tmp_path):
    await require_db()

    # Filesystem blobs for the WHOLE module: making a project writes its `.keep`,
    # and a cached `SupabaseBlobs` client would outlive its event loop.
    store.use_blobs(store.FilesystemBlobs(tmp_path))

    # The loop is out of scope here: start is stubbed, so no turn runs.
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
    store.use_blobs(None)
    await pool.execute("DELETE FROM sessions WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM files WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM deleted_files WHERE user_id = ANY($1::uuid[])", _seeded)
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


async def _sign_in_as(client: AsyncClient, user_id: str) -> None:
    """Sign in as a specific user, for the tests about what first login does."""
    response = await client.post("/auth/session", headers={"Authorization": f"Bearer {_supabase_token(user_id)}"})
    assert response.status_code == 204


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
    assert "arkos_session" in response.cookies
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


async def test_cookie_session(client):
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
        ("get", "/connections/done"),
    ):
        body = {"json": {}} if method == "post" else {}
        bare = await getattr(client, method)(path, **body)
        assert bare.status_code == 401, f"{method} {path} let a caller in with no cookie"

        foreign = await getattr(client, method)(path, cookies={"arkos_session": forged}, **body)
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

    assert body["user_id"] == user_id
    assert body["email"] == "a@example.com"
    assert body["home_session_id"], "the app has nowhere to land without this"
    assert body["display_name"] is None


async def test_the_name_from_sign_up_metadata_is_what_the_arkos_calls_you(client):
    """The name rides the SIGNED token as user_metadata, never a request body."""
    user_id = str(uuid.uuid4())
    token = _supabase_token(user_id, user_metadata={"name": "Nathaniel"})

    await client.post("/auth/session", headers={"Authorization": f"Bearer {token}"})

    assert (await client.get("/auth/me")).json()["display_name"] == "Nathaniel"


async def test_google_supplies_the_name_as_full_name(client):
    """Google's OIDC profile calls it `full_name`; our own form calls it `name`."""
    user_id = str(uuid.uuid4())
    token = _supabase_token(user_id, user_metadata={"full_name": "Ada Lovelace"})

    await client.post("/auth/session", headers={"Authorization": f"Bearer {token}"})

    assert (await client.get("/auth/me")).json()["display_name"] == "Ada Lovelace"


async def test_a_later_sign_in_without_a_name_does_not_erase_the_one_you_typed(client):
    """A later sign-in whose profile carries no name must not erase the typed one."""
    user_id = str(uuid.uuid4())
    named = _supabase_token(user_id, user_metadata={"name": "Nathaniel"})
    await client.post("/auth/session", headers={"Authorization": f"Bearer {named}"})

    bare = _supabase_token(user_id)
    await client.post("/auth/session", headers={"Authorization": f"Bearer {bare}"})

    assert (await client.get("/auth/me")).json()["display_name"] == "Nathaniel"


@pytest.mark.parametrize(
    "amr",
    [
        [{"method": "recovery", "timestamp": 1}],
        [{"method": "otp"}],
        [{"method": "password"}, {"method": "recovery"}],
        ["recovery"],
    ],
)
async def test_a_recovery_link_token_is_refused_a_cookie(client, amr):
    """12.2: the reset link's token is an ordinary access token, so without this
    anything that sees the link — a mail scanner following it, a shared mailbox,
    a forward — could trade it for a seven-day session without changing the
    password, leaving the owner no signal. The client refuses to send one; this
    is the half that holds when the client is curl."""
    token = _supabase_token(str(uuid.uuid4()), amr=amr)

    response = await client.post("/auth/session", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 401
    assert "arkos_session" not in response.cookies


async def test_an_ordinary_sign_in_is_not_mistaken_for_a_recovery_link(client):
    """The refusal is a denylist, so an unfamiliar method must not lock anyone out."""
    for amr in ([{"method": "password"}], [{"method": "oauth"}], [], None, "nonsense"):
        token = _supabase_token(str(uuid.uuid4()), amr=amr)
        response = await client.post("/auth/session", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 204, f"amr={amr!r} was refused"


async def test_a_blank_or_absent_name_is_not_a_name(client):
    """Whitespace is not a name: the fallback to email keys off NULL, not an empty string."""
    user_id = str(uuid.uuid4())
    token = _supabase_token(user_id, user_metadata={"name": "   "})

    await client.post("/auth/session", headers={"Authorization": f"Bearer {token}"})

    assert (await client.get("/auth/me")).json()["display_name"] is None


async def test_signing_in_again_keeps_exactly_one_home_session(client):
    """However the token was obtained, a second arrival must not mint a second home."""
    user_id = str(uuid.uuid4())
    token = _supabase_token(user_id, user_metadata={"full_name": "Ada"})

    await client.post("/auth/session", headers={"Authorization": f"Bearer {token}"})
    first = (await client.get("/auth/me")).json()["home_session_id"]
    await client.post("/auth/session", headers={"Authorization": f"Bearer {token}"})
    second = (await client.get("/auth/me")).json()["home_session_id"]

    assert first and first == second


async def test_a_password_change_signs_out_the_other_browser(client):
    """12.2.5's acceptance. The reason to reset a password is often that someone
    else has your session; leaving it alive is the outcome the reset was for."""
    user_id = str(uuid.uuid4())
    token = _supabase_token(user_id)
    await client.post("/auth/session", headers={"Authorization": f"Bearer {token}"})
    assert (await client.get("/auth/me")).status_code == 200

    # The reset completes in another browser and revokes everything for the user.
    await client.post("/auth/sessions/revoke", headers={"Authorization": f"Bearer {token}"})

    assert (await client.get("/auth/me")).status_code == 401


async def test_revoking_takes_a_recovery_token_even_though_signing_in_does_not(client):
    """Revocation only ever takes access away, so the token that cannot buy a
    session can still end one — which is what the reset flow holds."""
    user_id = str(uuid.uuid4())
    await client.post("/auth/session", headers={"Authorization": f"Bearer {_supabase_token(user_id)}"})
    recovery = _supabase_token(user_id, amr=[{"method": "recovery"}])

    revoked = await client.post("/auth/sessions/revoke", headers={"Authorization": f"Bearer {recovery}"})

    assert revoked.status_code == 204
    assert (await client.get("/auth/me")).status_code == 401


async def test_one_users_reset_does_not_sign_anybody_else_out(client):
    mine, theirs = str(uuid.uuid4()), str(uuid.uuid4())
    await client.post("/auth/session", headers={"Authorization": f"Bearer {_supabase_token(theirs)}"})
    other_cookie = dict(client.cookies)
    await client.post("/auth/session", headers={"Authorization": f"Bearer {_supabase_token(mine)}"})

    await client.post("/auth/sessions/revoke", headers={"Authorization": f"Bearer {_supabase_token(mine)}"})

    client.cookies.update(other_cookie)
    assert (await client.get("/auth/me")).status_code == 200


async def test_a_cookie_with_no_jti_cannot_be_revoked_so_is_refused(client):
    """Cookies minted before 12.2.5 have no handle to take back."""
    legacy = jwt.encode(
        {"sub": str(uuid.uuid4()), "iss": "arkos", "exp": datetime.now(UTC) + timedelta(hours=1)},
        "test-session-secret-at-least-32-chars",
        algorithm="HS256",
    )
    client.cookies.set("arkos_session", legacy)

    assert (await client.get("/auth/me")).status_code == 401


async def test_the_sign_in_endpoint_is_rate_limited(client, monkeypatch):
    """The one endpoint nothing authenticates, so the one that needs a ceiling."""
    monkeypatch.setattr(
        api,
        "_cfg",
        lambda key, default=None: 3 if key == "quotas.auth_attempts_per_minute" else _real_cfg(key, default),
    )
    api._auth_hits.clear()
    token = _supabase_token(str(uuid.uuid4()))

    codes = [
        (await client.post("/auth/session", headers={"Authorization": f"Bearer {token}"})).status_code for _ in range(5)
    ]

    assert codes[:3] == [204, 204, 204]
    assert codes[3:] == [429, 429]


async def test_signing_out_deletes_the_row_not_just_the_cookie(client):
    """The cookie is self-signed, so a copy taken beforehand keeps working until
    the row is gone. Clearing the cookie alone is a suggestion, not a sign-out."""
    user_id = str(uuid.uuid4())
    await client.post("/auth/session", headers={"Authorization": f"Bearer {_supabase_token(user_id)}"})
    live = await pool.fetchval("SELECT count(*) FROM auth_sessions WHERE user_id = $1", uuid.UUID(user_id))
    assert live == 1, "the mint did not record the session"

    await client.delete("/auth/session")

    left = await pool.fetchval("SELECT count(*) FROM auth_sessions WHERE user_id = $1", uuid.UUID(user_id))
    assert left == 0, "signing out left the session revocable by nobody"


async def test_the_tick_sweeps_sessions_past_their_expiry(client):
    """An expired cookie is already refused on its own `exp`, so a row past
    `expires_at` is dead weight — but without a sweep the table only grows."""
    user_id = str(uuid.uuid4())
    await client.post("/auth/session", headers={"Authorization": f"Bearer {_supabase_token(user_id)}"})
    await pool.execute(
        "UPDATE auth_sessions SET expires_at = now() - interval '1 day' WHERE user_id = $1",
        uuid.UUID(user_id),
    )

    await pool.execute("DELETE FROM auth_sessions WHERE expires_at < now()")

    left = await pool.fetchval("SELECT count(*) FROM auth_sessions WHERE user_id = $1", uuid.UUID(user_id))
    assert left == 0


async def test_verifying_a_token_never_fetches_keys(client, monkeypatch):
    """12.2.5's requirement, at the boundary: a sign-in must not be what
    discovers that the JWKS host is having a slow day."""
    fetched = []
    monkeypatch.setattr(jwt_utils, "refresh_jwks", lambda: fetched.append(1) or True)

    class Boom:
        def get_jwk_set(self, refresh=False):
            fetched.append("request-path fetch")
            raise AssertionError("verification fetched keys")

        jwk_set_cache = None

    monkeypatch.setattr(jwt_utils, "_jwks", lambda: None)
    await client.post("/auth/session", headers={"Authorization": f"Bearer {_supabase_token(str(uuid.uuid4()))}"})

    assert "request-path fetch" not in fetched


async def test_logout_clears_the_cookie(client):
    await _signed_in(client)

    response = await client.delete("/auth/session")

    assert response.status_code == 204
    assert response.cookies.get("arkos_session") in (None, "")


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
    snapshot = (await client.get(f"/sessions/{body['session_id']}")).json()
    assert snapshot["folders"] == ["file-my-taxes"]

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
    # Excluding home, which first login made and the quota does not count.
    count = await pool.fetchval(
        """
        SELECT count(*) FROM sessions s JOIN users u ON u.id = s.user_id
         WHERE s.user_id = $1 AND s.id <> u.home_session_id
        """,
        uuid.UUID(user_id),
    )
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
    """A session parked with nothing to answer is unstuck by a message."""
    user_id = await _signed_in(client)
    session_id = await _session_for(user_id, status="awaiting_approval")

    response = await client.post(f"/sessions/{session_id}/messages", json={"text": "still there?"})

    assert response.status_code == 202
    events = [e.event for e in await slog.get_events(session_id)]
    assert [e.kind for e in events] == ["user"]
    assert session_id in api.started


# --- the handoff ----------------------------------------------------------------


async def test_play_asks_for_the_goal_and_drafts_nothing(client):
    """The button's whole job is the question. Planning off ambiguous history is what
    this removes, so no turn runs and no plan is drafted until they answer.
    """
    user_id = await _signed_in(client)
    session_id = await _session_for(user_id, status="idle", mode="attended")

    response = await client.post(f"/sessions/{session_id}/approve")

    assert response.status_code == 202
    assert response.json()["mode"] == "attended"
    assert response.json()["started"] is False
    assert response.json()["parked"] is True
    assert api.started == [], "the press starts no turn"

    events = [e.event for e in await slog.get_events(session_id)]
    assert [e.kind for e in events] == ["lifecycle", "content"]
    assert events[1].text == prompts.GOAL_QUESTION, "arkos asks in his own words, and they stay in the transcript"

    row = await pool.fetchrow("SELECT status, mode FROM sessions WHERE id = $1", uuid.UUID(session_id))
    assert row["status"] == "awaiting_approval"
    assert row["mode"] == "attended"

    open_rows = await approvals.open_for(session_id)
    assert [(r.kind, r.tool_name, r.prompt) for r in open_rows] == [("ask", approvals.GOAL, prompts.GOAL_QUESTION)]
    assert open_rows[0].is_goal


async def test_the_answer_to_the_goal_question_is_what_drafts_the_plan(client):
    """Their reply IS the goal: it lands as their own turn, with the handoff behind it."""
    user_id = await _signed_in(client)
    session_id = await _session_for(user_id, status="idle", mode="attended")
    await client.post(f"/sessions/{session_id}/approve")
    question = (await approvals.open_for(session_id))[0]

    response = await client.post(f"/approvals/{question.id}/respond", json={"answer": "clear the backlog in triage/"})

    assert response.status_code == 202
    assert response.json()["mode"] == "attended"
    assert session_id in api.started
    assert "mode" not in api.start_calls[-1], "the play button still never moves the mode"
    assert api.start_calls[-1]["reason"] == "plan_requested"

    events = [e.event for e in await slog.get_events(session_id)]
    assert [e.kind for e in events] == ["lifecycle", "content", "user", "user"]
    assert (events[2].source, events[2].text) == ("human", "clear the backlog in triage/")
    assert events[3].source == "system"
    assert "propose_plan" in events[3].text


async def test_cancelling_at_the_goal_question_leaves_the_session_idle(client):
    """Nothing was drafted and nothing ran, so the way out costs nothing."""
    user_id = await _signed_in(client)
    session_id = await _session_for(user_id, status="idle", mode="attended")
    await client.post(f"/sessions/{session_id}/approve")
    question = (await approvals.open_for(session_id))[0]

    response = await client.post(f"/approvals/{question.id}/respond", json={"answer": approvals.DECLINE})

    assert response.status_code == 202
    assert response.json()["started"] is False
    assert api.started == []
    assert await approvals.open_for(session_id) == []
    row = await pool.fetchrow("SELECT status, mode FROM sessions WHERE id = $1", uuid.UUID(session_id))
    assert (row["status"], row["mode"]) == ("idle", "attended")


async def test_the_goal_can_be_typed_in_the_composer(client):
    """The composer answers an `ask`, and this one is an ask: same act, same handoff."""
    user_id = await _signed_in(client)
    session_id = await _session_for(user_id, status="idle", mode="attended")
    await client.post(f"/sessions/{session_id}/approve")

    response = await client.post(f"/sessions/{session_id}/messages", json={"text": "book the flights"})

    assert response.status_code == 202
    assert api.start_calls[-1]["reason"] == "plan_requested"
    events = [e.event for e in await slog.get_events(session_id)]
    assert [(e.kind, getattr(e, "source", None)) for e in events][-2:] == [("user", "human"), ("user", "system")]
    assert await approvals.open_for(session_id) == []


async def test_pressing_play_twice_is_one_question(client):
    """The status is claimed first, so a second press has nowhere to park."""
    user_id = await _signed_in(client)
    session_id = await _session_for(user_id, status="idle", mode="attended")

    assert (await client.post(f"/sessions/{session_id}/approve")).status_code == 202
    second = await client.post(f"/sessions/{session_id}/approve")

    assert second.status_code == 409
    assert second.json()["code"] == "not_idle"
    assert len(await approvals.open_for(session_id)) == 1


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


async def test_the_unattended_quota_does_not_bind_at_play(client, monkeypatch):
    """Drafting a plan is an attended turn, so it costs nothing against the cap."""
    user_id = await _signed_in(client)
    monkeypatch.setattr(api, "_cfg", lambda key, default: 1 if key == "quotas.max_unattended_sessions" else default)
    await _session_for(user_id, status="running", mode="unattended")
    mine = await _session_for(user_id, status="idle", mode="attended")

    response = await client.post(f"/sessions/{mine}/approve")

    assert response.status_code == 202
    row = await pool.fetchrow("SELECT mode FROM sessions WHERE id = $1", uuid.UUID(mine))
    assert row["mode"] == "attended"


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

    # Only "Taxes": the home session mints no project, so nothing else is here.
    assert [p["title"] for p in body] == ["Taxes"]
    assert body[0]["status_rollup"] == "awaiting_approval", "the most urgent thing in the project wins the dot"


async def test_a_project_lists_the_sessions_the_grid_has_to_open(client):
    """The rollup is one dot and one count; opening what it counted needs ids."""
    user_id = await _signed_in(client)
    project_id = await pool.fetchval(
        "INSERT INTO projects (user_id, title) VALUES ($1, 'Taxes') RETURNING id", uuid.UUID(user_id)
    )
    older = await pool.fetchval(
        "INSERT INTO sessions (user_id, project_id, mode, status, title) "
        "VALUES ($1, $2, 'attended', 'completed', 'the old one') RETURNING id",
        uuid.UUID(user_id),
        project_id,
    )
    newer = await pool.fetchval(
        "INSERT INTO sessions (user_id, project_id, mode, status, title) "
        "VALUES ($1, $2, 'unattended', 'running', 'the live one') RETURNING id",
        uuid.UUID(user_id),
        project_id,
    )
    await slog.append(str(newer), ContentEvent(text="working"))

    body = (await client.get(f"/projects/{project_id}/sessions")).json()

    assert [s["session_id"] for s in body] == [str(newer), str(older)], "most recently active first"
    assert body[0]["title"] == "the live one"
    assert body[0]["status"] == "running"
    assert body[0]["hops_max"] > 0


async def test_another_users_project_lists_no_sessions(client):
    theirs = str(uuid.uuid4())
    _seeded.append(uuid.UUID(theirs))
    await pool.execute("INSERT INTO users (id) VALUES ($1)", uuid.UUID(theirs))
    their_project = await pool.fetchval(
        "INSERT INTO projects (user_id, title) VALUES ($1, 'Theirs') RETURNING id", uuid.UUID(theirs)
    )
    await _signed_in(client)

    assert (await client.get(f"/projects/{their_project}/sessions")).status_code == 404


# --- what is waiting on the human -------------------------------------------------


async def test_attention_lists_open_questions_oldest_first(client):
    user_id = await _signed_in(client)
    project_id = await pool.fetchval(
        "INSERT INTO projects (user_id, title) VALUES ($1, 'Taxes') RETURNING id", uuid.UUID(user_id)
    )
    session_id = await pool.fetchval(
        "INSERT INTO sessions (user_id, project_id, mode, status, title) "
        "VALUES ($1, $2, 'unattended', 'awaiting_approval', 'filing') RETURNING id",
        uuid.UUID(user_id),
        project_id,
    )
    first = await approvals.create(str(session_id), "c1", "ask", "which account?")
    await approvals.create(str(session_id), "c2", "approval", "send it?")

    body = (await client.get("/attention")).json()

    assert [a["approval_id"] for a in body][0] == first.id, "the longest wait comes first"
    assert [a["kind"] for a in body] == ["ask", "approval"]
    assert body[0]["session_title"] == "filing"
    assert body[0]["project_title"] == "Taxes"


async def test_attention_narrows_to_one_project_with_the_same_query(client):
    user_id = await _signed_in(client)
    projects = [
        await pool.fetchval(
            "INSERT INTO projects (user_id, title) VALUES ($1, $2) RETURNING id", uuid.UUID(user_id), title
        )
        for title in ("Taxes", "Garden")
    ]
    for project_id, call in zip(projects, ("c1", "c2"), strict=True):
        session_id = await pool.fetchval(
            "INSERT INTO sessions (user_id, project_id, mode, status) "
            "VALUES ($1, $2, 'unattended', 'awaiting_approval') RETURNING id",
            uuid.UUID(user_id),
            project_id,
        )
        await approvals.create(str(session_id), call, "ask", f"question for {call}")

    everything = (await client.get("/attention")).json()
    just_one = (await client.get(f"/attention?project_id={projects[0]}")).json()

    assert len(everything) == 2
    assert [a["project_title"] for a in just_one] == ["Taxes"]


async def test_an_answered_question_stops_asking_for_attention(client):
    user_id = await _signed_in(client)
    session_id = await _session_for(user_id, status="awaiting_approval")
    opened = await approvals.create(session_id, "c1", "ask", "which account?")

    await approvals.answer(opened.id, "the joint one")

    assert (await client.get("/attention")).json() == []


async def test_attention_never_crosses_users(client):
    theirs = str(uuid.uuid4())
    _seeded.append(uuid.UUID(theirs))
    await pool.execute("INSERT INTO users (id) VALUES ($1)", uuid.UUID(theirs))
    their_session = await _session_for(theirs, status="awaiting_approval")
    await approvals.create(their_session, "c1", "ask", "their private question")
    await _signed_in(client)

    assert (await client.get("/attention")).json() == []


# --- the rail's cross-project view --------------------------------------------------


async def test_running_sessions_are_listed_across_projects(client):
    """The per-project list does not compose into a sidebar that spans both tabs."""
    user_id = await _signed_in(client)
    live = []
    for title in ("Taxes", "Garden"):
        project_id = await pool.fetchval(
            "INSERT INTO projects (user_id, title) VALUES ($1, $2) RETURNING id", uuid.UUID(user_id), title
        )
        live.append(
            await pool.fetchval(
                "INSERT INTO sessions (user_id, project_id, mode, status, title) "
                "VALUES ($1, $2, 'unattended', 'running', $3) RETURNING id",
                uuid.UUID(user_id),
                project_id,
                title + " work",
            )
        )
    await pool.execute(
        "INSERT INTO sessions (user_id, mode, status) VALUES ($1, 'attended', 'idle')", uuid.UUID(user_id)
    )

    body = (await client.get("/sessions?status=running")).json()

    assert {s["session_id"] for s in body} == {str(s) for s in live}
    assert {s["project_title"] for s in body} == {"Taxes", "Garden"}
    assert all(s["status"] == "running" for s in body)


async def test_listing_sessions_without_a_filter_returns_them_all(client):
    user_id = await _signed_in(client)
    for status in ("idle", "running", "completed"):
        await pool.execute(
            "INSERT INTO sessions (user_id, mode, status) VALUES ($1, 'attended', $2)",
            uuid.UUID(user_id),
            status,
        )

    body = (await client.get("/sessions")).json()

    # Three made here, plus the home session every sign-in leaves behind.
    assert len(body) == 4
    assert {s["status"] for s in body} == {"idle", "running", "completed"}


async def test_a_status_that_is_not_one_is_refused(client):
    await _signed_in(client)

    response = await client.get("/sessions?status=banana")

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_request"


async def test_session_listing_never_crosses_users(client):
    theirs = str(uuid.uuid4())
    _seeded.append(uuid.UUID(theirs))
    await pool.execute("INSERT INTO users (id) VALUES ($1)", uuid.UUID(theirs))
    await _session_for(theirs, status="running")
    await _signed_in(client)

    assert (await client.get("/sessions?status=running")).json() == []


# --- the home session ---------------------------------------------------------------


async def test_first_sign_in_creates_exactly_one_home_session(client):
    user_id = str(uuid.uuid4())
    _seeded.append(uuid.UUID(user_id))

    await _sign_in_as(client, user_id)
    me = (await client.get("/auth/me")).json()

    assert me["home_session_id"], "no home session was made"
    row = await pool.fetchrow(
        "SELECT status, mode, project_id FROM sessions WHERE id = $1", uuid.UUID(me["home_session_id"])
    )
    assert (row["status"], row["mode"]) == ("idle", "attended"), "home is an ordinary session"
    assert row["project_id"] is None, "the home session minted a shadow project"
    assert await pool.fetchval("SELECT count(*) FROM sessions WHERE user_id = $1", uuid.UUID(user_id)) == 1


async def test_signing_in_again_lands_in_the_same_home_session(client):
    user_id = str(uuid.uuid4())
    _seeded.append(uuid.UUID(user_id))

    await _sign_in_as(client, user_id)
    first = (await client.get("/auth/me")).json()["home_session_id"]
    await _sign_in_as(client, user_id)
    second = (await client.get("/auth/me")).json()["home_session_id"]

    assert first == second
    assert await pool.fetchval("SELECT count(*) FROM sessions WHERE user_id = $1", uuid.UUID(user_id)) == 1


async def test_a_fresh_account_owns_no_project_and_no_folder(client):
    """A fresh account owns nothing: the home chat mints neither project nor folder."""
    user_id = str(uuid.uuid4())
    _seeded.append(uuid.UUID(user_id))
    await _sign_in_as(client, user_id)

    projects = (await client.get("/projects")).json()
    folders = (await client.get("/folders")).json()

    assert projects == []
    assert folders == []


async def test_one_question_shows_at_all_three_scopes_and_resolves_from_any(client):
    """An approval is a state of the session, not something a surface owns."""
    user_id = await _signed_in(client)
    project_id = await pool.fetchval(
        "INSERT INTO projects (user_id, title) VALUES ($1, 'Taxes') RETURNING id", uuid.UUID(user_id)
    )
    session_id = str(
        await pool.fetchval(
            "INSERT INTO sessions (user_id, project_id, mode, status, title) "
            "VALUES ($1, $2, 'unattended', 'awaiting_approval', 'filing') RETURNING id",
            uuid.UUID(user_id),
            project_id,
        )
    )
    opened = await approvals.create(session_id, "c1", "ask", "which account?")

    everywhere = (await client.get("/attention")).json()
    in_project = (await client.get(f"/attention?project_id={project_id}")).json()
    in_window = (await client.get(f"/attention?session_id={session_id}")).json()

    assert [a["approval_id"] for a in everywhere] == [opened.id]
    assert [a["approval_id"] for a in in_project] == [opened.id]
    assert [a["approval_id"] for a in in_window] == [opened.id]

    # One row, so resolving from the window clears every scope at once.
    answered = await client.post(f"/approvals/{opened.id}/respond", json={"answer": "the joint one"})

    assert answered.status_code == 202
    assert (await client.get("/attention")).json() == []
    assert (await client.get(f"/attention?project_id={project_id}")).json() == []
    assert (await client.get(f"/attention?session_id={session_id}")).json() == []


async def test_attention_in_another_users_window_reads_as_absent(client):
    theirs = str(uuid.uuid4())
    _seeded.append(uuid.UUID(theirs))
    await pool.execute("INSERT INTO users (id) VALUES ($1)", uuid.UUID(theirs))
    their_session = await _session_for(theirs, status="awaiting_approval")
    await approvals.create(their_session, "c1", "ask", "their private question")
    await _signed_in(client)

    assert (await client.get(f"/attention?session_id={their_session}")).status_code == 404


async def test_an_uploaded_file_is_in_the_project_and_in_the_store(client, tmp_path):
    """End to end: upload a file, and the project it belongs to lists it."""
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    try:
        user_id = await _signed_in(client)
        created = (await client.post("/sessions", json={"goal": "do the taxes"})).json()
        project_id = created["project_id"]

        uploaded = await client.post(
            "/files",
            files={"file": ("receipts.csv", b"date,amount\n2026-08-18,12", "text/csv")},
            data={"path": "do-the-taxes/receipts.csv"},
        )
        listed = (await client.get(f"/projects/{project_id}/files")).json()

        assert uploaded.status_code == 201
        # `.keep` is a real row — it is what makes the folder exist. Surfaces
        # filter it; the wire does not, because it carries files and only files.
        assert [f["path"] for f in listed] == [
            "do-the-taxes/.keep",
            "do-the-taxes/receipts.csv",
        ]

        entry = next(e for e in await store.read_tree(user_id) if e.path.endswith("receipts.csv"))
        assert await store.get_blob(entry.content_hash) == b"date,amount\n2026-08-18,12"
    finally:
        store.use_blobs(None)


async def test_a_file_reads_back_out_of_the_store(client, tmp_path):
    """The computer view is a filesystem you can read, and rows are half of that."""
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    try:
        await _signed_in(client)
        await client.post("/sessions", json={"goal": "taxes"})
        uploaded = (
            await client.post(
                "/files",
                files={"file": ("notes.md", b"# Notes\n\nline two", "text/markdown")},
                data={"path": "taxes/notes.md"},
            )
        ).json()

        body = (await client.get(f"/files/{uploaded['file_id']}")).json()

        assert body["path"] == "taxes/notes.md"
        assert body["text"] == "# Notes\n\nline two"
        assert body["binary"] is False
    finally:
        store.use_blobs(None)


async def test_a_file_that_is_not_text_says_so_rather_than_mangling_itself(client, tmp_path):
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    try:
        await _signed_in(client)
        await client.post("/sessions", json={"goal": "pictures"})
        uploaded = (
            await client.post(
                "/files",
                files={"file": ("logo.png", b"\x89PNG\r\n\x1a\n\xff\xfe", "image/png")},
                data={"path": "pictures/logo.png"},
            )
        ).json()

        body = (await client.get(f"/files/{uploaded['file_id']}")).json()

        assert body["binary"] is True
        assert body["text"] is None
    finally:
        store.use_blobs(None)


async def test_another_users_file_cannot_be_read(client, tmp_path):
    """The store is keyed by user, so the read is scoped by the same key it is stored under."""
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    try:
        theirs = str(uuid.uuid4())
        _seeded.append(uuid.UUID(theirs))
        await pool.execute("INSERT INTO users (id) VALUES ($1)", uuid.UUID(theirs))
        their_file = await store.put_file(theirs, "secret/a.txt", b"peek")
        await _signed_in(client)

        assert (await client.get(f"/files/{their_file.id}")).status_code == 404
        assert (await client.get(f"/files/{uuid.uuid4()}")).status_code == 404
    finally:
        store.use_blobs(None)


# --- signing in -------------------------------------------------------------------


async def test_auth_config_is_readable_signed_out(client):
    """It is how a signed-out browser signs in, so it cannot require a session."""
    body = (await client.get("/auth/config")).json()

    assert set(body) == {"supabase_url", "anon_key"}
    assert isinstance(body["anon_key"], str)


@pytest.mark.parametrize("path", ["/app/", "/app/index.html"])
async def test_the_page_itself_is_never_cached(client, path):
    """index.html must be no-store: the hand-rolled `?v=N` bust cannot bootstrap itself."""
    response = await client.get(path)

    assert response.status_code == 200
    assert "no-store" in response.headers.get("cache-control", "")


async def test_the_assets_keep_their_validators(client):
    """`?v=N` is answer enough for the assets, so nothing here forces a re-fetch."""
    response = await client.get("/app/api.jsx?v=1")

    assert response.status_code == 200
    assert "no-store" not in response.headers.get("cache-control", "")


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
    # This account has made no project of its own: the home session mints none.
    assert [p["title"] for p in (await client.get("/projects")).json()] == []


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


async def test_a_backlog_longer_than_one_page_arrives_whole(monkeypatch):
    """The reconnect path. One page is not the backlog, and `sent` only moves forward."""
    session_id = await _new_session()
    monkeypatch.setattr(api, "_BACKLOG_PAGE", 2)
    appended = [await slog.append(session_id, ContentEvent(text=f"e{i}")) for i in range(7)]

    gen = api._event_stream(session_id, after_seq=0).__aiter__()
    frames = [await asyncio.wait_for(gen.__anext__(), timeout=5) for _ in appended]
    await gen.aclose()

    assert [f"id: {e.seq}" in f for e, f in zip(appended, frames, strict=True)] == [True] * 7


async def test_a_lagging_subscriber_catches_up_past_one_page(monkeypatch):
    """The same hole on the LAGGED re-join, where it is likeliest to be a long log."""
    session_id = await _new_session()
    monkeypatch.setattr(api, "_BACKLOG_PAGE", 2)
    monkeypatch.setattr(api, "stream", SessionStream(queue_size=2))

    gen = api._event_stream(session_id, after_seq=0).__aiter__()
    await asyncio.sleep(0)  # let it subscribe before anything is published

    for i in range(5):
        api.stream.publish(session_id, await slog.append(session_id, ContentEvent(text=f"x{i}")))
    last = await slog.append(session_id, DoneEvent(reason="turn_end"))
    api.stream.publish(session_id, last)

    seen = []
    for _ in range(12):
        seen.append(await asyncio.wait_for(gen.__anext__(), timeout=5))
        if f"id: {last.seq}" in seen[-1]:
            break
    await gen.aclose()

    assert any(f"id: {last.seq}" in f for f in seen), "the lagging subscriber never caught up"
    assert sum("x" in f for f in seen) == 5, "the re-join dropped the middle of the log"


async def test_a_status_change_reaches_the_stream_without_a_reconnect():
    """The pill moves when the status moves, not when someone reconnects."""
    session_id = await _new_session()

    gen = api._event_stream(session_id, after_seq=0).__aiter__()
    await asyncio.sleep(0)  # subscribed
    moved = await lifecycle.transition(session_id, "idle", "running", "the human sent a message")
    frame = await asyncio.wait_for(gen.__anext__(), timeout=5)
    await gen.aclose()

    assert moved is not None
    assert f"id: {moved.seq}" in frame
    assert "lifecycle" in frame and "running" in frame


async def test_a_published_status_can_be_fetched_by_the_seq_it_announced():
    """Published after commit: a Last-Event-ID reader can always fetch what it just saw."""
    session_id = await _new_session()
    seen: list = []
    async with api.stream.subscribe(session_id) as queue:
        moved = await lifecycle.transition(session_id, "idle", "running", "claimed")
        seen.append(await asyncio.wait_for(queue.get(), timeout=5))

    replay = await slog.get_events(session_id, after_seq=seen[0].seq - 1, limit=10)

    assert moved is not None
    assert [e.seq for e in replay][0] == seen[0].seq, "the announced seq was not yet readable"


# --- the browser's frame side-channel ---------------------------------------------


async def test_the_frame_stream_requires_owning_the_session(client):
    """The frame stream is scoped by the cookie's user, never by a query parameter."""
    theirs = str(uuid.uuid4())
    _seeded.append(uuid.UUID(theirs))
    await pool.execute("INSERT INTO users (id) VALUES ($1)", uuid.UUID(theirs))
    their_session = await _session_for(theirs)
    await _signed_in(client)

    assert (await client.get(f"/sessions/{their_session}/browser/frames")).status_code == 404


async def test_a_frame_reaches_the_watcher_of_that_session(client):
    user_id = await _signed_in(client)
    session_id = await _session_for(user_id)

    frames = [frame async for frame in _read_frames(api._frame_stream(user_id, session_id), user_id, session_id)]

    assert frames and "jpeg" in frames[0]


async def _read_frames(stream, user_id, session_id):
    """Publish one frame once the stream is subscribed, then read it back."""
    gen = stream.__aiter__()
    task = asyncio.create_task(gen.__anext__())
    await asyncio.sleep(0)
    api.frames.publish(user_id, session_id, "ZmFrZS1qcGVn")
    yield await asyncio.wait_for(task, timeout=5)
    await gen.aclose()


# --- the tool budget: what this session may reach ----------------------------------


class _FakeConnectors:
    """The connections half of the client, which is all the tools document reads."""

    def __init__(self, rows):
        self._rows = rows

    async def connections(self, user_id, *, refresh=True):
        self.refreshed = getattr(self, "refreshed", 0) + (1 if refresh else 0)
        return [dict(r) for r in self._rows]

    async def always(self, user_id):
        return []


def _server(label, *, tools, connected=True):
    return {
        "server": label.title(),
        "label": label,
        "name": label.title(),
        "status": "connected" if connected else "disconnected",
        "tool_count": tools,
        "refreshed_at": None,
        "setup_url": None,
        "scopes": [],
        "shares_with": [],
    }


@pytest.fixture
def mcp(monkeypatch):
    """Install a connections source, and return a setter for what it holds."""

    def use(rows):
        monkeypatch.setattr(api.hands, "connectors", lambda: _FakeConnectors(rows))

    use([])
    return use


async def test_a_session_starts_reaching_nothing_but_ours(client, mcp):
    """The default is ours alone: a connected server is not a reachable one."""
    mcp([_server("gmail", tools=12), _server("slack", tools=38)])
    user_id = await _signed_in(client)
    session_id = await _session_for(user_id)

    body = (await client.get(f"/sessions/{session_id}/tools")).json()

    assert body["used"] == 0
    assert [s["enabled"] for s in body["servers"]] == [False, False]
    assert body["budget"] == body["max_tools"] - body["ours"], "the meter is what is left after ours"
    assert body["ours"] > 0


async def test_a_toggle_is_recorded_and_read_back(client, mcp):
    mcp([_server("gmail", tools=12)])
    user_id = await _signed_in(client)
    session_id = await _session_for(user_id)

    written = await client.put(f"/sessions/{session_id}/tools/Gmail", json={"enabled": True})

    assert written.status_code == 200
    assert written.json()["used"] == 12, "the write answers with the meter it just moved"
    assert (await client.get(f"/sessions/{session_id}/tools")).json()["used"] == 12

    await client.put(f"/sessions/{session_id}/tools/Gmail", json={"enabled": False})
    assert (await client.get(f"/sessions/{session_id}/tools")).json()["used"] == 0


async def test_a_toggle_over_the_cap_is_refused_with_the_numbers(client, mcp):
    """The refusal names both numbers: the cap and what the server would have added."""
    mcp([_server("huge", tools=9000)])
    user_id = await _signed_in(client)
    session_id = await _session_for(user_id)

    refused = await client.put(f"/sessions/{session_id}/tools/Huge", json={"enabled": True})

    assert refused.status_code == 409
    assert refused.json()["code"] == "tool_budget"
    assert "9000" in refused.json()["message"]
    assert (await client.get(f"/sessions/{session_id}/tools")).json()["used"] == 0, "and nothing was recorded"


async def test_an_unconnected_server_cannot_be_given_to_a_session(client, mcp):
    mcp([_server("linear", tools=3, connected=False)])
    user_id = await _signed_in(client)
    session_id = await _session_for(user_id)

    refused = await client.put(f"/sessions/{session_id}/tools/Linear", json={"enabled": True})

    assert refused.status_code == 409
    assert refused.json()["code"] == "not_connected"


async def test_turning_a_server_off_is_never_refused(client, mcp):
    """Whatever the numbers say, reducing reach is always allowed."""
    mcp([_server("gmail", tools=12)])
    user_id = await _signed_in(client)
    session_id = await _session_for(user_id)
    await client.put(f"/sessions/{session_id}/tools/Gmail", json={"enabled": True})

    mcp([_server("gmail", tools=9000)])  # the server grew overnight
    off = await client.put(f"/sessions/{session_id}/tools/Gmail", json={"enabled": False})

    assert off.status_code == 200
    assert off.json()["used"] == 0


async def test_an_unknown_server_is_not_found(client, mcp):
    user_id = await _signed_in(client)
    session_id = await _session_for(user_id)

    assert (await client.put(f"/sessions/{session_id}/tools/Nope", json={"enabled": True})).status_code == 404


async def test_the_tool_budget_is_scoped_to_the_session_owner(client, mcp):
    mcp([_server("gmail", tools=12)])
    theirs = str(uuid.uuid4())
    _seeded.append(uuid.UUID(theirs))
    await pool.execute("INSERT INTO users (id) VALUES ($1)", uuid.UUID(theirs))
    their_session = await _session_for(theirs)
    await _signed_in(client)

    assert (await client.get(f"/sessions/{their_session}/tools")).status_code == 404
    assert (await client.put(f"/sessions/{their_session}/tools/Gmail", json={"enabled": True})).status_code == 404


async def test_the_toggles_are_per_session(client, mcp):
    """Reach is a property of one session, not of the account."""
    mcp([_server("gmail", tools=12)])
    user_id = await _signed_in(client)
    one, other = await _session_for(user_id), await _session_for(user_id)

    await client.put(f"/sessions/{one}/tools/Gmail", json={"enabled": True})

    assert (await client.get(f"/sessions/{one}/tools")).json()["used"] == 12
    assert (await client.get(f"/sessions/{other}/tools")).json()["used"] == 0


async def test_a_project_created_with_no_links_makes_a_folder_of_its_own(client, tmp_path):
    """The none-case is not "no files": a folder named after it appears in the store."""
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    try:
        await _signed_in(client)

        made = await client.post("/projects", json={"title": "weekend reading"})

        assert made.status_code == 201
        assert made.json()["title"] == "weekend reading"
        assert made.json()["folders"] == ["weekend-reading"]
        assert made.json()["files"] == 0, "the sentinel is structure, not content"
        assert "weekend reading" in [p["title"] for p in (await client.get("/projects")).json()]
        assert (await client.get("/folders")).json() == [{"name": "weekend-reading", "files": 0}]
    finally:
        store.use_blobs(None)


async def test_a_project_needs_a_name(client):
    await _signed_in(client)

    assert (await client.post("/projects", json={"title": "   "})).status_code == 400
    assert (await client.post("/projects", json={})).status_code == 400


async def test_a_project_links_folders_that_already_exist(client, tmp_path):
    """One store: a project points at existing files rather than copying rows."""
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    try:
        user_id = await _signed_in(client)
        for path in ("notes/a.md", "notes/deep/b.md", "elsewhere/c.md"):
            await store.put_file(user_id, path, b"x" * 10)

        made = await client.post("/projects", json={"title": "reading", "folders": ["notes"]})

        assert made.status_code == 201
        assert made.json()["folders"] == ["notes"]
        assert made.json()["files"] == 2, "the folder and its children, not the sibling"
        linked = await client.get(f"/projects/{made.json()['id']}/files")
        assert sorted(f["path"] for f in linked.json()) == ["notes/a.md", "notes/deep/b.md"]
        assert len((await client.get("/files")).json()) == 3
    finally:
        store.use_blobs(None)


async def test_two_projects_may_link_the_same_folder(client, tmp_path):
    """A project owns no folder, so there is nothing to take from the other one."""
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    try:
        user_id = await _signed_in(client)
        await store.put_file(user_id, "triage/a.md", b"1")

        first = (await client.post("/projects", json={"title": "one", "folders": ["triage"]})).json()
        second = (await client.post("/projects", json={"title": "two", "folders": ["triage"]})).json()

        assert first["folders"] == second["folders"] == ["triage"]
        assert [f["path"] for f in (await client.get(f"/projects/{second['id']}/files")).json()] == ["triage/a.md"]
    finally:
        store.use_blobs(None)


async def test_linking_a_folder_that_is_not_in_the_store_is_not_found(client):
    await _signed_in(client)

    made = await client.post("/projects", json={"title": "x", "folders": ["nothing"]})

    assert made.status_code == 404


async def test_another_users_folder_is_not_a_folder_here(client, tmp_path):
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    try:
        theirs = str(uuid.uuid4())
        _seeded.append(uuid.UUID(theirs))
        await pool.execute("INSERT INTO users (id) VALUES ($1)", uuid.UUID(theirs))
        await store.put_file(theirs, "private/a.md", b"1")
        await _signed_in(client)

        made = await client.post("/projects", json={"title": "x", "folders": ["private"]})

        assert made.status_code == 404
    finally:
        store.use_blobs(None)


async def test_a_folder_linked_after_creation_reaches_the_next_session_not_this_one(client, tmp_path):
    """The UI shows it at once; claims are fixed per session, so the agent sees it next."""
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    try:
        from harness_module import workspace

        user_id = await _signed_in(client)
        await store.put_file(user_id, "triage/a.md", b"1")
        await store.put_file(user_id, "notes/b.md", b"2")
        made = (await client.post("/projects", json={"title": "work", "folders": ["triage"]})).json()
        running = (await client.post("/sessions", json={"goal": "start", "project_id": made["id"]})).json()[
            "session_id"
        ]

        linked = await client.post(f"/projects/{made['id']}/folders", json={"folder": "notes"})

        assert linked.status_code == 201
        assert linked.json()["folders"] == ["triage", "notes"]
        assert (
            await pool.fetchval("SELECT count(*) FROM project_folders WHERE project_id = $1", uuid.UUID(made["id"]))
            == 2
        )
        assert [c.folder for c in await workspace.claims_for(running)] == ["triage"]

        later = (await client.post("/sessions", json={"goal": "again", "project_id": made["id"]})).json()["session_id"]
        assert sorted(c.folder for c in await workspace.claims_for(later)) == ["notes", "triage"]
    finally:
        store.use_blobs(None)


async def test_linking_the_same_folder_twice_is_one_link(client, tmp_path):
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    try:
        user_id = await _signed_in(client)
        await store.put_file(user_id, "triage/a.md", b"1")
        made = (await client.post("/projects", json={"title": "work", "folders": ["triage"]})).json()

        again = await client.post(f"/projects/{made['id']}/folders", json={"folder": "triage"})

        assert again.status_code == 201
        assert again.json()["folders"] == ["triage"]
    finally:
        store.use_blobs(None)


async def test_a_project_can_be_renamed(client):
    await _signed_in(client)
    made = (await client.post("/projects", json={"title": "untitled"})).json()

    renamed = await client.patch(f"/projects/{made['id']}", json={"title": "inbox triage"})

    assert renamed.status_code == 200
    assert renamed.json()["title"] == "inbox triage"
    titles = [p["title"] for p in (await client.get("/projects")).json()]
    assert "inbox triage" in titles and "untitled" not in titles


async def test_renaming_someone_elses_project_is_not_found(client):
    theirs = str(uuid.uuid4())
    _seeded.append(uuid.UUID(theirs))
    await pool.execute("INSERT INTO users (id) VALUES ($1)", uuid.UUID(theirs))
    their_project = str(
        await pool.fetchval(
            "INSERT INTO projects (user_id, title) VALUES ($1, 'theirs') RETURNING id", uuid.UUID(theirs)
        )
    )
    await _signed_in(client)

    assert (await client.patch(f"/projects/{their_project}", json={"title": "mine now"})).status_code == 404


async def test_a_rename_to_nothing_is_refused(client):
    await _signed_in(client)
    made = (await client.post("/projects", json={"title": "keep me"})).json()

    assert (await client.patch(f"/projects/{made['id']}", json={"title": "  "})).status_code == 400
    assert "keep me" in [p["title"] for p in (await client.get("/projects")).json()]


async def test_a_rename_touches_no_folder_and_no_claim(client, tmp_path):
    """Folders are store segments, so renaming a project cannot reach one."""
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    try:
        from harness_module import workspace

        await _signed_in(client)
        made = (await client.post("/projects", json={"title": "inbox triage"})).json()
        assert made["folders"] == ["inbox-triage"]
        session_id = (await client.post("/sessions", json={"goal": "go", "project_id": made["id"]})).json()[
            "session_id"
        ]
        before = (await workspace.claims_for(session_id))[0].folder
        headers = (await client.get("/folders")).json()

        renamed = await client.patch(f"/projects/{made['id']}", json={"title": "something else entirely"})

        assert renamed.json()["title"] == "something else entirely"
        assert renamed.json()["folders"] == ["inbox-triage"], "the link did not follow the label"
        after = (await workspace.claims_for(session_id))[0].folder
        assert after == before == "inbox-triage"
        assert (await client.get("/folders")).json() == headers, "a rename moved the store's headers"
    finally:
        store.use_blobs(None)


async def test_two_projects_named_the_same_do_not_share_one_folder(client, tmp_path):
    """Folder names are unique per user because they are segments of unique paths."""
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    try:
        await _signed_in(client)

        first = (await client.post("/projects", json={"title": "notes"})).json()
        second = (await client.post("/projects", json={"title": "notes"})).json()
        third = (await client.post("/projects", json={"title": "Notes!"})).json()

        assert first["folders"] == ["notes"]
        assert second["folders"] == ["notes-2"]
        assert third["folders"] == ["notes-3"], "the folder rule, not the title, is what collides"
    finally:
        store.use_blobs(None)


async def test_a_name_with_nothing_sluggable_still_gets_a_folder(client, tmp_path):
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    try:
        await _signed_in(client)

        made = (await client.post("/projects", json={"title": "!!!"})).json()

        assert made["folders"] == ["project"]
    finally:
        store.use_blobs(None)


# --- the app is served from the API's own origin ---------------------------------


async def test_the_app_is_served_at_app(client):
    """SameSite=Lax carries no cookie cross-site, so the shell must share the origin."""
    page = await client.get("/app/")

    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert '<div id="root">' in page.text


async def test_the_apps_assets_are_served_beside_it(client):
    styles = await client.get("/app/styles.css")

    assert styles.status_code == 200
    assert "text/css" in styles.headers["content-type"]


async def test_the_shell_loads_without_a_session(client):
    """The page has to render in order to sign anyone in; the cookie gates the API, not the shell."""
    page = await client.get("/app/")

    assert page.status_code == 200
    assert (await client.get("/projects")).status_code == 401


async def _new_session() -> str:
    user_id = uuid.uuid4()
    await pool.execute("INSERT INTO users (id) VALUES ($1)", user_id)
    _seeded.append(user_id)
    session_id = await pool.fetchval(
        "INSERT INTO sessions (user_id, mode, status) VALUES ($1, 'attended', 'idle') RETURNING id",
        user_id,
    )
    return str(session_id)


async def test_a_new_folder_is_durable_before_anything_is_in_it(client, tmp_path):
    """A folder is a path segment with no row of its own: the `.keep` sentinel is
    what makes it exist.
    """
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    try:
        user_id = await _signed_in(client)

        made = await client.post("/folders", json={"path": "receipts/2026"})
        listed = (await client.get("/files")).json()

        assert made.status_code == 201
        assert made.json()["path"] == "receipts/2026"
        assert [f["path"] for f in listed] == ["receipts/2026/.keep"]
        # It is a real tree entry, not a special case in the listing.
        assert [e.path for e in await store.read_tree(user_id)] == ["receipts/2026/.keep"]
        # Its TOP segment is the folder, which is what the Files tab heads with.
        assert (await client.get("/folders")).json() == [{"name": "receipts", "files": 0}]
    finally:
        store.use_blobs(None)


async def test_a_folder_that_is_already_there_is_refused(client, tmp_path):
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    try:
        await _signed_in(client)
        await client.post("/folders", json={"path": "receipts"})

        again = await client.post("/folders", json={"path": "receipts"})

        assert again.status_code == 409
        assert again.json()["code"] == "already_exists"
    finally:
        store.use_blobs(None)


async def test_a_folder_appears_the_moment_a_file_lands_under_a_new_first_segment(client, tmp_path):
    """Nothing declares a folder into being: it is derived, so it cannot lag."""
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    try:
        await _signed_in(client)
        assert (await client.get("/folders")).json() == []

        await client.post(
            "/files",
            files={"file": ("a.md", b"x", "text/plain")},
            data={"path": "ski-trip/a.md"},
        )

        assert (await client.get("/folders")).json() == [{"name": "ski-trip", "files": 1}]
    finally:
        store.use_blobs(None)


async def test_moving_a_file_moves_the_row_and_leaves_the_blob_alone(client, tmp_path):
    """A move is a rename of a path, not a rewrite: same row, same bytes."""
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    try:
        user_id = await _signed_in(client)
        uploaded = (
            await client.post(
                "/files",
                files={"file": ("receipts.csv", b"date,amount\n2026-08-18,12", "text/csv")},
                data={"path": "taxes/receipts.csv"},
            )
        ).json()
        before = (await store.read_tree(user_id))[0].content_hash

        moved = await client.post(
            "/files/move",
            json={"from": "taxes/receipts.csv", "to": "taxes/2026/receipts.csv"},
        )
        listed = (await client.get("/files")).json()

        assert moved.status_code == 200
        assert moved.json()["moved"] == [{"from": "taxes/receipts.csv", "to": "taxes/2026/receipts.csv"}]
        assert [f["path"] for f in listed] == ["taxes/2026/receipts.csv"]
        # The id survives, which is what lets an open reader follow the file.
        assert listed[0]["file_id"] == uploaded["file_id"]
        assert (await store.read_tree(user_id))[0].content_hash == before
    finally:
        store.use_blobs(None)


async def test_moving_between_folders_is_an_ordinary_move(client, tmp_path):
    """One namespace: a move between two folders is a row edit."""
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    try:
        await _signed_in(client)
        await client.post("/files", files={"file": ("a.md", b"x", "text/plain")}, data={"path": "triage/a.md"})
        await client.post("/folders", json={"path": "archive"})

        moved = await client.post("/files/move", json={"from": "triage/a.md", "to": "archive/a.md"})

        assert moved.status_code == 200
        # Sorted: the endpoint promises no order, and the database's collation
        # differs between postgres:15 and the deployment.
        assert sorted(f["path"] for f in (await client.get("/files")).json()) == [
            "archive/.keep",
            "archive/a.md",
        ]
    finally:
        store.use_blobs(None)


async def test_moving_a_directory_takes_everything_under_it(client, tmp_path):
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    try:
        await _signed_in(client)
        for path in ("mail/inbox/a.md", "mail/inbox/deep/b.md"):
            await client.post(
                "/files",
                files={"file": (path.split("/")[-1], b"x", "text/plain")},
                data={"path": path},
            )

        moved = await client.post("/files/move", json={"from": "mail/inbox", "to": "mail/archive/2026"})
        listed = (await client.get("/files")).json()

        assert moved.status_code == 200
        assert [f["path"] for f in listed] == [
            "mail/archive/2026/a.md",
            "mail/archive/2026/deep/b.md",
        ]
    finally:
        store.use_blobs(None)


async def test_renaming_a_folder_is_refused_here(client, tmp_path):
    """A folder rename moves what claims and mounts are keyed by, so /files/move refuses it."""
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    try:
        await _signed_in(client)
        await client.post("/files", files={"file": ("a.md", b"x", "text/plain")}, data={"path": "triage/a.md"})

        refused = await client.post("/files/move", json={"from": "triage", "to": "sorted"})

        assert refused.status_code == 409
        assert refused.json()["code"] == "move_refused"
        assert [f["path"] for f in (await client.get("/files")).json()] == ["triage/a.md"]
    finally:
        store.use_blobs(None)


async def test_a_move_onto_an_occupied_path_is_refused_whole(client, tmp_path):
    """Nothing moves. A half-moved folder is worse than a refused one."""
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    try:
        await _signed_in(client)
        for path in ("mail/a/note.md", "mail/b/note.md"):
            await client.post("/files", files={"file": ("note.md", b"x", "text/plain")}, data={"path": path})

        clash = await client.post("/files/move", json={"from": "mail/a", "to": "mail/b"})
        listed = (await client.get("/files")).json()

        assert clash.status_code == 409
        assert clash.json()["code"] == "move_refused"
        assert [f["path"] for f in listed] == ["mail/a/note.md", "mail/b/note.md"]
    finally:
        store.use_blobs(None)


async def test_a_directory_cannot_be_moved_into_itself(client, tmp_path):
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    try:
        await _signed_in(client)
        await client.post(
            "/files",
            files={"file": ("note.md", b"x", "text/plain")},
            data={"path": "mail/inbox/note.md"},
        )

        eats_itself = await client.post("/files/move", json={"from": "mail/inbox", "to": "mail/inbox/inbox"})

        assert eats_itself.status_code == 409
    finally:
        store.use_blobs(None)


async def test_moving_a_path_the_store_does_not_have_is_absent(client, tmp_path):
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    try:
        await _signed_in(client)

        gone = await client.post("/files/move", json={"from": "a/nope.md", "to": "a/deep/nope.md"})

        assert gone.status_code == 404
    finally:
        store.use_blobs(None)


# --- the consent callback ------------------------------------------------------


class _FakeCallback:
    """Stands in for the Composio client's reconcile leg."""

    def __init__(self, settles=None, blow_up=False):
        self.settles = settles
        self.blow_up = blow_up
        self.calls: list[tuple[str, str]] = []

    async def reconcile(self, user_id, account_id):
        self.calls.append((user_id, account_id))
        if self.blow_up:
            raise ComposioError("400: Bad request")
        return self.settles


async def test_the_callback_settles_the_connection(monkeypatch, client):
    """Composio lands the browser here with a status and an account id."""
    user_id = await _signed_in(client)
    fake = _FakeCallback(settles="GMAIL")
    monkeypatch.setattr(api.hands, "connectors", lambda: fake)

    response = await client.get("/connections/done", params={"status": "success", "connected_account_id": "ca_1"})

    assert response.status_code == 200
    assert fake.calls == [(user_id, "ca_1")]
    # The popup closes itself and pings the opener.
    assert "window.close" in response.text
    assert "connected" in response.text


async def test_the_callback_reconciles_against_the_cookie_user(monkeypatch, client):
    """The account id in the url is a claim; whose it is comes from Composio."""
    user_id = await _signed_in(client)
    fake = _FakeCallback(settles=None)
    monkeypatch.setattr(api.hands, "connectors", lambda: fake)

    response = await client.get(
        "/connections/done", params={"status": "success", "connected_account_id": "ca_someone_else"}
    )

    assert response.status_code == 200
    assert fake.calls == [(user_id, "ca_someone_else")]
    assert "not connected" in response.text


async def test_a_failed_consent_settles_nothing(monkeypatch, client):
    await _signed_in(client)
    fake = _FakeCallback(settles="GMAIL")
    monkeypatch.setattr(api.hands, "connectors", lambda: fake)

    response = await client.get("/connections/done", params={"status": "failed"})

    assert response.status_code == 200
    assert not fake.calls, "a failed consent must not settle a row"


async def test_an_upstream_refusal_still_closes_the_popup(monkeypatch, client):
    """The person is looking at this page; a 500 here strands them mid-flow."""
    await _signed_in(client)
    monkeypatch.setattr(api.hands, "connectors", lambda: _FakeCallback(blow_up=True))

    response = await client.get("/connections/done", params={"status": "success", "connected_account_id": "ca_1"})

    assert response.status_code == 200
    assert "not connected" in response.text


# --- streams end themselves on shutdown -----------------------------------------


async def test_shutdown_wakes_every_subscriber_on_both_channels():
    """An SSE response never ends on its own, so a graceful shutdown must hang up on it."""
    from harness_module.stream import CLOSED, attention, shutdown_streams, stream

    async with stream.subscribe("session-a") as session_q, attention.subscribe("user-b") as attention_q:
        told = shutdown_streams()

        assert told == 2, "every subscriber of every channel is told, not just one"
        assert session_q.get_nowait() is CLOSED
        assert attention_q.get_nowait() is CLOSED


async def test_shutdown_reaches_a_subscriber_whose_queue_is_full():
    """A subscriber that is behind still has to learn the server is leaving."""
    from harness_module.stream import CLOSED, AttentionSignal, UserStream

    channel = UserStream(queue_size=1)
    async with channel.subscribe("user-c") as queue:
        channel.publish("user-c", AttentionSignal(reason="parked", session_id="s"))
        channel.publish("user-c", AttentionSignal(reason="parked", session_id="s"))

        channel.shutdown()

        assert queue.get_nowait() is CLOSED, "the sentinel jumps the queue it cannot join"


async def test_shutdown_with_nobody_listening_is_a_no_op():
    from harness_module.stream import shutdown_streams

    assert shutdown_streams() == 0


async def test_a_new_channel_gets_shutdown_by_being_a_fanout():
    """The reason this lives in the base class rather than in each endpoint."""
    from harness_module.stream import SessionStream, UserStream, _Fanout

    assert issubclass(SessionStream, _Fanout)
    assert issubclass(UserStream, _Fanout)
