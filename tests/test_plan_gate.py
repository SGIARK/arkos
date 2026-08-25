"""An unattended run starts from an approved plan, and never dies confused (11.8.5).

The rule this file exists to pin: there is exactly ONE door into an unattended
run. `propose_plan` parks the session on a row of kind `plan` carrying the plan
itself, and approving that row — not pressing play, not the model deciding — is
what writes `plan.md` and flips the mode.

What it replaced: the play arrow handed the model a transcript rather than a
task. The 2026-08-20 Marketplace run went unattended with the model's own
unanswered question as the last event, burned a browser run and five bare-text
hops greeting nobody, and ended `failed{model_error}` though nothing errored.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from agent_module.events import ToolCallEvent
from db import pool
from harness_module import api, approvals, lifecycle, runner, store
from harness_module import session_log as slog
from tests.dbgate import require_db
from tests.test_api import _supabase_token

_seeded: list[uuid.UUID] = []

PLAN = {
    "goal": "clear the weekend backlog in triage/",
    "done_when": "every thread is answered or drafted, and nothing has been sent",
    "steps": ["read the unread threads", "draft the two that need a reply"],
    "inputs": [{"label": "triage/", "note": "this project's directory"}],
    "missing": ["is there a deadline i should work against?"],
}


@pytest_asyncio.fixture(autouse=True)
async def _db(monkeypatch, tmp_path):
    await require_db()

    # Blobs on disk, as everywhere else that writes a file in a unit test.
    # `SupabaseBlobs` caches one httpx client, and pytest-asyncio gives every
    # test its own event loop, so the second test in a module to write a blob
    # gets "Event loop is closed" from a pool belonging to the first one's loop.
    # Approving a plan writes `plan.md`, so more than one test here does.
    store.use_blobs(store.FilesystemBlobs(tmp_path))

    # The loop is out of scope: the turn a wake would drive is not what these
    # tests are about, and the kwargs `start` was called with are.
    start_calls: list[dict] = []

    async def fake_start(session_id: str, **kw) -> bool:
        start_calls.append({"session_id": session_id, **kw})
        return True

    monkeypatch.setattr(runner, "start", fake_start)
    api.start_calls = start_calls
    yield
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
    response = await client.post(
        "/auth/session", headers={"Authorization": f"Bearer {_supabase_token(user_id)}"}
    )
    assert response.status_code == 204
    return user_id


async def _project_session(user_id: str, *, status: str = "awaiting_approval") -> tuple[str, str]:
    """A project linking one folder, and a session in it claiming that folder.

    The claim is what decides where `plan.md` lands (11.9): the FIRST folder the
    session writes, which is the first the project linked.
    """
    project_id = await api._new_project(user_id, "inbox triage")
    await api._link_folder(project_id, await api._make_folder(user_id, "inbox-triage"))
    session_id = await pool.fetchval(
        "INSERT INTO sessions (user_id, project_id, mode, status, title) "
        "VALUES ($1, $2, 'attended', $3, 'a session') RETURNING id",
        uuid.UUID(user_id),
        project_id,
        status,
    )
    await api._record_claims(str(session_id), project_id, None, user_id)
    return str(project_id), str(session_id)


async def _propose(session_id: str, args: dict, *, call_id: str = "c1") -> approvals.Approval:
    """Put a session where the park leaves it: the call closed, the row carrying the plan."""
    await slog.append(session_id, ToolCallEvent(id=call_id, name="propose_plan", args=args))
    await approvals.supersede_plans(session_id)
    row = await approvals.create(
        session_id, call_id, "plan", args["goal"], tool_name="propose_plan", tool_args=args
    )
    await pool.execute(
        "UPDATE sessions SET status = 'awaiting_approval' WHERE id = $1", uuid.UUID(session_id)
    )
    return row


# --- the tool parks on the plan -------------------------------------------------


@pytest.mark.asyncio
async def test_propose_plan_is_a_park_tool_of_kind_plan():
    from tool_module.tools.control import PARK_KINDS, ProposePlan

    assert PARK_KINDS[ProposePlan.spec.name] == "plan"
    # An under-informed plan is still a plan: `missing` is how insufficiency
    # renders, so nothing about it is required.
    assert set(ProposePlan.spec.input_schema["required"]) == {"goal", "done_when", "steps"}


@pytest.mark.asyncio
async def test_a_plan_park_carries_the_args_on_the_row(client):
    """Consent binds to the plan, not to prose about it — the `call` rule, applied."""
    user_id = await _user(client)
    _, session_id = await _project_session(user_id)

    row = await _propose(session_id, PLAN)

    assert row.kind == "plan" and row.is_plan
    assert row.tool_name == "propose_plan"
    assert row.tool_args == PLAN
    assert row.prompt == PLAN["goal"]
    assert [r.id for r in await approvals.open_for(session_id)] == [row.id]


@pytest.mark.asyncio
async def test_the_park_prompt_is_the_goal():
    assert runner._park_prompt("propose_plan", PLAN) == PLAN["goal"]
    assert runner._park_prompt("propose_plan", {}) == "(no goal given)"


# --- answering it ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_approving_saves_the_plan_and_starts_the_run_unattended(client):
    user_id = await _user(client)
    _, session_id = await _project_session(user_id)
    row = await _propose(session_id, PLAN)

    response = await client.post(f"/approvals/{row.id}/respond", json={"answer": "approve"})

    assert response.status_code == 202
    assert response.json()["mode"] == "unattended"
    # Mode and status move in the SAME conditional UPDATE, inside `start`.
    assert api.start_calls[-1]["mode"] == "unattended"
    assert api.start_calls[-1]["reason"] == "plan_approved"

    saved = await pool.fetchrow(
        "SELECT path, content_hash FROM files WHERE user_id = $1 AND path = 'inbox-triage/plan.md'",
        uuid.UUID(user_id),
    )
    assert saved is not None, "the plan did not land in the session's first linked folder"
    text = (await store.get_blob(saved["content_hash"])).decode()
    assert PLAN["goal"] in text
    assert PLAN["done_when"] in text
    for step in PLAN["steps"]:
        assert step in text
    assert PLAN["missing"][0] in text, "an open question is part of the plan that was approved"


@pytest.mark.asyncio
async def test_the_unattended_quota_binds_at_plan_approval(client, monkeypatch):
    """The one point a user's unattended load grows, so the one point it is counted.

    Checked BEFORE the row is answered: a user at their limit keeps their plan
    rather than losing it to an approval that then cannot start anything.
    """
    user_id = await _user(client)
    monkeypatch.setattr(
        api, "_cfg", lambda key, default: 1 if key == "quotas.max_unattended_sessions" else default
    )
    await pool.execute(
        "INSERT INTO sessions (user_id, mode, status) VALUES ($1, 'unattended', 'running')",
        uuid.UUID(user_id),
    )
    _, session_id = await _project_session(user_id)
    row = await _propose(session_id, PLAN)

    response = await client.post(f"/approvals/{row.id}/respond", json={"answer": "approve"})

    assert response.status_code == 429
    assert response.json()["code"] == "quota_exceeded"
    still_open = await approvals.open_for(session_id)
    assert [r.id for r in still_open] == [row.id], "the refusal cost them nothing"


@pytest.mark.asyncio
async def test_declining_closes_the_park_and_leaves_an_attended_chat(client):
    user_id = await _user(client)
    _, session_id = await _project_session(user_id)
    row = await _propose(session_id, PLAN)

    response = await client.post(f"/approvals/{row.id}/respond", json={"answer": "decline"})

    assert response.status_code == 202
    assert response.json()["mode"] == "attended"
    assert api.start_calls == [], "nothing ran"
    assert await approvals.open_for(session_id) == []
    session = await pool.fetchrow(
        "SELECT status, mode FROM sessions WHERE id = $1", uuid.UUID(session_id)
    )
    assert (session["status"], session["mode"]) == ("idle", "attended")
    assert await pool.fetchval(
        "SELECT count(*) FROM files WHERE user_id = $1 AND path = 'inbox-triage/plan.md'",
        uuid.UUID(user_id),
    ) == 0


@pytest.mark.asyncio
async def test_a_reply_closes_the_plan_and_asks_for_the_next_one(client):
    """Anything that is not one of the two words is a reply.

    Two events land, and the second is the point: the human's own words, then
    the instruction that makes them produce a PLAN. Without it the model answers
    inline and the session goes idle with the card gone and nothing to approve —
    the run they were setting up quietly stops existing.
    """
    user_id = await _user(client)
    _, session_id = await _project_session(user_id)
    row = await _propose(session_id, PLAN)

    response = await client.post(
        f"/approvals/{row.id}/respond", json={"answer": "don't send anything, draft it all"}
    )

    assert response.status_code == 202
    assert response.json()["mode"] == "attended"
    assert "mode" not in api.start_calls[-1], "replying is an attended turn"
    assert api.start_calls[-1]["reason"] == "plan_reply"

    events = [e.event for e in await slog.get_events(session_id)]
    assert [(e.kind, e.source) for e in events[-2:]] == [("user", "human"), ("user", "system")]
    assert events[-2].text == "don't send anything, draft it all"
    assert "propose_plan" in events[-1].text

    assert await approvals.open_for(session_id) == [], "the card closed with the reply"
    assert await pool.fetchval(
        "SELECT count(*) FROM files WHERE user_id = $1 AND path = 'inbox-triage/plan.md'",
        uuid.UUID(user_id),
    ) == 0


@pytest.mark.asyncio
async def test_a_composer_message_to_a_plan_parked_session_is_refused(client):
    """`plan` joins `approval` and `call` in the 409: prose never answers a plan.

    "yes do that" typed in the composer would read as FEEDBACK, spending a round
    of the workshop on an approval the human thought they had given.
    """
    user_id = await _user(client)
    _, session_id = await _project_session(user_id)
    await _propose(session_id, PLAN)

    response = await client.post(f"/sessions/{session_id}/messages", json={"text": "yes do that"})

    assert response.status_code == 409
    assert response.json()["code"] == "awaiting_approval"
    assert "a plan" in response.json()["message"]
    assert api.start_calls == []


# --- versions -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_second_proposal_supersedes_the_first_and_bumps_the_version(client):
    user_id = await _user(client)
    _, session_id = await _project_session(user_id)
    first = await _propose(session_id, PLAN, call_id="c1")

    revised = {**PLAN, "goal": "clear the backlog, drafting everything", "missing": []}
    second = await _propose(session_id, revised, call_id="c2")

    assert [r.id for r in await approvals.open_for(session_id)] == [second.id], "one live plan"
    closed = await approvals.get(first.id, user_id)
    assert closed.answer == approvals.SUPERSEDED
    assert closed.approved is False, "superseding is not consent"
    history = await approvals.plan_history(session_id)
    assert [r.id for r in history] == [first.id, second.id]


@pytest.mark.asyncio
async def test_attention_carries_the_plan_and_its_version_and_nothing_else(client):
    """No diff against the previous version.

    Edits stack: by v3 the "changed since v{n-1}" list was longer than the plan
    and said less than reading the plan does. A reply is answered by a whole new
    plan, so the whole new plan is all that is sent.
    """
    user_id = await _user(client)
    _, session_id = await _project_session(user_id)
    first = await _propose(session_id, PLAN, call_id="c1")
    await approvals.answer(first.id, "don't send anything")
    revised = {**PLAN, "goal": "clear the backlog, drafting everything", "missing": []}
    await _propose(session_id, revised, call_id="c2")

    rows = (await client.get(f"/attention?session_id={session_id}")).json()

    assert len(rows) == 1
    assert rows[0]["kind"] == "plan"
    assert rows[0]["tool_args"] == revised
    assert rows[0]["version"] == 2
    assert "previous_args" not in rows[0] and "last_ask" not in rows[0]


@pytest.mark.asyncio
async def test_a_decision_is_stored_normalized_however_it_was_typed(client):
    """The consent table must say what was decided, not how it was spelled.

    `{"answer": "Approve"}` started the run and wrote plan.md while the row read
    unapproved, so the one table whose whole job is binding consent disagreed
    with what had happened.
    """
    user_id = await _user(client)
    _, session_id = await _project_session(user_id)
    row = await _propose(session_id, PLAN)

    assert (await client.post(f"/approvals/{row.id}/respond", json={"answer": "  Approve "})).status_code == 202

    settled = await approvals.get(row.id, user_id)
    assert settled.answer == approvals.APPROVE
    assert settled.approved is True


@pytest.mark.asyncio
async def test_a_plan_the_run_could_not_start_is_reopened(client, monkeypatch):
    """A lost status race must not consume the plan.

    Answered-and-never-started leaves a row nothing can approve again, and the
    human's only recourse is getting the whole plan proposed from scratch.
    """
    user_id = await _user(client)
    _, session_id = await _project_session(user_id)
    row = await _propose(session_id, PLAN)

    async def lost(session_id: str, **kw) -> bool:
        return False

    monkeypatch.setattr(runner, "start", lost)
    response = await client.post(f"/approvals/{row.id}/respond", json={"answer": "approve"})

    assert response.status_code == 409
    assert response.json()["code"] == "not_idle"
    assert [r.id for r in await approvals.open_for(session_id)] == [row.id], "the card is back"


@pytest.mark.asyncio
async def test_a_session_given_no_folder_cannot_approve_a_plan(client):
    """The unattended prompt promises `plan.md` at the root of the first folder.

    The home chat is exactly this session: no project, no links, nothing durable
    to write into. Refused BEFORE the row is answered, so the plan survives.
    """
    user_id = await _user(client)
    session_id = str(
        await pool.fetchval(
            "INSERT INTO sessions (user_id, mode, status) VALUES ($1, 'attended', 'awaiting_approval') "
            "RETURNING id",
            uuid.UUID(user_id),
        )
    )
    row = await _propose(session_id, PLAN)

    response = await client.post(f"/approvals/{row.id}/respond", json={"answer": "approve"})

    assert response.status_code == 409
    assert response.json()["code"] == "no_folder"
    assert [r.id for r in await approvals.open_for(session_id)] == [row.id]
    assert api.start_calls == []


@pytest.mark.asyncio
async def test_the_handoff_carries_the_plan_rather_than_sending_the_model_to_read_it(client):
    """A fact the harness knows is injected; the model's tools are for the world.

    "read plan.md FIRST" spent a tool call on something already in hand — and
    after a DECLINED plan it was a guaranteed FileNotFound, because nothing had
    written the file.
    """
    user_id = await _user(client)
    _, session_id = await _project_session(user_id, status="idle")

    await client.post(f"/sessions/{session_id}/approve")
    fresh = [e.event for e in await slog.get_events(session_id)][-1]

    assert fresh.source == "system"
    assert "No plan exists for this session yet" in fresh.text
    assert "read plan.md" not in fresh.text.lower()

    # Once a run has happened here, the file's CONTENT rides along.
    await runner.save_plan(session_id, PLAN, 1)
    await pool.execute("UPDATE sessions SET status = 'idle' WHERE id = $1", uuid.UUID(session_id))
    await client.post(f"/sessions/{session_id}/approve")
    again = [e.event for e in await slog.get_events(session_id)][-1]

    assert PLAN["goal"] in again.text, "the plan itself was not injected"
    assert "CONTINUATION" in again.text
    assert "read plan.md" not in again.text.lower()


@pytest.mark.asyncio
async def test_play_on_a_cancelled_run_drafts_a_continuation(client):
    """The header's ▶ reads "resume" on a spent plan, and this is what it calls.

    `approve_session` refused anything that was not idle or pending, so the one
    press the design asks for — pick a cancelled run back up — 409'd. A terminal
    session is a legal starting point; the `terminal -> running` reopen exists
    for exactly this, and the handoff makes it a CONTINUATION rather than a
    fresh v1.

    A run that was cancelled HAS a plan — that is what made it a run — so the
    handoff carries it. A session cancelled before any plan was approved has
    nothing to continue from and is told so instead; that case is
    `test_the_handoff_carries_the_plan_rather_than_sending_the_model_to_read_it`.
    """
    user_id = await _user(client)
    _, session_id = await _project_session(user_id, status="cancelled")
    await pool.execute(
        "UPDATE sessions SET terminal_reason = 'cancelled' WHERE id = $1", uuid.UUID(session_id)
    )
    await runner.save_plan(session_id, PLAN, 1)

    response = await client.post(f"/sessions/{session_id}/approve")

    assert response.status_code == 202
    assert response.json()["mode"] == "attended"
    events = [e.event for e in await slog.get_events(session_id)]
    assert events[-1].source == "system"
    assert "continuation" in events[-1].text.lower()
    assert "plan.md" in events[-1].text


@pytest.mark.asyncio
async def test_play_is_still_refused_while_a_run_is_live(client):
    user_id = await _user(client)
    _, session_id = await _project_session(user_id, status="running")

    response = await client.post(f"/sessions/{session_id}/approve")

    assert response.status_code == 409
    assert response.json()["code"] == "not_idle"


@pytest.mark.asyncio
async def test_the_session_snapshot_carries_the_newest_plan(client):
    """The pinned card reads this, not a count of `propose_plan` calls.

    `recent_events` is a capped window, so counting them renders a version the
    server does not agree with once a session gets long — or loses the line.
    """
    user_id = await _user(client)
    _, session_id = await _project_session(user_id)
    await _propose(session_id, PLAN, call_id="c1")
    second = await _propose(session_id, {**PLAN, "goal": "the revised goal"}, call_id="c2")
    await client.post(f"/approvals/{second.id}/respond", json={"answer": "approve"})

    snapshot = (await client.get(f"/sessions/{session_id}")).json()

    assert snapshot["plan"]["version"] == 2
    assert snapshot["plan"]["goal"] == "the revised goal"
    assert snapshot["plan"]["answer"] == approvals.APPROVE
    assert snapshot["folders"] == ["inbox-triage"], "where the work lands, in claim order"


@pytest.mark.asyncio
async def test_a_new_proposal_is_the_only_live_card(client):
    """Two plans are never on screen at once, whatever prompted the second."""
    user_id = await _user(client)
    _, session_id = await _project_session(user_id)
    await _propose(session_id, PLAN, call_id="c1")
    await _propose(session_id, {**PLAN, "goal": "a tighter goal"}, call_id="c2")

    rows = (await client.get(f"/attention?session_id={session_id}")).json()

    assert len(rows) == 1
    assert rows[0]["version"] == 2
    assert rows[0]["tool_args"]["goal"] == "a tighter goal"


# --- the file the run starts from -----------------------------------------------


def test_plan_markdown_is_a_pure_render_of_the_approved_args():
    text = runner.plan_markdown(PLAN, 3)

    assert text.startswith(f"# {PLAN['goal']}\n")
    assert "_plan v3_" in text
    assert "1. read the unread threads" in text
    assert "- triage/ — this project's directory" in text
    assert "## still open" in text
    # Nothing the human did not read on the card.
    assert "propose_plan" not in text


def test_a_plan_with_only_the_required_fields_still_renders():
    text = runner.plan_markdown({"goal": "g", "done_when": "d", "steps": ["one"]}, 1)

    assert "## inputs" not in text and "## still open" not in text
    assert text.endswith("\n")


# --- the transitions this card moved --------------------------------------------


def test_a_declined_plan_has_a_transition_of_its_own():
    """The only answer that ends a park without waking the session."""
    assert ("awaiting_approval", "idle") in lifecycle.ALLOWED
    assert ("awaiting_approval", "running") in lifecycle.ALLOWED


def test_the_new_reasons_are_failures_with_names():
    from agent_module.events import DoneEvent

    assert lifecycle.status_for(DoneEvent(reason="stalled_progress")) == "failed"
    assert lifecycle.status_for(DoneEvent(reason="internal_error")) == "failed"
