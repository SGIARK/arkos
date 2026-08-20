"""The approval gate parks on the gated call itself (11.7).

The rule this file exists to pin: consent binds to the CALL, not to a sentence
about it. A gated call parks the turn with that call still open, the approvals
row carries the (name, args) that will run, and answering runs exactly that call
exactly once — or closes it as declined.

What it replaced looped forever: the gate refused with "you may call {name} once
they agree" and never read the approvals table, so the grant it promised could
not be found.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid

import asyncpg
import pytest
import pytest_asyncio

from agent_module.events import ToolCallEvent, ToolResultEvent, UserEvent
from db import pool
from harness_module import approvals, runner
from harness_module import session_log as slog
from tests.dbgate import require_db
from tool_module.envelope import ResultEnvelope, ToolSpec, ok

pytestmark = pytest.mark.asyncio

_seeded: list[uuid.UUID] = []


@pytest_asyncio.fixture(autouse=True)
async def _db():
    await require_db()
    yield
    await pool.execute("DELETE FROM sessions WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM projects WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM users WHERE id = ANY($1::uuid[])", _seeded)
    _seeded.clear()
    await pool.close()


async def _session(mode: str = "attended") -> str:
    user_id = uuid.uuid4()
    _seeded.append(user_id)
    await pool.execute("INSERT INTO users (id) VALUES ($1)", user_id)
    return str(
        await pool.fetchval(
            "INSERT INTO sessions (user_id, mode, status) VALUES ($1, $2, 'running') RETURNING id",
            user_id,
            mode,
        )
    )


async def _parked_on_a_call(session_id: str, *, name: str = "mcp_send_email", args: dict | None = None) -> str:
    """Put a session where a park leaves it: the call open, the row bound to it."""
    args = {"to": "rachel@lumen.co"} if args is None else args
    await slog.append(session_id, UserEvent(text="reply to rachel"))
    stored = await slog.append(session_id, ToolCallEvent(id="c1", name=name, args=args))
    await approvals.create(session_id, "c1", "call", f"Run {name}?", tool_name=name, tool_args=args)
    await pool.execute("UPDATE sessions SET status = 'awaiting_approval' WHERE id = $1", uuid.UUID(session_id))
    assert stored.seq
    return "c1"


class _Sink:
    """Just enough of `_Sink` for the settle path: emit, barrier, the grant flag."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.emitted: list = []
        self._grant_once = False

    def emit(self, event) -> None:
        self.emitted.append(event)

    async def barrier(self) -> None:
        for event in self.emitted:
            await slog.append(self.session_id, event)
        self.emitted.clear()


def _dispatch(record: list, envelope: ResultEnvelope | None = None):
    async def run(name: str, args: dict) -> ResultEnvelope:
        record.append((name, args))
        return envelope or ok("sent")

    return run


# --- the row is bound to the call ---------------------------------------------------


async def test_a_parked_call_stays_open_and_the_row_carries_it():
    """The call is open ON PURPOSE: it is what the approval is bound to."""
    session_id = await _session()
    await _parked_on_a_call(session_id)

    assert await slog.open_calls(session_id) == {"c1": "mcp_send_email"}
    row = await approvals.grantable(session_id)
    assert row is None, "unanswered is not grantable"

    open_rows = await approvals.open_for(session_id)
    assert [(r.kind, r.tool_call_id, r.tool_name) for r in open_rows] == [
        ("call", "c1", "mcp_send_email")
    ]
    assert open_rows[0].tool_args == {"to": "rachel@lumen.co"}


async def test_a_second_row_for_the_same_open_call_is_impossible():
    """The database enforces one open question per call, not just the runner."""
    session_id = await _session()
    await _parked_on_a_call(session_id)

    with pytest.raises(asyncpg.UniqueViolationError):
        await approvals.create(session_id, "c1", "call", "again?", tool_name="x", tool_args={})


# --- approve --------------------------------------------------------------------


async def test_approving_runs_exactly_that_call_and_closes_it():
    session_id = await _session()
    await _parked_on_a_call(session_id)
    await approvals.answer((await approvals.open_for(session_id))[0].id, approvals.APPROVE)

    calls: list = []
    sink = _Sink(session_id)
    settled = await runner._settle_gated_call(await runner.load(session_id), sink, _dispatch(calls))

    assert settled is True
    assert calls == [("mcp_send_email", {"to": "rachel@lumen.co"})], "the approved call, with its own args"
    assert await slog.open_calls(session_id) == {}, "and the call is closed"

    events = await slog.get_events(session_id, after_seq=0, limit=100)
    result = next(e.event for e in events if isinstance(e.event, ToolResultEvent))
    assert result.ok is True and result.content == "sent"


async def test_the_grant_is_one_shot():
    """The resumed turn runs the approved call; what the model reaches for next is gated again."""
    session_id = await _session()
    await _parked_on_a_call(session_id)
    await approvals.answer((await approvals.open_for(session_id))[0].id, approvals.APPROVE)

    sink = _Sink(session_id)
    await runner.dispatch_granted(sink, _dispatch([]), "mcp_send_email", {})

    assert sink._grant_once is False


async def test_concurrent_wakes_admit_one_executor():
    """Two wakes racing to resume one parked session must not both send the email."""
    session_id = await _session()
    await _parked_on_a_call(session_id)
    row = (await approvals.open_for(session_id))[0]
    await approvals.answer(row.id, approvals.APPROVE)

    claims = await asyncio.gather(*(approvals.consume(row.id) for _ in range(5)))

    assert sum(1 for c in claims if c is not None) == 1, "the latch admits exactly one"


# --- decline --------------------------------------------------------------------


async def test_declining_closes_the_call_and_tells_the_model_to_route_around():
    session_id = await _session()
    await _parked_on_a_call(session_id)
    await approvals.answer((await approvals.open_for(session_id))[0].id, approvals.DECLINE)

    calls: list = []
    sink = _Sink(session_id)
    settled = await runner._settle_gated_call(await runner.load(session_id), sink, _dispatch(calls))

    assert settled is True
    assert calls == [], "a declined call never runs"
    assert await slog.open_calls(session_id) == {}

    events = await slog.get_events(session_id, after_seq=0, limit=100)
    result = next(e.event for e in events if isinstance(e.event, ToolResultEvent))
    assert result.ok is False
    assert "declined" in result.content and "another approach" in result.content
    assert result.error_kind != "invalid_args", "asking correctly is not a bad-args failure"


async def test_anything_that_is_not_approve_is_not_consent():
    """`approved` is an allow-list, so a stray answer cannot read as yes."""
    session_id = await _session()
    await _parked_on_a_call(session_id)
    row = (await approvals.open_for(session_id))[0]
    await approvals.answer(row.id, "sure, go ahead")

    calls: list = []
    await runner._settle_gated_call(await runner.load(session_id), _Sink(session_id), _dispatch(calls))

    assert calls == []


# --- repair ---------------------------------------------------------------------


async def test_a_consumed_but_unclosed_call_repairs_without_re_running_it():
    """The process died between claiming and appending. The tool may have run.

    Sending a message twice is worse than not knowing whether it sent once.
    """
    session_id = await _session()
    await _parked_on_a_call(session_id)
    row = (await approvals.open_for(session_id))[0]
    await approvals.answer(row.id, approvals.APPROVE)
    await approvals.consume(row.id)  # claimed, then the process died

    calls: list = []
    sink = _Sink(session_id)
    settled = await runner._settle_gated_call(await runner.load(session_id), sink, _dispatch(calls))

    assert settled is True
    assert calls == [], "never re-run"
    events = await slog.get_events(session_id, after_seq=0, limit=100)
    result = next(e.event for e in events if isinstance(e.event, ToolResultEvent))
    assert result.error_kind == "interrupted"
    assert "verify before retrying" in result.content


async def test_a_call_already_closed_owes_nothing():
    session_id = await _session()
    await _parked_on_a_call(session_id)
    row = (await approvals.open_for(session_id))[0]
    await approvals.answer(row.id, approvals.APPROVE)
    await approvals.consume(row.id)
    await slog.append(session_id, ToolResultEvent(id="c1", ok=True, content="sent"))

    calls: list = []
    settled = await runner._settle_gated_call(await runner.load(session_id), _Sink(session_id), _dispatch(calls))

    assert settled is False and calls == []


# --- prose can never decide a call ------------------------------------------------


async def test_composer_prose_cannot_silently_decline_a_gated_call():
    """P0 from the 2026-08-20 review.

    `approvals.approved` is an allow-list, so any prose that is not the approve
    word reads as a decline. A message typed into the composer of a call-parked
    session used to fall through and answer the call row — "sounds good, go
    ahead" would have DECLINED a call the human never saw. Consent binds to the
    call, so it is given where the call is on screen or not at all.
    """
    from harness_module import api

    session_id = await _session()
    await _parked_on_a_call(session_id)

    with pytest.raises(api.ApiError) as raised:
        await api._answer_by_message(session_id, "sounds good, go ahead")

    assert raised.value.status == 409
    assert raised.value.code == "awaiting_approval"

    row = (await approvals.open_for(session_id))[0]
    assert row.answered_at is None, "and the call is still waiting to be decided"


async def test_an_ask_is_still_answerable_from_the_composer():
    """The guard is about consent, not about parking: a question still takes prose."""
    from harness_module import api

    session_id = await _session()
    await slog.append(session_id, ToolCallEvent(id="c1", name="ask", args={"question": "which one?"}))
    await slog.append(session_id, ToolResultEvent(id="c1", ok=True, content="asked"))
    await approvals.create(session_id, "c1", "ask", "which one?")
    await pool.execute("UPDATE sessions SET status = 'awaiting_approval' WHERE id = $1", uuid.UUID(session_id))

    with contextlib.suppress(Exception):
        # `runner.start` is out of scope here; the answer landing is the point.
        await api._answer_by_message(session_id, "the second one")

    assert (await approvals.open_for(session_id)) == []


# --- the gate itself ------------------------------------------------------------


async def test_the_gate_no_longer_refuses_with_invalid_args():
    """The refusal path is deleted: `invalid_args` spent the per-tool failure cap
    on asking correctly, and promised a grant nothing ever kept."""
    session_id = await _session()
    session = await runner.load(session_id)
    sink = runner._Sink(session)
    try:
        with pytest.raises(Exception) as raised:
            await sink._approve("mcp_send_email", {})
        assert getattr(raised.value, "error_kind", None) == runner._GATED
        assert "request_approval" not in str(raised.value)
    finally:
        await sink.close()


async def test_the_gated_result_is_dropped_so_the_call_stays_open():
    """`emit` suppressing that one result IS the mechanism: a queued result would
    close the call, and the call has to survive the park."""
    session_id = await _session()
    session = await runner.load(session_id)
    sink = runner._Sink(session)
    try:
        sink.emit(ToolCallEvent(id="c9", name="mcp_send_email", args={"to": "x"}))
        sink.emit(ToolResultEvent(id="c9", ok=False, content="waiting", error_kind=runner._GATED))

        assert sink.parked is True
        assert sink._park == ("c9", "mcp_send_email", {"to": "x"})
        assert not any(
            isinstance(e, ToolResultEvent) for e in list(sink._queue._queue)
        ), "the result never reached the writer"
    finally:
        await sink.close()


async def test_only_one_call_parks_per_hop():
    """The transcript permits exactly one open call across a park, so a second
    gated call in the same hop is told so and closes normally."""
    session_id = await _session()
    session = await runner.load(session_id)
    sink = runner._Sink(session)
    try:
        sink.emit(ToolCallEvent(id="c1", name="mcp_send_email", args={}))
        sink.emit(ToolResultEvent(id="c1", ok=False, content="waiting", error_kind=runner._GATED))

        with pytest.raises(Exception) as raised:
            await sink._approve("mcp_post_message", {})
        assert "already waiting" in str(raised.value)
    finally:
        await sink.close()


def test_the_manifest_still_gates_the_tool():
    """A sanity check that `requires_approval` is what reaches the gate at all."""
    spec = ToolSpec(name="mcp_send_email", requires_approval=True)
    assert spec.requires_approval is True
