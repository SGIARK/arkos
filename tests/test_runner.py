"""The fold, the drive, and the translation from events to a record.

Runs against a real Postgres with migration 0 applied; the model is mocked.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
import pytest_asyncio

from agent_module.events import ContentEvent, DoneEvent, ToolCallEvent, ToolResultEvent, UserEvent
from db import pool
from harness_module import runner
from harness_module import session_log as slog
from model_module import client as mc
from tool_module.envelope import ToolSpec, ok

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
    runner._running.clear()
    await pool.execute("DELETE FROM sessions WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM users WHERE id = ANY($1::uuid[])", _seeded)
    _seeded.clear()
    await pool.close()


async def _session(mode: str = "attended", status: str = "idle", goal: str = "do the thing") -> str:
    user_id = uuid.uuid4()
    await pool.execute("INSERT INTO users (id) VALUES ($1)", user_id)
    _seeded.append(user_id)
    session_id = await pool.fetchval(
        "INSERT INTO sessions (user_id, mode, status, goal) VALUES ($1, $2, $3, $4) RETURNING id",
        user_id,
        mode,
        status,
        goal,
    )
    return str(session_id)


def _text(*chunks):
    return [mc.TextDelta(text=c) for c in chunks] + [mc.Finish(reason="stop")]


def _call(name, args="{}", *, id="c1"):
    return [
        mc.ToolCallDelta(index=0, id=id, name=name, arguments=args),
        mc.Finish(reason="tool_calls"),
    ]


@pytest.fixture
def model(monkeypatch):
    """Arm the model with one delta list per hop, and record what it was sent."""

    class Recorder:
        def __init__(self):
            self.hops = 0
            self.messages_seen = []
            self._hops = []

        def arm(self, *hops):
            self._hops = list(hops)

            def generate(messages, tools=None, **kw):
                self.hops += 1
                self.messages_seen.append([dict(m) for m in messages])
                deltas = self._hops.pop(0) if self._hops else _text("done")

                async def gen():
                    for d in deltas:
                        await asyncio.sleep(0)
                        yield d

                return gen()

            monkeypatch.setattr(mc, "generate", generate)

    return Recorder()


@pytest.fixture
def tools(monkeypatch):
    """A manifest of two tools, dispatched to a recorder rather than the world."""
    calls: list[tuple[str, dict]] = []

    async def manifest(user_id, mcp=None):
        return [ToolSpec(name="grep", readonly=True), ToolSpec(name="finish_task")]

    def bind(ctx, **kw):
        async def dispatch(name, args):
            calls.append((name, args))
            return ok(f"{name} ran")

        return dispatch

    monkeypatch.setattr(runner.registry, "manifest", manifest)
    monkeypatch.setattr(runner.registry, "bind", bind)
    return calls


# --- the fold ------------------------------------------------------------------


async def test_the_fold_rebuilds_messages_from_the_log():
    session_id = await _session()
    await slog.append(session_id, UserEvent(text="find the receipt"))
    await slog.append(session_id, ContentEvent(text="Looking"))
    await slog.append(session_id, ContentEvent(text=" now."))
    await slog.append(session_id, ToolCallEvent(id="c1", name="grep", args={"q": "receipt"}))
    await slog.append(session_id, ToolResultEvent(id="c1", ok=True, content="found it"))

    messages, hops = await runner.fold(await runner.load(session_id))

    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "find the receipt"}
    # Streamed chunks are one assistant message, carrying that hop's tool calls.
    assert messages[2]["content"] == "Looking now."
    assert messages[2]["tool_calls"][0]["function"]["name"] == "grep"
    assert messages[3] == {"role": "tool", "tool_call_id": "c1", "content": "found it"}
    assert hops == 0


async def test_reasoning_is_never_replayed_into_context():
    from agent_module.events import ReasoningEvent

    session_id = await _session()
    await slog.append(session_id, UserEvent(text="hi"))
    await slog.append(session_id, ReasoningEvent(text="the user said hi, I should..."))
    await slog.append(session_id, ContentEvent(text="hello"))

    messages, _ = await runner.fold(await runner.load(session_id))

    assert "I should" not in str(messages)


async def test_event_replay_deterministic():
    """Same log, byte-identical context — the prompt cache and replay both need it."""
    session_id = await _session()
    await slog.append(session_id, UserEvent(text="hi"))
    await slog.append(session_id, ContentEvent(text="hello"))

    session = await runner.load(session_id)
    first, _ = await runner.fold(session)
    second, _ = await runner.fold(session)

    assert first == second


async def test_hops_reset_at_a_done_and_carry_inside_one_run():
    from agent_module.events import BudgetEvent

    session_id = await _session()
    await slog.append(session_id, BudgetEvent(hops_used=3, hops_max=6))
    _, mid_run = await runner.fold(await runner.load(session_id))

    await slog.append(session_id, DoneEvent(reason="turn_end"))
    _, after_done = await runner.fold(await runner.load(session_id))

    assert mid_run == 3, "a resume inside a run keeps counting"
    assert after_done == 0, "a new turn budgets from zero"


async def test_a_stored_result_carries_its_ref_so_the_tail_is_recoverable():
    session_id = await _session()
    await slog.append(session_id, ToolCallEvent(id="c1", name="grep", args={}))
    await slog.append(
        session_id,
        ToolResultEvent(id="c1", ok=True, content="head", total_chars=9000, ref="b_1"),
    )

    messages, _ = await runner.fold(await runner.load(session_id))

    assert "read_result" in messages[-1]["content"]
    assert "b_1" in messages[-1]["content"]


# --- driving -------------------------------------------------------------------


async def test_an_attended_turn_ends_in_idle(model, tools):
    session_id = await _session(mode="attended")
    await slog.append(session_id, UserEvent(text="hello"))
    model.arm(_text("hi ", "there"))

    await runner.start(session_id)
    await _settle(session_id)

    row = await pool.fetchrow("SELECT status, terminal_reason FROM sessions WHERE id = $1", uuid.UUID(session_id))
    kinds = [e.event.kind for e in await slog.get_events(session_id)]

    assert (row["status"], row["terminal_reason"]) == ("idle", None)
    assert kinds[-2:] == ["done", "lifecycle"]
    assert "content" in kinds


async def test_the_transcript_records_the_whole_turn(model, tools):
    session_id = await _session(mode="attended")
    await slog.append(session_id, UserEvent(text="grep for it"))
    model.arm(_call("grep", '{"q": "x"}'), _text("found it"))

    await runner.start(session_id)
    await _settle(session_id)

    kinds = [e.event.kind for e in await slog.get_events(session_id)]

    assert kinds.count("tool_call") == 1
    assert kinds.count("tool_result") == 1
    assert kinds.index("tool_call") < kinds.index("tool_result")
    assert kinds[-1] == "lifecycle"


async def test_streamed_content_is_coalesced_not_one_row_per_chunk(model, tools):
    """A 500-token reply must not be 500 rows and 500 round trips."""
    session_id = await _session()
    await slog.append(session_id, UserEvent(text="go"))
    model.arm(_text(*[f"chunk{i} " for i in range(40)]))

    await runner.start(session_id)
    await _settle(session_id)

    content = [e.event for e in await slog.get_events(session_id) if e.event.kind == "content"]

    assert 1 <= len(content) < 40, f"expected coalescing, got {len(content)} rows"
    assert "".join(c.text for c in content) == "".join(f"chunk{i} " for i in range(40))


async def test_a_second_start_while_running_is_steering_not_a_second_loop(model, tools):
    session_id = await _session()
    await slog.append(session_id, UserEvent(text="go"))
    model.arm(_text("slow"))

    await runner.start(session_id)
    second = await runner.start(session_id)
    await _settle(session_id)

    assert second is False


async def test_cancel_leaves_a_transcript_that_says_why(model, tools, monkeypatch):
    session_id = await _session()
    await slog.append(session_id, UserEvent(text="go"))

    def slow(messages, tools=None, **kw):
        async def gen():
            yield mc.TextDelta(text="starting")
            await asyncio.sleep(30)
            yield mc.Finish(reason="stop")

        return gen()

    monkeypatch.setattr(mc, "generate", slow)

    await runner.start(session_id)
    await asyncio.sleep(0.2)
    await runner.cancel(session_id)

    row = await pool.fetchrow("SELECT status, terminal_reason FROM sessions WHERE id = $1", uuid.UUID(session_id))
    dones = [e.event for e in await slog.get_events(session_id) if e.event.kind == "done"]

    assert (row["status"], row["terminal_reason"]) == ("cancelled", "cancelled")
    assert [d.reason for d in dones] == ["cancelled"]


async def test_cancel_on_an_idle_session_writes_it_straight_to_cancelled():
    session_id = await _session(status="idle")

    assert await runner.cancel(session_id)

    row = await pool.fetchrow("SELECT status FROM sessions WHERE id = $1", uuid.UUID(session_id))
    assert row["status"] == "cancelled"


async def test_resume_verify_on_wake(model, tools):
    """A call the last run left open is closed before anything else, or the session is unloadable."""
    session_id = await _session()
    await slog.append(session_id, UserEvent(text="go"))
    await slog.append(session_id, ToolCallEvent(id="orphan", name="grep", args={}))
    model.arm(_text("carrying on"))

    await runner.start(session_id)
    await _settle(session_id)

    results = [e.event for e in await slog.get_events(session_id) if e.event.kind == "tool_result"]
    interrupted = [r for r in results if r.id == "orphan"]

    assert interrupted, "the dangling call was never closed"
    assert interrupted[0].error_kind == "interrupted"
    assert not interrupted[0].ok
    # And the model was handed the repair, not a broken message list.
    assert any(m.get("tool_call_id") == "orphan" for m in model.messages_seen[0])


async def test_the_hop_count_is_written_back_as_a_cache(model, tools):
    session_id = await _session()
    await slog.append(session_id, UserEvent(text="go"))
    model.arm(_text("done"))

    await runner.start(session_id)
    await _settle(session_id)

    row = await pool.fetchrow("SELECT hops_used, cursor_seq FROM sessions WHERE id = $1", uuid.UUID(session_id))

    assert row["hops_used"] == 1
    assert row["cursor_seq"] > 0


async def _settle(session_id: str, timeout: float = 10.0) -> None:
    """Wait for the background turn to finish."""
    task = runner._running.get(session_id)
    if task is not None:
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
