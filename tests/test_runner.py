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
from tests.dbgate import require_db
from tool_module.envelope import ToolSpec, fail, ok

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
    runner._cancelling.clear()
    await asyncio.sleep(0)
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
    """A manifest of two tools whose dispatch records the calls it is given."""
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

    folded = await runner.fold(await runner.load(session_id))
    messages, hops = folded.messages, folded.hops_used

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

    messages = (await runner.fold(await runner.load(session_id))).messages

    assert "I should" not in str(messages)


async def test_event_replay_deterministic():
    """Folding one log twice produces identical messages."""
    session_id = await _session()
    await slog.append(session_id, UserEvent(text="hi"))
    await slog.append(session_id, ContentEvent(text="hello"))

    session = await runner.load(session_id)
    first = (await runner.fold(session)).messages
    second = (await runner.fold(session)).messages

    assert first == second


async def test_hops_reset_at_a_done_and_carry_inside_one_run():
    from agent_module.events import BudgetEvent

    session_id = await _session()
    await slog.append(session_id, BudgetEvent(hops_used=3, hops_max=6))
    mid_run = (await runner.fold(await runner.load(session_id))).hops_used

    await slog.append(session_id, DoneEvent(reason="turn_end"))
    after_done = (await runner.fold(await runner.load(session_id))).hops_used

    assert mid_run == 3, "a resume inside a run keeps counting"
    assert after_done == 0, "a new turn budgets from zero"


async def test_a_stored_result_carries_its_ref_so_the_tail_is_recoverable():
    session_id = await _session()
    await slog.append(session_id, ToolCallEvent(id="c1", name="grep", args={}))
    await slog.append(
        session_id,
        ToolResultEvent(id="c1", ok=True, content="head", total_chars=9000, ref="b_1"),
    )

    messages = (await runner.fold(await runner.load(session_id))).messages

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
    """Streamed text chunks are merged into a few content events."""
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


async def test_cancel_on_an_idle_session_still_writes_a_done():
    """Cancelling an idle session appends a done event alongside the status change."""
    session_id = await _session(status="idle")

    assert await runner.cancel(session_id)

    row = await pool.fetchrow("SELECT status FROM sessions WHERE id = $1", uuid.UUID(session_id))
    dones = [e.event for e in await slog.get_events(session_id) if e.event.kind == "done"]

    assert row["status"] == "cancelled"
    assert [d.reason for d in dones] == ["cancelled"]


async def test_cancelling_an_idle_session_closes_a_call_a_dead_run_left_open():
    session_id = await _session(status="idle")
    await slog.append(session_id, ToolCallEvent(id="c1", name="run_command", args={}))

    assert await runner.cancel(session_id)

    kinds = [e.event.kind for e in await slog.get_events(session_id)]
    assert kinds == ["tool_call", "tool_result", "done", "lifecycle"]


async def test_a_restarted_session_budgets_from_zero_after_a_cancel():
    from agent_module.events import BudgetEvent

    session_id = await _session(status="running")
    await slog.append(session_id, BudgetEvent(hops_used=6, hops_max=6))
    await runner.cancel(session_id)

    hops = (await runner.fold(await runner.load(session_id))).hops_used

    assert hops == 0, "the hop count did not reset"


async def test_two_concurrent_cancels_produce_exactly_one_terminal(model, tools, monkeypatch):
    """Two cancels racing each other write one terminal between them."""
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
    await asyncio.sleep(0.4)
    await asyncio.gather(runner.cancel(session_id), runner.cancel(session_id))

    row = await pool.fetchrow("SELECT status, terminal_reason FROM sessions WHERE id = $1", uuid.UUID(session_id))
    dones = [e.event for e in await slog.get_events(session_id) if e.event.kind == "done"]

    assert (row["status"], row["terminal_reason"]) == ("cancelled", "cancelled")
    assert len(dones) == 1, f"expected one terminal, got {[d.reason for d in dones]}"
    # The session starts again after the cancel.
    assert await runner.start(session_id)
    await runner.cancel(session_id)


async def test_the_reaper_lands_a_terminal_the_database_refused(model, tools, monkeypatch):
    """A terminal transition the database refuses is retried by the reaper until it lands."""
    session_id = await _session()
    await slog.append(session_id, UserEvent(text="go"))
    model.arm(_text("done"))

    real_transition = runner.lifecycle.transition
    failures = {"left": 1}

    async def flaky(*args, **kwargs):
        if failures["left"] and args[2] in ("idle", "completed", "failed", "cancelled"):
            failures["left"] -= 1
            raise RuntimeError("the database went away")
        return await real_transition(*args, **kwargs)

    monkeypatch.setattr(runner.lifecycle, "transition", flaky)
    # Delays only; the attempt cap keeps its configured value.
    delays = {"harness.terminal_retry_s": 0.05, "harness.terminal_retry_max_s": 0.05}
    monkeypatch.setattr(runner, "_cfg", lambda key, default: delays.get(key, default))

    await runner.start(session_id)
    await _settle(session_id)

    # The turn ended without a terminal; the reaper retries behind it.
    for _ in range(40):
        row = await pool.fetchrow("SELECT status FROM sessions WHERE id = $1", uuid.UUID(session_id))
        if row["status"] != "running":
            break
        await asyncio.sleep(0.1)

    dones = [e.event for e in await slog.get_events(session_id) if e.event.kind == "done"]

    assert row["status"] == "idle", "the reaper never landed the terminal"
    assert len(dones) == 1, "the retry appended a second done"


async def test_resume_verify_on_wake(model, tools):
    """A call the previous run left open is closed before the resumed turn calls the model."""
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
    # The repaired result is in the messages the model was sent.
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


async def _settle(session_id: str, timeout: float = 45.0) -> None:
    """Wait for the background turn to finish; the timeout bounds a hung suite."""
    task = runner._running.get(session_id)
    if task is not None:
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)


# --- a session must always fold back into a loadable conversation --------------


def _assert_loadable(messages: list[dict]) -> None:
    """Every tool message answers a call in the nearest preceding assistant message."""
    open_ids: set[str] = set()
    for i, message in enumerate(messages):
        role = message.get("role")
        if role == "assistant":
            open_ids = {c["id"] for c in message.get("tool_calls") or []}
        elif role == "tool":
            assert message["tool_call_id"] in open_ids, (
                f"message {i}: tool result {message['tool_call_id']!r} answers no preceding tool_calls"
            )
        elif role == "user":
            # A user message ends the assistant turn it follows.
            open_ids = set()


async def test_a_message_typed_mid_call_still_folds_legally():
    """A user message appended between a call and its result still folds legally."""
    session_id = await _session()
    await slog.append(session_id, UserEvent(text="do it"))
    await slog.append(session_id, ToolCallEvent(id="c1", name="grep", args={}))
    await slog.append(session_id, UserEvent(text="actually, hurry"))
    await slog.append(session_id, ToolResultEvent(id="c1", ok=True, content="found"))

    messages = (await runner.fold(await runner.load(session_id))).messages

    _assert_loadable(messages)
    # The message is kept, ordered after the result that was already open.
    assert [m["content"] for m in messages if m["role"] == "user"] == ["do it", "actually, hurry"]
    assert messages.index({"role": "tool", "tool_call_id": "c1", "content": "found"}) < len(messages) - 1


async def test_an_interrupted_call_repaired_after_a_user_message_still_folds_legally():
    """A synthesized result appended after a user message still folds legally."""
    session_id = await _session()
    await slog.append(session_id, ToolCallEvent(id="c1", name="run_command", args={}))
    await slog.append(session_id, UserEvent(text="any update?"))
    await slog.close_dangling(session_id)

    messages = (await runner.fold(await runner.load(session_id))).messages

    _assert_loadable(messages)


async def test_a_reused_tool_call_id_across_runs_still_folds_legally():
    session_id = await _session()
    for _ in range(2):
        await slog.append(session_id, ToolCallEvent(id="call_1", name="grep", args={}))
        await slog.append(session_id, ToolResultEvent(id="call_1", ok=True, content="x"))
        await slog.append(session_id, DoneEvent(reason="turn_end"))

    messages = (await runner.fold(await runner.load(session_id))).messages

    _assert_loadable(messages)


async def test_the_loop_mints_tool_call_ids_that_do_not_repeat_across_turns():
    from agent_module import loop as lp

    first = lp._Hop()
    first._calls[0] = lp._PartialCall(id="", name="grep")
    second = lp._Hop()
    second._calls[0] = lp._PartialCall(id="", name="grep")

    minted = [first.finish(set())[0].id, second.finish(set())[0].id]

    assert minted[0] != minted[1], "two turns minted the same tool call id"


# --- the handoff to an unattended run -----------------------------------------


async def test_start_flips_the_mode_in_the_same_update_as_the_status(monkeypatch):
    """start writes the new mode and the new status in one update."""
    session_id = await _session(status="idle", mode="attended")

    async def noop(_session_id):
        return None

    monkeypatch.setattr(runner, "_drive", noop)
    assert await runner.start(session_id, mode="unattended", reason="approved")
    await _settle(session_id)

    row = await pool.fetchrow("SELECT mode, status FROM sessions WHERE id = $1", uuid.UUID(session_id))
    events = [e.event for e in await slog.get_events(session_id)]

    assert (row["mode"], row["status"]) == ("unattended", "running")
    assert [e.kind for e in events] == ["lifecycle"]
    assert (events[0].from_, events[0].to, events[0].reason) == ("idle", "running", "approved")


async def test_a_plain_wake_leaves_the_mode_alone(monkeypatch):
    session_id = await _session(status="idle", mode="unattended")

    async def noop(_session_id):
        return None

    monkeypatch.setattr(runner, "_drive", noop)
    await runner.start(session_id)
    await _settle(session_id)

    row = await pool.fetchrow("SELECT mode FROM sessions WHERE id = $1", uuid.UUID(session_id))
    assert row["mode"] == "unattended"


# --- how an unattended run may and may not end ---------------------------------


@pytest.fixture
def small_budgets(monkeypatch):
    """A hop cap low enough that a run exhausts it in a few round trips."""

    def load(mode="attended"):
        return runner.Budgets(max_hops=2, wall_clock_s=30.0, per_tool_attempts=3, model_retries=1)

    monkeypatch.setattr(runner.Budgets, "load", load)


async def test_an_unattended_run_completes_only_through_finish_task(model, tools):
    session_id = await _session(mode="unattended")
    await slog.append(session_id, UserEvent(text="do the thing"))
    model.arm(_call("finish_task", '{"summary": "did it"}'))

    await runner.start(session_id)
    await _settle(session_id)

    row = await pool.fetchrow("SELECT status, mode, terminal_reason FROM sessions WHERE id = $1", uuid.UUID(session_id))

    assert (row["status"], row["terminal_reason"]) == ("completed", "completed")
    # The finished run hands the session back as attended.
    assert row["mode"] == "attended"


async def test_text_alone_never_completes_an_unattended_run(model, tools, small_budgets):
    """An unattended run that only produces text ends as failed with reason max_hops."""
    session_id = await _session(mode="unattended")
    await slog.append(session_id, UserEvent(text="do the thing"))
    model.arm(_text("I have finished!"), _text("Truly finished."), _text("Done."))

    await runner.start(session_id)
    await _settle(session_id)

    row = await pool.fetchrow("SELECT status, terminal_reason FROM sessions WHERE id = $1", uuid.UUID(session_id))
    dones = [e.event.reason for e in await slog.get_events(session_id) if e.event.kind == "done"]

    assert (row["status"], row["terminal_reason"]) == ("failed", "max_hops")
    assert dones == ["max_hops"]


async def test_the_nudge_lands_in_the_transcript_before_the_budget_runs_out(model, tools, small_budgets):
    session_id = await _session(mode="unattended")
    await slog.append(session_id, UserEvent(text="go"))
    model.arm(_text("thinking out loud"), _text("still going"))

    await runner.start(session_id)
    await _settle(session_id)

    users = [e.event for e in await slog.get_events(session_id) if e.event.kind == "user"]

    assert [u.source for u in users] == ["human", "system"]
    assert "finish_task" in users[1].text


async def test_a_finish_task_that_fails_does_not_complete_the_run(model, monkeypatch, small_budgets):
    """A finish_task whose result is an error leaves the run uncompleted."""
    session_id = await _session(mode="unattended")
    await slog.append(session_id, UserEvent(text="go"))

    async def manifest(user_id, mcp=None):
        return [ToolSpec(name="finish_task")]

    def bind(ctx, **kw):
        async def dispatch(name, args):
            return fail("upstream_error", "the summary was rejected")

        return dispatch

    monkeypatch.setattr(runner.registry, "manifest", manifest)
    monkeypatch.setattr(runner.registry, "bind", bind)
    model.arm(_call("finish_task", '{"summary": "s"}'), _call("finish_task", '{"summary": "s"}', id="c2"))

    await runner.start(session_id)
    await _settle(session_id)

    row = await pool.fetchrow("SELECT status, terminal_reason FROM sessions WHERE id = $1", uuid.UUID(session_id))

    assert row["status"] == "failed"
    assert row["terminal_reason"] != "completed"


# --- resume ---------------------------------------------------------------------


async def test_a_killed_run_resumes_without_repeating_its_side_effect(model, tools):
    """A resumed run reports the unclosed call as interrupted and does not dispatch it again."""
    session_id = await _session(mode="unattended", status="running")
    await slog.append(session_id, UserEvent(text="send the invoice"))
    await slog.append(session_id, ToolCallEvent(id="c1", name="send_email", args={"to": "a@b.c"}))

    # The process died here; the startup sweep fails the session.
    await runner.lifecycle.sweep_interrupted()
    failed = await pool.fetchrow("SELECT status, terminal_reason FROM sessions WHERE id = $1", uuid.UUID(session_id))

    # A human restarts it.
    model.arm(_call("finish_task", '{"summary": "verified, already sent"}'))
    await runner.start(session_id)
    await _settle(session_id)

    dispatched = [name for name, _ in tools]
    results = [e.event for e in await slog.get_events(session_id) if e.event.kind == "tool_result"]

    assert (failed["status"], failed["terminal_reason"]) == ("failed", "interrupted")
    assert "send_email" not in dispatched, "the side effect was repeated"
    assert results[0].error_kind == "interrupted"
    assert "verify before retrying" in results[0].content


# --- the context ladder ---------------------------------------------------------


@pytest.fixture
def tiny_window(monkeypatch):
    """A window small enough that a few results overflow it.

    The ceiling sits above the ~1035-token system prompt, which rung 1 cannot clear.
    """

    def cfg(key, default):
        return {
            "llm.context_window": 2090,
            "llm.max_tokens": 400,
            "context.recovery_threshold": 0.8,
            "context.chars_per_token": 4,
        }.get(key, default)

    monkeypatch.setattr(runner, "_cfg", cfg)


async def _bulky_log(session_id: str, results: int = 6, size: int = 900) -> list[str]:
    """A log of blobbed results, oldest first."""
    refs = []
    await slog.append(session_id, UserEvent(text="do the thing"))
    for i in range(results):
        ref = await slog.save_blob(session_id, "z" * size)
        refs.append(ref)
        await slog.append(session_id, ToolCallEvent(id=f"c{i}", name="grep", args={}))
        await slog.append(
            session_id,
            ToolResultEvent(id=f"c{i}", ok=True, content="z" * 200, total_chars=size, ref=ref),
        )
    return refs


async def test_a_log_over_the_input_budget_folds_to_a_view_under_it(tiny_window):
    session_id = await _session()
    await _bulky_log(session_id)

    folded = await runner.fold(await runner.load(session_id))
    ceiling = int(runner._input_budget() * 0.8)

    assert runner._estimate_tokens(folded.messages) <= ceiling
    assert folded.transform is not None
    assert folded.transform.rung == 1


async def test_the_ladder_clears_the_oldest_results_first(tiny_window):
    session_id = await _session()
    refs = await _bulky_log(session_id)

    folded = await runner.fold(await runner.load(session_id))
    dropped = folded.transform.dropped_refs

    assert dropped == refs[: len(dropped)], "clearing must start at the oldest"
    # A cleared result keeps its ref in the view, next to the tool that reads it back.
    for ref in dropped:
        assert any(ref in str(m.get("content")) for m in folded.messages)
    assert any("read_result" in str(m.get("content")) for m in folded.messages)


async def test_a_result_with_no_ref_is_never_cleared(tiny_window):
    """A result carrying no blob ref stays in the view."""
    session_id = await _session()
    await slog.append(session_id, UserEvent(text="go"))
    for i in range(6):
        await slog.append(session_id, ToolCallEvent(id=f"c{i}", name="grep", args={}))
        await slog.append(session_id, ToolResultEvent(id=f"c{i}", ok=True, content="y" * 900))

    folded = await runner.fold(await runner.load(session_id))

    assert folded.transform is None
    assert all("cleared from view" not in str(m.get("content")) for m in folded.messages)


async def test_the_same_log_folds_byte_identically_twice(tiny_window):
    """A log that trips the ladder folds to the same messages and the same drops twice."""
    session_id = await _session()
    await _bulky_log(session_id)
    session = await runner.load(session_id)

    first = await runner.fold(session)
    second = await runner.fold(session)

    assert first.messages == second.messages
    assert first.transform.dropped_refs == second.transform.dropped_refs


async def test_a_view_under_budget_is_left_alone(tiny_window):
    session_id = await _session()
    await slog.append(session_id, UserEvent(text="hello"))

    folded = await runner.fold(await runner.load(session_id))

    assert folded.transform is None


async def test_the_drop_is_recorded_in_the_transcript(model, tools, tiny_window):
    """A drop appends one view_transform event and leaves the stored results as they were."""
    session_id = await _session()
    await _bulky_log(session_id)
    before = len(await slog.get_events(session_id))
    model.arm(_text("carrying on"))

    await runner.start(session_id)
    await _settle(session_id)

    events = await slog.get_events(session_id)
    transforms = [e.event for e in events if e.event.kind == "view_transform"]
    results = [e.event for e in events if e.event.kind == "tool_result"]

    assert len(transforms) == 1 and transforms[0].dropped_refs
    # The stored results still hold their own text.
    assert all("cleared from view" not in r.content for r in results)
    assert len(events) > before


async def test_rung_1_clears_results_and_nothing_else(monkeypatch):
    """Rung 1 clears results and stops there, even when the view is still over the ceiling."""

    def cfg(key, default):
        return {
            "llm.context_window": 500,
            "llm.max_tokens": 100,
            "context.recovery_threshold": 0.8,
            "context.chars_per_token": 4,
        }.get(key, default)

    monkeypatch.setattr(runner, "_cfg", cfg)
    session_id = await _session()
    refs = await _bulky_log(session_id, results=2)

    folded = await runner.fold(await runner.load(session_id))

    # Every clearable result is gone and the view is still over the ceiling.
    assert folded.transform.dropped_refs == refs
    assert runner._estimate_tokens(folded.messages) > int(runner._input_budget() * 0.8)
