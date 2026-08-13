"""One LLM call per hop, structural streaming, and termination an unattended run cannot fake."""

import asyncio
import time

import pytest

from agent_module import events as ev
from agent_module import loop as lp
from model_module import client as mc
from model_module.errors import ModelError

CAP = int(lp._cfg("tools.result_view_cap_chars", 4000))


def _budgets(**over):
    """Config-loaded budgets with per-test overrides."""
    b = lp.Budgets.load("worker")
    for key, value in over.items():
        setattr(b, key, value)
    return b


TOOLS = [
    lp.ToolSpec(name="grep", readonly=True),
    lp.ToolSpec(name="write_file"),
    lp.ToolSpec(name=lp.FINISH_TOOL),
]


def _text(*chunks):
    return [mc.TextDelta(text=c) for c in chunks] + [mc.Finish(reason="stop")]


def _call(name, args="{}", *, id="c1", index=0):
    return [
        mc.ToolCallDelta(index=index, id=id, name=name, arguments=args),
        mc.Finish(reason="tool_calls"),
    ]


@pytest.fixture
def model(monkeypatch):
    """Arm the model with one delta list per hop, counting hops."""

    class Recorder:
        def __init__(self):
            self.hops = 0
            self.messages_seen = []

        def arm(self, *hops):
            self._hops = list(hops)

            def generate(messages, tools=None, **kw):
                self.hops += 1
                self.messages_seen.append(list(messages))
                deltas = self._hops.pop(0) if self._hops else _text("done")

                async def gen():
                    if isinstance(deltas, BaseException):
                        raise deltas
                    for d in deltas:
                        yield d

                return gen()

            monkeypatch.setattr(mc, "generate", generate)
            monkeypatch.setattr(lp.model_client, "generate", generate)

    return Recorder()


def _dispatch(result=None, *, record=None):
    async def dispatch(name, args):
        if record is not None:
            record.append((name, args))
        return result or lp.ResultEnvelope(ok=True, content="ok")

    return dispatch


async def _run(model_fixture, *, mode="attended", messages=None, dispatch=None, budgets=None, tools=TOOLS):
    msgs = messages if messages is not None else [{"role": "user", "content": "go"}]
    return [
        e
        async for e in lp.run_turn(
            msgs,
            tools,
            budgets or _budgets(),
            mode,
            dispatch=dispatch or _dispatch(),
        )
    ], msgs


# --- the hop count ----------------------------------------------------------


@pytest.mark.asyncio
async def test_two_tool_calls_then_text_costs_three_llm_calls(model):
    model.arm(_call("grep", '{"q":"x"}', id="a"), _call("grep", '{"q":"y"}', id="b"), _text("all done"))

    events, _ = await _run(model)

    assert model.hops == 3
    assert isinstance(events[-1], ev.DoneEvent) and events[-1].reason == "turn_end"
    assert [e.id for e in events if isinstance(e, ev.ToolResultEvent)] == ["a", "b"]


@pytest.mark.asyncio
async def test_content_is_yielded_per_delta_not_accumulated(model):
    model.arm(_text("Hel", "lo", " there"))

    events, _ = await _run(model)

    assert [e.text for e in events if isinstance(e, ev.ContentEvent)] == ["Hel", "lo", " there"]


@pytest.mark.asyncio
async def test_reasoning_streams_but_never_enters_messages(model):
    """Qwen3's template strips prior thinking, so replaying it is wrong."""
    model.arm([mc.ReasoningDelta(text="hmm"), mc.TextDelta(text="answer"), mc.Finish(reason="stop")])

    events, messages = await _run(model)

    assert [e.text for e in events if isinstance(e, ev.ReasoningEvent)] == ["hmm"]
    assert all("hmm" not in str(m.get("content") or "") for m in messages)


# --- termination ------------------------------------------------------------


@pytest.mark.asyncio
async def test_attended_ends_on_bare_text(model):
    model.arm(_text("here you go"))
    events, _ = await _run(model, mode="attended")
    assert events[-1].reason == "turn_end"


@pytest.mark.asyncio
async def test_unattended_bare_text_does_not_end_the_run(model):
    model.arm(_text("I think I am done"), _call(lp.FINISH_TOOL))

    events, _ = await _run(model, mode="unattended")

    assert model.hops == 2
    assert events[-1].reason == "completed"


@pytest.mark.asyncio
async def test_unattended_gets_one_nudge_near_the_cap(model):
    model.arm(*[_text("still talking")] * 4)

    events, _ = await _run(model, mode="unattended", budgets=_budgets(max_hops=3))

    nudges = [e for e in events if isinstance(e, ev.UserEvent)]
    assert len(nudges) == 1 and lp.FINISH_TOOL in nudges[0].text
    assert events[-1].reason == "max_hops"


@pytest.mark.asyncio
async def test_budget_exhaustion_is_never_completed(model):
    model.arm(*[_call("grep") for _ in range(10)])

    events, _ = await _run(model, mode="unattended", budgets=_budgets(max_hops=3))

    assert events[-1].reason == "max_hops"
    assert [e.hops_used for e in events if isinstance(e, ev.BudgetEvent)] == [1, 2, 3]


# --- errors are model input -------------------------------------------------


@pytest.mark.asyncio
async def test_tool_failure_comes_back_as_a_readable_result(model):
    model.arm(_call("grep"), _text("I will try something else"))
    dispatch = _dispatch(lp.ResultEnvelope(ok=False, content="disk on fire", error_kind="upstream_error"))

    events, messages = await _run(model, dispatch=dispatch)

    result = next(e for e in events if isinstance(e, ev.ToolResultEvent))
    assert result.ok is False and result.error_kind == "upstream_error"
    assert any(m.get("role") == "tool" and "disk on fire" in m["content"] for m in messages)
    assert events[-1].reason == "turn_end"


@pytest.mark.asyncio
async def test_unknown_tool_is_reported_not_raised(model):
    model.arm(_call("teleport"), _text("ok"))

    events, _ = await _run(model)

    result = next(e for e in events if isinstance(e, ev.ToolResultEvent))
    assert result.error_kind == "not_found" and "teleport" in result.content


@pytest.mark.asyncio
async def test_malformed_args_get_one_free_repair_hop(model):
    model.arm(_call("grep", "{not json"), _call("grep", '{"q":"x"}'), _text("done"))

    events, _ = await _run(model)

    result = next(e for e in events if isinstance(e, ev.ToolResultEvent))
    assert result.error_kind == "invalid_args"
    # Three model calls, but the repair is not charged, so only two hops.
    assert model.hops == 3
    assert [e.hops_used for e in events if isinstance(e, ev.BudgetEvent)] == [1, 2]


@pytest.mark.asyncio
async def test_a_tool_that_keeps_failing_is_cut_off(model):
    model.arm(*[_call("grep") for _ in range(5)], _text("giving up"))
    calls = []
    dispatch = _dispatch(lp.ResultEnvelope(ok=False, content="nope", error_kind="timeout"), record=calls)

    events, _ = await _run(model, budgets=_budgets(per_tool_attempts=2), dispatch=dispatch)

    assert len(calls) == 2, "the cap stops us dispatching a fourth time"
    cutoff = [e for e in events if isinstance(e, ev.ToolResultEvent) and e.error_kind == "upstream_error"]
    assert cutoff and "another approach" in cutoff[0].content


@pytest.mark.asyncio
async def test_a_success_clears_the_failure_streak(model):
    outcomes = [
        lp.ResultEnvelope(ok=False, content="x", error_kind="timeout"),
        lp.ResultEnvelope(ok=True, content="fine"),
        lp.ResultEnvelope(ok=False, content="x", error_kind="timeout"),
    ]
    model.arm(*[_call("grep") for _ in range(3)], _text("done"))

    async def dispatch(name, args):
        return outcomes.pop(0)

    events, _ = await _run(model, budgets=_budgets(per_tool_attempts=2), dispatch=dispatch)

    assert not [e for e in events if isinstance(e, ev.ToolResultEvent) and e.error_kind == "upstream_error"]


# --- model failure ----------------------------------------------------------


@pytest.mark.asyncio
async def test_retryable_model_error_reattempts_the_hop(model):
    model.arm(ModelError("timeout", retryable=True, kind="timeout"), _text("recovered"))

    events, _ = await _run(model)

    assert model.hops == 2
    assert events[-1].reason == "turn_end"


@pytest.mark.asyncio
async def test_hop_reattempts_are_bounded(model):
    model.arm(*[ModelError("timeout", retryable=True, kind="timeout")] * 10)

    events, _ = await _run(model, budgets=_budgets(model_retries=2))

    assert model.hops == 3, "the first attempt plus two re-attempts"
    assert events[-1].reason == "model_error"


@pytest.mark.asyncio
async def test_terminal_model_error_ends_immediately(model):
    model.arm(ModelError("bad", retryable=False, kind="bad_request"))

    events, _ = await _run(model)

    assert model.hops == 1
    assert events[-1].reason == "model_error"


# --- the transcript invariant -----------------------------------------------


@pytest.mark.asyncio
async def test_every_tool_call_is_closed_by_exactly_one_result(model):
    model.arm(_call("grep", id="a"), _call("teleport", id="b"), _call("grep", "{bad", id="c"), _text("done"))

    events, _ = await _run(model)

    opened = [e.id for e in events if isinstance(e, ev.ToolCallEvent)]
    closed = [e.id for e in events if isinstance(e, ev.ToolResultEvent)]
    assert opened == closed == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_oversized_result_is_view_capped_with_a_total(model):
    model.arm(_call("grep"), _text("done"))
    big = "x" * (CAP + 500)

    events, messages = await _run(model, dispatch=_dispatch(lp.ResultEnvelope(ok=True, content=big, ref="r1")))

    result = next(e for e in events if isinstance(e, ev.ToolResultEvent))
    assert len(result.content) == CAP
    assert result.total_chars == len(big) and result.ref == "r1"
    # The model still sees the whole thing; only the view is capped.
    assert messages[-2]["content"] == big


# --- readonly drives concurrency --------------------------------------------


def _calls(*specs):
    """One hop issuing several tool calls."""
    deltas = []
    for i, (name, cid) in enumerate(specs):
        deltas.append(mc.ToolCallDelta(index=i, id=cid, name=name, arguments="{}"))
    return deltas + [mc.Finish(reason="tool_calls")]


@pytest.mark.asyncio
async def test_readonly_calls_run_in_parallel(model):
    """Three 50ms reads take ~50ms together, not 150ms."""
    model.arm(_calls(("grep", "a"), ("grep", "b"), ("grep", "c")), _text("done"))

    async def slow_read(name, args):
        await asyncio.sleep(0.05)
        return lp.ResultEnvelope(ok=True, content="found")

    started = time.monotonic()
    events, _ = await _run(model, dispatch=slow_read)
    elapsed = time.monotonic() - started

    assert elapsed < 0.12, "serial execution would take at least 0.15s"
    assert {e.id for e in events if isinstance(e, ev.ToolResultEvent)} == {"a", "b", "c"}


@pytest.mark.asyncio
async def test_writes_run_serially_and_in_order(model):
    model.arm(_calls(("write_file", "a"), ("write_file", "b")), _text("done"))
    live = 0
    order = []

    async def tracked(name, args):
        nonlocal live
        live += 1
        assert live == 1, "two writes overlapped"
        await asyncio.sleep(0.01)
        order.append(name)
        live -= 1
        return lp.ResultEnvelope(ok=True, content="written")

    events, _ = await _run(model, dispatch=tracked)

    assert len(order) == 2
    assert [e.id for e in events if isinstance(e, ev.ToolResultEvent)] == ["a", "b"]


@pytest.mark.asyncio
async def test_a_write_between_reads_splits_the_batches(model):
    """read read write read must not hoist the last read past the write."""
    model.arm(_calls(("grep", "a"), ("grep", "b"), ("write_file", "w"), ("grep", "c")), _text("done"))
    sequence = []

    async def tracked(name, args):
        sequence.append(f"start:{name}")
        await asyncio.sleep(0.01)
        sequence.append(f"end:{name}")
        return lp.ResultEnvelope(ok=True, content="ok")

    await _run(model, dispatch=tracked)

    assert sequence.index("end:write_file") > sequence.index("start:grep")
    # The trailing read begins only after the write has finished.
    assert sequence[-2:] == ["start:grep", "end:grep"]


@pytest.mark.asyncio
async def test_every_parallel_call_is_still_closed_exactly_once(model):
    """The transcript invariant holds when a batch has a bad call in it."""
    model.arm(_calls(("grep", "a"), ("teleport", "b"), ("grep", "c")), _text("done"))

    events, messages = await _run(model)

    opened = sorted(e.id for e in events if isinstance(e, ev.ToolCallEvent))
    closed = sorted(e.id for e in events if isinstance(e, ev.ToolResultEvent))
    assert opened == closed == ["a", "b", "c"]
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert sorted(m["tool_call_id"] for m in tool_msgs) == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_a_failed_finish_task_does_not_complete_the_run(model):
    """Completion comes from the result, never from the call being issued."""
    model.arm(_call(lp.FINISH_TOOL, id="f"), _text("still here"))
    dispatch = _dispatch(lp.ResultEnvelope(ok=False, content="could not finish", error_kind="upstream_error"))

    events, _ = await _run(model, mode="unattended", dispatch=dispatch, budgets=_budgets(max_hops=2))

    assert events[-1].reason != "completed"


@pytest.mark.asyncio
async def test_finish_task_with_bad_args_does_not_complete_the_run(model):
    model.arm(_call(lp.FINISH_TOOL, "{bad", id="f"), _text("still here"))

    events, _ = await _run(model, mode="unattended", budgets=_budgets(max_hops=2))

    assert events[-1].reason != "completed"


@pytest.mark.asyncio
async def test_a_raising_dispatch_becomes_a_failed_result(model):
    """A tool bug must not take the run down or leave a call unclosed."""
    model.arm(_calls(("grep", "a"), ("grep", "b")), _text("recovered"))

    async def explode(name, args):
        if args is not None and name == "grep":
            raise RuntimeError("tool is broken")
        return lp.ResultEnvelope(ok=True, content="ok")

    events, _ = await _run(model, dispatch=explode)

    results = [e for e in events if isinstance(e, ev.ToolResultEvent)]
    assert sorted(r.id for r in results) == ["a", "b"]
    assert all(r.ok is False and r.error_kind == "upstream_error" for r in results)
    assert events[-1].reason == "turn_end"


@pytest.mark.asyncio
async def test_cancellation_closes_open_calls_and_says_why(model):
    """A tool_call with no result cannot be resumed, so aborts synthesize one."""
    model.arm(_calls(("grep", "a"), ("grep", "b")))

    async def never(name, args):
        await asyncio.sleep(10)
        return lp.ResultEnvelope(ok=True, content="never")

    messages = [{"role": "user", "content": "go"}]
    seen = []

    async def consume():
        async for event in lp.run_turn(messages, TOOLS, _budgets(), "attended", dispatch=never):
            seen.append(event)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert any(isinstance(e, ev.DoneEvent) and e.reason == "cancelled" for e in seen)
    opened = {e.id for e in seen if isinstance(e, ev.ToolCallEvent)}
    closed = {m["tool_call_id"] for m in messages if m.get("role") == "tool"}
    assert opened == closed == {"a", "b"}


@pytest.mark.asyncio
async def test_duplicate_and_empty_tool_call_ids_are_made_unique(model):
    model.arm(_calls(("grep", ""), ("grep", "")), _text("done"))

    events, messages = await _run(model)

    ids = [e.id for e in events if isinstance(e, ev.ToolCallEvent)]
    assert len(set(ids)) == 2 and all(ids)
    assert sorted(m["tool_call_id"] for m in messages if m.get("role") == "tool") == sorted(ids)


@pytest.mark.asyncio
async def test_wall_clock_bounds_a_slow_tool_not_just_the_gap_between_hops(model):
    model.arm(_call("grep"), _text("done"))

    async def slow(name, args):
        await asyncio.sleep(5)
        return lp.ResultEnvelope(ok=True, content="eventually")

    started = time.monotonic()
    events, _ = await _run(model, dispatch=slow, budgets=_budgets(wall_clock_s=0.2))
    elapsed = time.monotonic() - started

    assert elapsed < 1, "the budget was advisory, not enforced"
    assert events[-1].reason == "wall_clock"


@pytest.mark.asyncio
async def test_the_per_tool_cap_holds_inside_one_parallel_batch(model):
    """Five parallel calls to one tool must not all dispatch under a cap of two."""
    model.arm(_calls(*[("grep", f"c{i}") for i in range(5)]), _text("done"))
    dispatched = []

    async def counted(name, args):
        dispatched.append(name)
        return lp.ResultEnvelope(ok=False, content="nope", error_kind="timeout")

    events, _ = await _run(model, dispatch=counted, budgets=_budgets(per_tool_attempts=2))

    assert len(dispatched) == 2
    assert len([e for e in events if isinstance(e, ev.ToolResultEvent)]) == 5


@pytest.mark.asyncio
async def test_the_hop_meter_does_not_repeat_on_a_model_retry(model):
    model.arm(ModelError("timeout", retryable=True, kind="timeout"), _text("recovered"))

    events, _ = await _run(model)

    assert [e.hops_used for e in events if isinstance(e, ev.BudgetEvent)] == [1]


@pytest.mark.asyncio
async def test_a_truncated_reply_is_not_a_clean_turn_end(model):
    """finish_reason 'length' means max_tokens cut it off mid-thought."""
    model.arm([mc.TextDelta(text="half a sen"), mc.Finish(reason="length")])

    events, _ = await _run(model)

    assert events[-1].reason == "context_overflow"


@pytest.mark.asyncio
async def test_an_empty_completion_does_not_spin(model):
    model.arm(*[[mc.Finish(reason="stop")]] * 5)

    events, _ = await _run(model, mode="unattended", budgets=_budgets(max_hops=5))

    assert model.hops == 1
    assert events[-1].reason == "model_error"


@pytest.mark.asyncio
async def test_readonly_batch_really_overlaps(model):
    model.arm(_calls(("grep", "a"), ("grep", "b"), ("grep", "c")), _text("done"))
    live = 0
    peak = 0

    async def tracked(name, args):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.01)
        live -= 1
        return lp.ResultEnvelope(ok=True, content="found")

    await _run(model, dispatch=tracked)

    assert peak == 3
