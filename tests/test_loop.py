"""
Task 2 conformance: one LLM call per hop, structural streaming, and termination
that an unattended run cannot fake.
"""

import asyncio
import time

import pytest

from agent_module import events as ev
from agent_module import loop as lp
from model_module import client as mc
from model_module.errors import ModelError

TOOLS = [
    lp.ToolSpec(name="grep", readonly=True),
    lp.ToolSpec(name="write_file"),
    lp.ToolSpec(name=lp.FINISH_TOOL),
]


def _text(*chunks):
    return [mc.TextDelta(text=c) for c in chunks] + [mc.Finish(reason="stop")]


def _call(name, args='{}', *, id="c1", index=0):
    return [
        mc.ToolCallDelta(index=index, id=id, name=name, arguments=args),
        mc.Finish(reason="tool_calls"),
    ]


@pytest.fixture
def model(monkeypatch):
    """Arm the model with one delta list per hop; counts hops."""

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
            budgets or lp.Budgets(),
            mode,
            dispatch=dispatch or _dispatch(),
        )
    ], msgs


# --- the hop count ----------------------------------------------------------


@pytest.mark.asyncio
async def test_two_tool_calls_then_text_costs_three_llm_calls(model):
    """The headline: one model call per hop, not the old three to five."""
    model.arm(_call("grep", '{"q":"x"}', id="a"), _call("grep", '{"q":"y"}', id="b"), _text("all done"))

    events, _ = await _run(model)

    assert model.hops == 3
    assert isinstance(events[-1], ev.DoneEvent) and events[-1].reason == "turn_end"
    assert [e.id for e in events if isinstance(e, ev.ToolResultEvent)] == ["a", "b"]


@pytest.mark.asyncio
async def test_content_is_yielded_per_delta_not_accumulated(model):
    """Streaming is structural: no accumulation step exists."""
    model.arm(_text("Hel", "lo", " there"))

    events, _ = await _run(model)

    assert [e.text for e in events if isinstance(e, ev.ContentEvent)] == ["Hel", "lo", " there"]


@pytest.mark.asyncio
async def test_reasoning_streams_but_never_enters_messages(model):
    """Qwen3's template strips prior thinking; replaying it is wrong and costly."""
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
    """Only finish_task makes an unattended exit safe."""
    model.arm(_text("I think I am done"), _call(lp.FINISH_TOOL))

    events, _ = await _run(model, mode="unattended")

    assert model.hops == 2
    assert events[-1].reason == "completed"


@pytest.mark.asyncio
async def test_unattended_gets_one_nudge_near_the_cap(model):
    model.arm(*[_text("still talking")] * 4)

    events, _ = await _run(model, mode="unattended", budgets=lp.Budgets(max_hops=3))

    nudges = [e for e in events if isinstance(e, ev.UserEvent)]
    assert len(nudges) == 1 and lp.FINISH_TOOL in nudges[0].text
    assert events[-1].reason == "max_hops"


@pytest.mark.asyncio
async def test_budget_exhaustion_is_never_completed(model):
    model.arm(*[_call("grep") for _ in range(10)])

    events, _ = await _run(model, mode="unattended", budgets=lp.Budgets(max_hops=3))

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
    # The model gets to read it and react, rather than the run dying.
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
    """The repair must not eat a hop the actual work needs."""
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

    events, _ = await _run(model, budgets=lp.Budgets(per_tool_attempts=2), dispatch=dispatch)

    assert len(calls) == 2, "the cap stops us dispatching a fourth time"
    cutoff = [e for e in events if isinstance(e, ev.ToolResultEvent) and e.error_kind == "upstream_error"]
    assert cutoff and "another approach" in cutoff[0].content


@pytest.mark.asyncio
async def test_a_success_clears_the_failure_streak(model):
    """A tool that works again is not punished for an earlier bad patch."""
    outcomes = [
        lp.ResultEnvelope(ok=False, content="x", error_kind="timeout"),
        lp.ResultEnvelope(ok=True, content="fine"),
        lp.ResultEnvelope(ok=False, content="x", error_kind="timeout"),
    ]
    model.arm(*[_call("grep") for _ in range(3)], _text("done"))

    async def dispatch(name, args):
        return outcomes.pop(0)

    events, _ = await _run(model, budgets=lp.Budgets(per_tool_attempts=2), dispatch=dispatch)

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

    events, _ = await _run(model, budgets=lp.Budgets(model_retries=2))

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
    big = "x" * (lp.RESULT_VIEW_CAP + 500)

    events, messages = await _run(model, dispatch=_dispatch(lp.ResultEnvelope(ok=True, content=big, ref="r1")))

    result = next(e for e in events if isinstance(e, ev.ToolResultEvent))
    assert len(result.content) == lp.RESULT_VIEW_CAP
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
    """A write may depend on what came before it, so it never overlaps."""
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
