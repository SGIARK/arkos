"""
The agent loop. One `run_turn`, replacing Agent.step/step_stream/choose_transition,
all of state_module, and ComputerAgent.run.

Never touches Postgres. Only caller of the model. Errors are model input, not
control flow: a failed tool comes back as a tool_result the model can read and
react to. Only cancellation propagates.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from agent_module.events import (
    BudgetEvent,
    ContentEvent,
    DoneEvent,
    Event,
    ReasoningEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserEvent,
)
from model_module import client as model_client
from model_module.errors import ModelError

logger = logging.getLogger(__name__)

Mode = Literal["attended", "unattended"]

# The control tool that makes an unattended exit safe.
FINISH_TOOL = "finish_task"

# View cap for tool_result.content. The full text goes to a blob and rides `ref`.
RESULT_VIEW_CAP = 2000


@dataclass(slots=True)
class ToolSpec:
    """Mirrors tool_module's ToolSpec; Task 3 owns the real one."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    readonly: bool = False
    requires_approval: bool = False

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema or {"type": "object", "properties": {}},
            },
        }


@dataclass(slots=True)
class ResultEnvelope:
    """Mirrors tool_module's envelope; dispatch returns this and never raises."""

    ok: bool
    content: str
    error_kind: str | None = None
    retryable: bool = False
    ref: str | None = None


@dataclass(slots=True)
class Budgets:
    max_hops: int = 15
    per_tool_attempts: int = 3
    wall_clock_s: float = 300.0
    # Consecutive hop re-attempts after the client exhausts its own retries.
    model_retries: int = 3


Dispatch = Callable[[str, dict[str, Any]], Awaitable[ResultEnvelope]]


@dataclass(slots=True)
class _PartialCall:
    """Tool-call fragments accumulate here, keyed by the stream's index."""

    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass(slots=True)
class _State:
    """Carried across hops within one turn."""

    budgets: Budgets
    # Consecutive failures per tool. Reset by a success, so a tool that works
    # again is not punished for an earlier bad patch.
    failures: dict[str, int] = field(default_factory=dict)
    # One malformed-args round trip per turn is free, so a repair does not eat
    # a hop the work needs. 0a measured zero malformed calls, so this is a
    # safety net, not a hot path.
    repair_available: bool = True
    repair_pending: bool = False


async def run_turn(
    messages: list[dict[str, Any]],
    tools: Sequence[ToolSpec],
    budgets: Budgets,
    mode: Mode,
    *,
    dispatch: Dispatch,
    hops_used: int = 0,
    source: model_client.Source = "interactive",
    options: dict[str, Any] | None = None,
) -> AsyncIterator[Event]:
    """
    Run one turn to its end, yielding events as they happen.

    `messages` is mutated in place, so a caller that persists it sees the same
    history the model saw.

    Args:
        messages: OpenAI-shape history, built by the fold.
        tools: the manifest for this session.
        budgets: hop, attempt and wall-clock caps.
        mode: attended turns may end on bare text; unattended may not.
        dispatch: executes one tool and returns an envelope.
        hops_used: hops already spent, counted from the log across a resume.
        source: passed to the model client.
        options: per-call model params from config.

    Yields:
        Events from the vocabulary, ending with exactly one `done`.
    """
    by_name = {t.name: t for t in tools}
    schemas = [t.to_openai() for t in tools]
    started = time.monotonic()
    state = _State(budgets=budgets)
    nudged = False
    model_retries = 0

    while True:
        if hops_used >= budgets.max_hops:
            yield DoneEvent(reason="max_hops")
            return
        if time.monotonic() - started >= budgets.wall_clock_s:
            yield DoneEvent(reason="wall_clock")
            return

        # A repair round trip is not charged: the previous hop produced no
        # usable work, only malformed arguments.
        if state.repair_pending:
            state.repair_pending = False
        else:
            hops_used += 1
            yield BudgetEvent(hops_used=hops_used, hops_max=budgets.max_hops)

        try:
            hop = _Hop()
            async for delta in model_client.generate(messages, schemas, source=source, options=options):
                event = hop.absorb(delta)
                if event is not None:
                    yield event
        except ModelError as e:
            # Two bounded layers: the client retried within its own call, this
            # re-attempts the hop, and neither nests unbounded.
            if e.retryable and model_retries < budgets.model_retries:
                model_retries += 1
                hops_used -= 1
                logger.warning("hop failed (%s), re-attempting %d/%d", e.kind, model_retries, budgets.model_retries)
                continue
            logger.error("model error ends the run: %s", e)
            yield DoneEvent(reason="model_error")
            return

        model_retries = 0
        calls = hop.finish()

        if not calls:
            messages.append({"role": "assistant", "content": hop.text})
            if mode == "attended":
                # The human is the continuation, so stopping is always safe.
                yield DoneEvent(reason="turn_end")
                return
            # Unattended: bare text is not an exit. One nudge near the cap, then
            # keep going until a budget stops us.
            if not nudged and hops_used >= budgets.max_hops - 1:
                nudged = True
                nudge = (
                    f"You have {budgets.max_hops - hops_used} hop(s) left and have not called "
                    f"{FINISH_TOOL}. Finish the work and call {FINISH_TOOL}, or explain what blocked you."
                )
                messages.append({"role": "user", "content": nudge})
                yield UserEvent(text=nudge, source="system")
            continue

        messages.append(
            {
                "role": "assistant",
                "content": hop.text or None,
                "tool_calls": [
                    {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                    for c in calls
                ],
            }
        )

        finished = False
        for call in calls:
            async for event in _run_one_call(call, by_name, dispatch, state, messages):
                yield event
            if call.name == FINISH_TOOL and call.name in by_name:
                finished = True

        if finished and mode == "unattended":
            yield DoneEvent(reason="completed")
            return


class _Hop:
    """Accumulates one streamed completion. Text is yielded as it arrives."""

    def __init__(self) -> None:
        self.text = ""
        self._calls: dict[int, _PartialCall] = {}

    def absorb(self, delta: model_client.Delta) -> Event | None:
        """Map one delta to at most one event. Tool-call fragments buffer instead."""
        if isinstance(delta, model_client.TextDelta):
            self.text += delta.text
            return ContentEvent(text=delta.text)
        if isinstance(delta, model_client.ReasoningDelta):
            # Streamed, never folded back into messages.
            return ReasoningEvent(text=delta.text)
        if isinstance(delta, model_client.ToolCallDelta):
            partial = self._calls.setdefault(delta.index, _PartialCall())
            if delta.id:
                partial.id = delta.id
            if delta.name:
                partial.name = delta.name
            partial.arguments += delta.arguments
        return None

    def finish(self) -> list[_PartialCall]:
        return [self._calls[i] for i in sorted(self._calls)]


async def _run_one_call(
    call: _PartialCall,
    by_name: dict[str, ToolSpec],
    dispatch: Dispatch,
    state: _State,
    messages: list[dict[str, Any]],
) -> AsyncIterator[Event]:
    """Validate, execute and report one call. Never raises except cancellation."""
    args, parse_error = _parse_args(call.arguments)

    if parse_error is not None:
        yield ToolCallEvent(id=call.id, name=call.name, args={})
        if state.repair_available:
            state.repair_available = False
            state.repair_pending = True
        yield _close(call, "invalid_args", parse_error, messages)
        return

    yield ToolCallEvent(id=call.id, name=call.name, args=args)

    if call.name not in by_name:
        known = ", ".join(sorted(by_name)) or "none"
        yield _close(call, "not_found", f"No tool named {call.name!r}. Available: {known}", messages)
        return

    cap = state.budgets.per_tool_attempts
    if state.failures.get(call.name, 0) >= cap:
        yield _close(
            call,
            "upstream_error",
            f"{call.name} has failed {cap} times. Do not call it again; try another approach.",
            messages,
        )
        return

    envelope = await dispatch(call.name, args)
    if envelope.ok:
        state.failures.pop(call.name, None)
    else:
        state.failures[call.name] = state.failures.get(call.name, 0) + 1

    content, total = _cap_view(envelope.content)
    messages.append({"role": "tool", "tool_call_id": call.id, "content": envelope.content})
    yield ToolResultEvent(
        id=call.id,
        ok=envelope.ok,
        content=content,
        error_kind=envelope.error_kind,
        total_chars=total,
        ref=envelope.ref,
    )


def _close(call: _PartialCall, error_kind: str, message: str, messages: list[dict[str, Any]]) -> ToolResultEvent:
    """Close a tool_call with a failure the model can read and act on."""
    messages.append({"role": "tool", "tool_call_id": call.id, "content": message})
    return ToolResultEvent(id=call.id, ok=False, content=message, error_kind=error_kind)


def _parse_args(raw: str) -> tuple[dict[str, Any], str | None]:
    """Parse tool arguments. Returns (args, error_written_for_the_model)."""
    if not raw.strip():
        return {}, None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        return {}, f"Arguments were not valid JSON ({e}). Call the tool again with valid JSON."
    if not isinstance(parsed, dict):
        return {}, f"Arguments must be a JSON object, got {type(parsed).__name__}."
    return parsed, None


def _cap_view(content: str) -> tuple[str, int | None]:
    """View-cap the result; the caller stores the full text and sets `ref`."""
    if len(content) <= RESULT_VIEW_CAP:
        return content, None
    return content[:RESULT_VIEW_CAP], len(content)
