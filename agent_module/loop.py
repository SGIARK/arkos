"""
The agent loop: one `run_turn`. Never touches Postgres, and is the only caller
of the model. Tool errors are model input, not control flow: a failed tool comes
back as a tool_result the model can read.
"""

from __future__ import annotations

import asyncio
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
from config_module.loader import config
from model_module import client as model_client
from model_module.errors import ModelError
from tool_module.envelope import ResultEnvelope, ToolSpec

logger = logging.getLogger(__name__)

Mode = Literal["attended", "unattended"]
Profile = Literal["interactive", "worker"]

# The control tool that makes an unattended exit safe.
FINISH_TOOL = "finish_task"


def _cfg(key: str, default: Any) -> Any:
    value = config.get(key)
    return default if value is None else value


@dataclass(slots=True)
class Budgets:
    max_hops: int
    wall_clock_s: float
    per_tool_attempts: int
    model_retries: int

    @classmethod
    def load(cls, profile: Profile = "interactive") -> Budgets:
        """Load the budgets for one profile from config."""
        fallback = {"interactive": (6, 300.0), "worker": (15, 1800.0)}[profile]
        return cls(
            max_hops=int(_cfg(f"budgets.{profile}.max_hops", fallback[0])),
            wall_clock_s=float(_cfg(f"budgets.{profile}.wall_clock_s", fallback[1])),
            per_tool_attempts=int(_cfg("budgets.per_tool_attempts", 3)),
            model_retries=int(_cfg("budgets.model_retries", 3)),
        )


Dispatch = Callable[[str, dict[str, Any]], Awaitable[ResultEnvelope]]


@dataclass(slots=True)
class _PartialCall:
    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass(slots=True)
class _State:
    """Carried across hops within one turn."""

    budgets: Budgets
    # Consecutive failures; a success clears the streak.
    failures: dict[str, int] = field(default_factory=dict)
    # Counts dispatches in flight, so a parallel fan-out cannot outrun the cap.
    in_flight: dict[str, int] = field(default_factory=dict)
    # One malformed-args round trip per turn is not charged as a hop.
    repair_available: bool = True
    repair_pending: bool = False
    # Set when finish_task actually returns ok, never from the call alone.
    finished: bool = False


async def run_turn(
    messages: list[dict[str, Any]],
    tools: Sequence[ToolSpec],
    budgets: Budgets,
    mode: Mode,
    *,
    dispatch: Dispatch,
    hops_used: int = 0,
    options: dict[str, Any] | None = None,
) -> AsyncIterator[Event]:
    """
    Run one turn to its end, yielding events as they happen.

    `messages` is mutated in place.

    Args:
        messages: OpenAI-shape history, built by the fold.
        tools: the manifest for this session.
        budgets: hop, attempt and wall-clock caps, from `Budgets.load`.
        mode: attended turns may end on bare text; unattended may not.
        dispatch: executes one tool and returns an envelope.
        hops_used: hops already spent, counted from the log across a resume.
        options: per-call model params from config.

    Yields:
        Events from the vocabulary, ending with exactly one `done`.
    """
    # Unattended runs yield the GPU slot on overload instead of queueing for it.
    source: model_client.Source = "background" if mode == "unattended" else "interactive"
    by_name = {t.name: t for t in tools}
    schemas = [t.to_openai() for t in tools]
    deadline = time.monotonic() + budgets.wall_clock_s
    state = _State(budgets=budgets)
    seen_ids: set[str] = set()
    nudged = False
    model_retries = 0
    reattempt = False

    while True:
        if hops_used >= budgets.max_hops:
            yield DoneEvent(reason="max_hops")
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            yield DoneEvent(reason="wall_clock")
            return

        # A repair round trip or a model re-attempt is the same hop trying again.
        charged = not state.repair_pending and not reattempt
        state.repair_pending = False
        reattempt = False
        if charged:
            hops_used += 1
            yield BudgetEvent(hops_used=hops_used, hops_max=budgets.max_hops)

        hop = _Hop()
        try:
            # Bounds the whole hop, model plus tools, not just the gap between hops.
            async with asyncio.timeout(remaining):
                async for delta in model_client.generate(messages, schemas, source=source, options=options):
                    event = hop.absorb(delta)
                    if event is not None:
                        yield event

                calls = hop.finish(seen_ids)

                if hop.truncated and not calls:
                    messages.append({"role": "assistant", "content": hop.text})
                    yield DoneEvent(reason="context_overflow")
                    return

                if not calls:
                    if hop.text:
                        messages.append({"role": "assistant", "content": hop.text})
                    if mode == "attended":
                        # The human is the continuation, so stopping is safe.
                        yield DoneEvent(reason="turn_end")
                        return
                    if not hop.text:
                        # Nothing to act on; looping would only spend budget.
                        yield DoneEvent(reason="model_error")
                        return
                    if not nudged and hops_used == budgets.max_hops - 1:
                        nudged = True
                        nudge = (
                            f"You have 1 hop left and have not called {FINISH_TOOL}. "
                            f"Finish the work and call {FINISH_TOOL}, or explain what blocked you."
                        )
                        messages.append({"role": "user", "content": nudge})
                        yield UserEvent(text=nudge, source="system")
                    model_retries = 0
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

                for batch in _batch_by_readonly(calls, by_name):
                    async for event in _run_batch(batch, by_name, dispatch, state, messages):
                        yield event

        except TimeoutError:
            yield DoneEvent(reason="wall_clock")
            return
        except ModelError as e:
            if e.retryable and model_retries < budgets.model_retries:
                model_retries += 1
                reattempt = True
                logger.warning("hop failed (%s), re-attempting %d/%d", e.kind, model_retries, budgets.model_retries)
                continue
            logger.error("model error ends the run: %s", e)
            yield DoneEvent(reason="model_error")
            return
        except asyncio.CancelledError:
            # Record why the run stopped before the cancellation propagates.
            yield DoneEvent(reason="cancelled")
            raise

        model_retries = 0
        if state.finished and mode == "unattended":
            yield DoneEvent(reason="completed")
            return


class _Hop:
    """Accumulate one streamed completion."""

    def __init__(self) -> None:
        self.text = ""
        self.truncated = False
        self._calls: dict[int, _PartialCall] = {}
        self._next_synthetic = 0

    def absorb(self, delta: model_client.Delta) -> Event | None:
        """Map one delta to at most one event; tool-call fragments buffer instead."""
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
        elif isinstance(delta, model_client.Finish):
            # "length" means max_tokens truncated the reply.
            self.truncated = delta.reason == "length"
        return None

    def finish(self, seen_ids: set[str]) -> list[_PartialCall]:
        """Return the calls in order, replacing empty or reused ids so each pairs with one result."""
        calls = [self._calls[i] for i in sorted(self._calls)]
        for call in calls:
            if not call.id or call.id in seen_ids:
                self._next_synthetic += 1
                call.id = f"call_{len(seen_ids)}_{self._next_synthetic}"
            seen_ids.add(call.id)
        return calls


def _batch_by_readonly(calls: list[_PartialCall], by_name: dict[str, ToolSpec]) -> list[list[_PartialCall]]:
    """Group consecutive readonly calls; anything else stands alone."""
    batches: list[list[_PartialCall]] = []
    open_batch = False
    for call in calls:
        spec = by_name.get(call.name)
        readonly = spec is not None and spec.readonly
        if readonly and open_batch:
            batches[-1].append(call)
        else:
            batches.append([call])
        open_batch = readonly
    return batches


async def _run_batch(
    batch: list[_PartialCall],
    by_name: dict[str, ToolSpec],
    dispatch: Dispatch,
    state: _State,
    messages: list[dict[str, Any]],
) -> AsyncIterator[Event]:
    """
    Validate the batch, then dispatch what survives concurrently.

    Results are yielded as they land, not in call order. Every call is closed by
    exactly one result before this returns, on every exit path.
    """
    runnable: list[tuple[_PartialCall, dict[str, Any]]] = []

    for call in batch:
        args, parse_error = _parse_args(call.arguments)
        if parse_error is not None:
            yield ToolCallEvent(id=call.id, name=call.name, args={})
            if state.repair_available:
                state.repair_available = False
                state.repair_pending = True
            yield _close(call, "invalid_args", parse_error, messages)
            continue

        yield ToolCallEvent(id=call.id, name=call.name, args=args)

        if call.name not in by_name:
            known = ", ".join(sorted(by_name)) or "none"
            yield _close(call, "not_found", f"No tool named {call.name!r}. Available: {known}", messages)
            continue

        cap = state.budgets.per_tool_attempts
        # in_flight counts this batch, so parallel calls cannot all dispatch past the cap.
        if state.failures.get(call.name, 0) + state.in_flight.get(call.name, 0) >= cap:
            yield _close(
                call,
                "upstream_error",
                f"{call.name} is at its {cap}-attempt cap. Do not call it again; try another approach.",
                messages,
            )
            continue

        state.in_flight[call.name] = state.in_flight.get(call.name, 0) + 1
        runnable.append((call, args))

    if not runnable:
        return

    # asyncio.wait, not as_completed: the task object is what maps a result to its call.
    tasks = {asyncio.create_task(dispatch(c.name, a)): c for c, a in runnable}
    pending = set(tasks)
    try:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                call = tasks[task]
                yield _settle(call, _envelope_of(task, call), state, messages)
    except (asyncio.CancelledError, GeneratorExit):
        # Close every open call, or the session cannot be resumed.
        for task in pending:
            task.cancel()
        for task in pending:
            call = tasks[task]
            state.in_flight[call.name] = max(0, state.in_flight.get(call.name, 1) - 1)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": _INTERRUPTED})
        raise


_INTERRUPTED = "Interrupted before this returned. The outcome is unknown; verify before retrying."


def _envelope_of(task: asyncio.Task[ResultEnvelope], call: _PartialCall) -> ResultEnvelope:
    """Return the task's envelope; a dispatch that raises is a failed tool, not a failed run."""
    try:
        return task.result()
    except asyncio.CancelledError:
        return ResultEnvelope(ok=False, content=_INTERRUPTED, error_kind="interrupted")
    except Exception as e:
        logger.exception("dispatch raised for %s", call.name)
        return ResultEnvelope(ok=False, content=f"{call.name} failed: {e}", error_kind="upstream_error")


def _settle(
    call: _PartialCall,
    envelope: ResultEnvelope,
    state: _State,
    messages: list[dict[str, Any]],
) -> ToolResultEvent:
    state.in_flight[call.name] = max(0, state.in_flight.get(call.name, 1) - 1)
    if envelope.ok:
        state.failures.pop(call.name, None)
        if call.name == FINISH_TOOL:
            state.finished = True
    else:
        state.failures[call.name] = state.failures.get(call.name, 0) + 1

    content, total = _cap_view(envelope.content)
    messages.append({"role": "tool", "tool_call_id": call.id, "content": envelope.content})
    return ToolResultEvent(
        id=call.id,
        ok=envelope.ok,
        content=content,
        error_kind=envelope.error_kind if envelope.error_kind != "none" else None,
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
    """Truncate the result to the view cap, returning the original length when it was cut."""
    cap = int(_cfg("tools.result_view_cap_chars", 4000))
    if len(content) <= cap:
        return content, None
    return content[:cap], len(content)
