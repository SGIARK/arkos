"""The agent loop: one `run_turn`."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from agent_module import prompts
from agent_module.events import (
    TODO_TOOL,
    BudgetEvent,
    ContentEvent,
    DoneEvent,
    Event,
    ReasoningEvent,
    StatusEvent,
    TodoTracker,
    ToolCallEvent,
    ToolResultEvent,
    UserEvent,
)
from config_module.loader import cfg as _cfg
from config_module.loader import config
from model_module import client as model_client
from model_module.errors import ModelError
from tool_module.envelope import ResultEnvelope, ToolSpec

logger = logging.getLogger(__name__)

Mode = Literal["attended", "unattended"]

# The only tool that ends an unattended run.
FINISH_TOOL = "finish_task"


def _require(key: str) -> Any:
    """Read a config value that has no default in code."""
    value = config.get(key)
    if value is None:
        raise RuntimeError(f"config is missing {key!r}, and budgets have no defaults in code")
    return value


@dataclass(slots=True)
class Budgets:
    max_hops: int
    wall_clock_s: float
    per_tool_attempts: int
    model_retries: int
    # Consecutive UNPRODUCTIVE hops an unattended run may take before it is called
    # stalled: a continuation, then the finish nudge, then the run ends.
    stall_streak: int

    @classmethod
    def load(cls, mode: Mode = "attended") -> Budgets:
        """Load the budgets for one session mode from config.

        Raises:
            RuntimeError: when a key is missing.
        """
        return cls(
            max_hops=int(_require(f"budgets.{mode}.max_hops")),
            wall_clock_s=float(_require(f"budgets.{mode}.wall_clock_s")),
            per_tool_attempts=int(_require("budgets.per_tool_attempts")),
            model_retries=int(_require("budgets.model_retries")),
            stall_streak=int(_require("budgets.stall_streak")),
        )


Dispatch = Callable[[str, dict[str, Any]], Awaitable[ResultEnvelope]]
StoreBlob = Callable[[str], Awaitable[str]]
# What the human has said since this was last called.
Steer = Callable[[], Awaitable[list[str]]]

# What the teardown cancelling this turn recorded that it MEANT: `"stopped"`,
# `"cancelled"`, or None when nothing recorded an intent.
TeardownIntent = Callable[[], str | None]


@dataclass(slots=True)
class _PartialCall:
    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass(slots=True)
class _State:
    """Carried across hops within one turn."""

    budgets: Budgets
    # Consecutive failures per tool; a success clears the streak.
    failures: dict[str, int] = field(default_factory=dict)
    # Dispatches in flight per tool, so a parallel fan-out cannot outrun the cap.
    in_flight: dict[str, int] = field(default_factory=dict)
    # One malformed-args round trip per turn is not charged as a hop.
    repair_available: bool = True
    repair_pending: bool = False
    # Set when finish_task returns ok, not when it is called.
    finished: bool = False
    # Consecutive hops that produced no work, unattended only. Bare text is one;
    # so is a hop whose only calls were bookkeeping.
    unproductive_streak: int = 0
    # Its own latch: fires once near the hop cap, on a schedule and a budget
    # separate from the streak.
    near_cap_nudged: bool = False


def _unproductive(state: _State, budgets: Budgets, hops_used: int, opener: str) -> tuple[bool, str | None]:
    """Charge one hop that produced no work, and decide what it draws.

    Returns `(ended, injected)`: `ended` means the streak is spent and the run is
    over; `injected` is the nudge to append, or None when no hop is left to act on
    one. `opener` is what the FIRST hop of a streak says, which differs by what the
    hop did: nothing at all, or nothing but bookkeeping.
    """
    state.unproductive_streak += 1
    if state.unproductive_streak >= budgets.stall_streak:
        return True, None
    if hops_used >= budgets.max_hops:
        # Only nudge while a hop remains to act on the nudge.
        return False, None
    near_cap = not state.near_cap_nudged and hops_used == budgets.max_hops - 1
    if state.unproductive_streak > 1 or near_cap:
        # Only the near-cap latch moves here; the streak escalates on its own schedule.
        state.near_cap_nudged = state.near_cap_nudged or near_cap
        return False, prompts.finish_nudge(FINISH_TOOL, budgets.max_hops - hops_used)
    return False, opener


def _changes_the_world(name: str | None, by_name: dict[str, ToolSpec]) -> bool:
    """Whether a successful call did something the checklist could be behind on.

    A read is not one, and neither is writing the list itself.
    """
    if name is None or name == TODO_TOOL:
        return False
    spec = by_name.get(name)
    return spec is None or not spec.readonly


async def run_turn(
    messages: list[dict[str, Any]],
    tools: Sequence[ToolSpec],
    budgets: Budgets,
    mode: Mode,
    *,
    dispatch: Dispatch,
    hops_used: int = 0,
    options: dict[str, Any] | None = None,
    store_blob: StoreBlob | None = None,
    steer: Steer | None = None,
    teardown_intent: TeardownIntent | None = None,
    todo_items: Sequence[dict[str, Any]] = (),
) -> AsyncIterator[Event]:
    """Run one turn to its end, yielding events as they happen.

    `messages` is OpenAI-shape history and is mutated in place.

    Args:
        mode: attended turns may end on bare text; unattended may not.
        hops_used: hops already spent, counted from the log across a resume.
        store_blob: stores the full text of an oversized result and returns its ref.
        steer: anything the human has said since it was last asked; called once per hop.
        todo_items: the checklist the log ends on, so a resumed run states the list it
            actually has rather than claiming it has none.

    Yields:
        Events from the vocabulary, ending with exactly one `done`.
    """
    source: model_client.Source = "background" if mode == "unattended" else "interactive"
    by_name = {t.name: t for t in tools}
    schemas = [t.to_openai() for t in tools]
    deadline = time.monotonic() + budgets.wall_clock_s
    state = _State(budgets=budgets)
    seen_ids: set[str] = set()
    model_retries = 0
    reattempt = False
    todo = TodoTracker()
    todo.items = [dict(item) for item in todo_items]
    # The checklist is STANDING STATE, so it rides the system message and is
    # rewritten there each hop: one copy, and never the newest thing in the
    # context. Restating it as the last message every hop would teach a run to
    # answer it every hop (12.3.9).
    system = messages[0] if messages and messages[0].get("role") == "system" else None
    if mode == "unattended" and system is None:
        logger.warning("unattended turn with no system message: the checklist has nowhere to ride")
    system_base = str(system.get("content") or "") if system is not None else ""
    # Set when work landed and the list did not move; cleared by the write it draws.
    list_stale = False
    reminded = False

    while True:
        if hops_used >= budgets.max_hops:
            yield DoneEvent(reason="max_hops")
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            yield DoneEvent(reason="wall_clock")
            return

        # A repair round trip or a model re-attempt continues the current hop.
        charged = not state.repair_pending and not reattempt
        state.repair_pending = False
        reattempt = False
        if charged:
            hops_used += 1
            yield BudgetEvent(hops_used=hops_used, hops_max=budgets.max_hops)

        # Injected here and nowhere else: at the top of a hop the previous hop's
        # tool results are already in `messages`, the ordering the API requires.
        # Not an event: the endpoint that accepted the message already logged it.
        if steer is not None:
            for said in await steer():
                messages.append({"role": "user", "content": said})

        if mode == "unattended":
            # Replaced in place, never appended: one copy of the list, carried where
            # standing state belongs.
            if system is not None:
                system["content"] = system_base + prompts.checklist_state(todo.items)
            if list_stale and not reminded:
                # The ONE time a run is told to write the list: it is actually behind.
                # Into the CONTEXT and not the log: derived state, refolded from the
                # log next turn, not something that happened.
                reminded = True
                messages.append({"role": "user", "content": prompts.checklist_stale_nudge()})

        hop = _Hop()
        try:
            # Bounds the whole hop, model plus tools.
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
                        yield DoneEvent(reason="turn_end")
                        return
                    if not hop.text:
                        # Nothing to continue from, and not a model_error:
                        # nothing errored, the model said nothing.
                        yield DoneEvent(reason="stalled_progress")
                        return

                    ended, injected = _unproductive(state, budgets, hops_used, prompts.continue_nudge(FINISH_TOOL))
                    if ended:
                        yield DoneEvent(reason="stalled_progress")
                        return
                    if injected is not None:
                        messages.append({"role": "user", "content": injected})
                        yield UserEvent(text=injected, source="system")
                    model_retries = 0
                    continue

                # Bookkeeping is not work. A hop whose only calls were `todo_write`
                # does NOT clear the streak: clearing it on ANY call would let a run
                # spend hops rewriting one list and read as progress (12.3.9).
                worked = any(call.name != TODO_TOOL for call in calls)
                if worked:
                    state.unproductive_streak = 0

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

                names: dict[str, str] = {}
                wrote_list = False
                did_work = False
                for batch in _batch_by_readonly(calls, by_name):
                    async for event in _run_batch(batch, by_name, dispatch, state, messages, store_blob):
                        if isinstance(event, ToolCallEvent):
                            names[event.id] = event.name
                            todo.saw_call(event.id, event.name, event.args)
                        elif isinstance(event, ToolResultEvent):
                            if todo.saw_result(event.id, event.ok):
                                wrote_list = True
                            elif event.ok and _changes_the_world(names.get(event.id), by_name):
                                did_work = True
                        yield event

                # A hop that wrote the list is current whatever else it did; otherwise
                # work done leaves the list behind until it is written.
                answered_reminder = wrote_list and reminded
                list_stale = False if wrote_list else (list_stale or did_work)
                if wrote_list:
                    reminded = False

                # Writing the list this loop just ASKED for is neither work nor a stall:
                # charging it would scold the run for doing what it was told.
                if mode == "unattended" and not worked and not answered_reminder:
                    ended, injected = _unproductive(state, budgets, hops_used, prompts.bookkeeping_nudge(FINISH_TOOL))
                    if ended:
                        yield DoneEvent(reason="stalled_progress")
                        return
                    if injected is not None:
                        messages.append({"role": "user", "content": injected})
                        yield UserEvent(text=injected, source="system")

        except TimeoutError:
            yield DoneEvent(reason="wall_clock")
            return
        except ModelError as e:
            if e.kind == "context_overflow":
                # A recoverable condition, kept out of the model_error bucket.
                logger.error("hop exceeded the context window: %s", e)
                yield DoneEvent(reason="context_overflow")
                return
            if e.retryable and model_retries < budgets.model_retries:
                model_retries += 1
                reattempt = True
                logger.warning("hop failed (%s), re-attempting %d/%d", e.kind, model_retries, budgets.model_retries)
                continue
            attempts = getattr(e, "attempts", 1)
            logger.error("model error ends the run after %d attempt(s): %s", attempts, e)
            # In the transcript, not only the log: the attempt count is part of
            # the failure the person is reading.
            yield StatusEvent(label=f"model failed after {attempts} attempt(s): {e.kind}")
            yield DoneEvent(reason="model_error")
            return
        except asyncio.CancelledError:
            # A stop and a cancel arrive by the identical mechanism, so the
            # recorded intent decides, not this CancelledError.
            if (teardown_intent() if teardown_intent is not None else None) != "stopped":
                # A stop writes no terminal here: its caller lands one.
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

    def absorb(self, delta: model_client.Delta) -> Event | None:
        """Map one delta to at most one event; tool-call fragments buffer instead."""
        if isinstance(delta, model_client.TextDelta):
            self.text += delta.text
            return ContentEvent(text=delta.text)
        if isinstance(delta, model_client.ReasoningDelta):
            # Streamed, never folded back into messages.
            return ReasoningEvent(text=delta.text)
        if isinstance(delta, model_client.RetryDelta):
            # A status event, not a lifecycle state: the run stays RUNNING while
            # the client waits.
            return StatusEvent(
                label=f"model busy ({delta.kind}) — retry {delta.attempt}/{delta.of}, {delta.delay_s:.1f}s"
            )
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
                # `seen_ids` is per-turn, so the replacement must be unique
                # across every turn of every session.
                call.id = f"call_{uuid.uuid4().hex[:12]}"
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
    store_blob: StoreBlob | None = None,
) -> AsyncIterator[Event]:
    """Validate the batch, then dispatch what survives concurrently.

    Results land out of call order; every call is closed by exactly one result
    on every exit path.
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

    tasks = {asyncio.create_task(dispatch(c.name, a)): c for c, a in runnable}
    pending = set(tasks)
    try:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                call = tasks[task]
                yield await _settle(call, _envelope_of(task, call), state, messages, store_blob)
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


async def _settle(
    call: _PartialCall,
    envelope: ResultEnvelope,
    state: _State,
    messages: list[dict[str, Any]],
    store_blob: StoreBlob | None = None,
) -> ToolResultEvent:
    state.in_flight[call.name] = max(0, state.in_flight.get(call.name, 1) - 1)
    if envelope.ok:
        state.failures.pop(call.name, None)
        if call.name == FINISH_TOOL:
            state.finished = True
    else:
        state.failures[call.name] = state.failures.get(call.name, 0) + 1

    content, total = cap_view(envelope.content)
    ref = envelope.ref
    if total is not None and ref is None and store_blob is not None:
        # The event carries only the preview; the blob holds the rest.
        try:
            ref = await store_blob(envelope.content)
        except Exception:
            logger.exception("could not store the full result of %s", call.name)

    messages.append({"role": "tool", "tool_call_id": call.id, "content": envelope.content})
    return ToolResultEvent(
        id=call.id,
        ok=envelope.ok,
        content=content,
        error_kind=envelope.error_kind if envelope.error_kind != "none" else None,
        total_chars=total,
        ref=ref,
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


def cap_view(content: str) -> tuple[str, int | None]:
    """Truncate the result to the view cap, returning the original length when it was cut.

    Public: the harness settles calls outside the loop too and must apply this same cap.
    """
    cap = int(_cfg("tools.result_view_cap_chars", 4000))
    if len(content) <= cap:
        return content, None
    return content[:cap], len(content)
