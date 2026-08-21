"""Drives one turn of a session.

Folds the session's event log into a message list, runs `run_turn` over it, and
translates what the loop yields into log appends and status transitions.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import posixpath
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from agent_module import prompts
from agent_module.events import (
    BudgetEvent,
    ContentEvent,
    DoneEvent,
    Event,
    ReasoningEvent,
    StatusEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserEvent,
    ViewTransformEvent,
)
from agent_module.loop import Budgets, Dispatch, run_turn
from config_module.loader import cfg as _cfg
from config_module.loader import config
from db import pool
from db.ids import as_uuid as _uuid
from harness_module import approvals, hands, leases, lifecycle, store, system_log, workspace
from harness_module import session_log as slog
from harness_module.stream import stream
from tool_module import registry
from tool_module.envelope import ResultEnvelope, ToolContext, ToolSpec, ToolUnavailable
from tool_module.sandbox import manager as sandbox_manager
from tool_module.tools.control import PARK_KINDS

logger = logging.getLogger(__name__)




@dataclass(slots=True)
class Session:
    """The session columns a turn needs, read once at the start of the turn."""

    id: str
    user_id: str
    project_id: str | None
    mode: str
    status: str
    goal: str | None
    created_at: datetime
    cursor_seq: int
    hops_used: int


# The live turn per session. At most one; a second start() is a no-op.
_running: dict[str, asyncio.Task[None]] = {}

# Sessions already signalled to stop. A second cancel waits for the turn to end.
_cancelling: set[str] = set()

# Background terminal retries, held so they are not garbage collected.
_reapers: set[asyncio.Task[None]] = set()


async def load(session_id: str) -> Session | None:
    """Returns the session, or None if there is no such row."""
    row = await pool.fetchrow(
        """
        SELECT id, user_id, project_id, mode, status, goal, created_at, cursor_seq, hops_used
          FROM sessions WHERE id = $1
        """,
        _uuid(session_id),
    )
    if row is None:
        return None
    return Session(
        id=str(row["id"]),
        user_id=str(row["user_id"]),
        project_id=str(row["project_id"]) if row["project_id"] else None,
        mode=row["mode"],
        status=row["status"],
        goal=row["goal"],
        created_at=row["created_at"],
        cursor_seq=row["cursor_seq"],
        hops_used=row["hops_used"],
    )


# --- the fold ----------------------------------------------------------------


@dataclass(slots=True)
class Folded:
    """One fold's output: the message list, the hop count behind it, and any transform applied."""

    messages: list[dict[str, Any]]
    hops_used: int
    transform: ViewTransformEvent | None = None
    # The last event this view contains. Steering reads from here, so a message
    # that landed between the fold and the first hop is carried, not skipped.
    last_seq: int = 0


def _cleared_text(ref: str) -> str:
    """Returns the placeholder a cleared result shows in the view; the text stays in result_blobs."""
    return f"[cleared from view to make room. read_result(ref={ref!r}) to re-read it]"


def _input_budget() -> int:
    """Returns the tokens available for the view: the context window less the output reserve."""
    return max(0, int(_cfg("llm.context_window", 0)) - int(_cfg("llm.max_tokens", 0)))


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimates the view's size in tokens, from a character-per-token ratio."""
    chars = 0
    for message in messages:
        chars += len(str(message.get("content") or ""))
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            chars += len(str(function.get("name") or "")) + len(str(function.get("arguments") or ""))
    return int(chars / max(1, int(_cfg("context.chars_per_token", 4))))


async def fold(
    session: Session,
    reach: Sequence[registry.ServerReach] = (),
    *,
    now: datetime | None = None,
) -> Folded:
    """Rebuilds the model's message list from the session's log.

    `user` and `content` events become messages, `tool_call` and `tool_result` become the
    paired assistant and tool messages, `reasoning` is dropped, and the remaining kinds are
    UI-only.

    The output is a function of (log, config, mode, memory, reach, mounts, now) — all arguments,
    which is what keeps it deterministic now that the prompt carries a clock. `now`
    defaults to reading one, ONCE, so a fold is internally consistent; a caller that
    needs two folds to compare must pass the same instant to both.

    `reach` is this turn's manifest, which is why the caller builds it FIRST — the system
    prompt names the services the request actually carries, and a prompt written before
    the manifest could only name the ones somebody asked for.
    """
    now = now or datetime.now(UTC)
    events = await _all_events(session.id)
    memory = _capped_memory(await store.read_memory(session.user_id))
    # The folders this session was given. Read here rather than inside
    # `_assemble` for the same reason memory is: the assembly stays a pure
    # function of what it is handed, and a caller comparing two folds gets the
    # same prompt from the same inputs.
    mounts = await workspace.claims_for(session.id)
    messages, hops_used = _assemble(session, events, frozenset(), memory, reach, now, mounts)
    last_seq = events[-1].seq if events else 0

    # Rung 0 measures the view; rung 1 clears the oldest results holding a blob ref
    # until it fits. A result with no ref stays, since nothing can read it back.
    budget = _input_budget()
    threshold = float(_cfg("context.recovery_threshold", 0.8))
    ceiling = int(budget * threshold)
    if ceiling <= 0 or _estimate_tokens(messages) <= ceiling:
        return Folded(messages, hops_used, last_seq=last_seq)

    cleared: list[str] = []
    for ref in _clearable_refs(events):
        cleared.append(ref)
        messages, hops_used = _assemble(session, events, frozenset(cleared), memory, reach, now, mounts)
        if _estimate_tokens(messages) <= ceiling:
            break

    if not cleared:
        logger.warning("session %s: the view is over budget and nothing holds a ref to clear", session.id)
        return Folded(messages, hops_used, last_seq=last_seq)

    if _estimate_tokens(messages) > ceiling:
        # Rung 1 clears results and nothing else, so a view dominated by the system
        # prompt and the conversation stays over budget and the hop can come back
        # done{context_overflow}.
        logger.warning(
            "session %s: cleared every stored result and the view is still over budget",
            session.id,
        )
    logger.info("session %s: cleared %d result(s) from the view", session.id, len(cleared))
    return Folded(messages, hops_used, ViewTransformEvent(rung=1, dropped_refs=cleared), last_seq=last_seq)


def _capped_memory(core: str) -> str:
    """Cut the memory document to what the system prompt will carry.

    The tail is not lost, it is just not free: `read_memory` returns the whole
    document, and the marker is there so the model knows to ask rather than
    curate from a copy that stops mid-sentence.
    """
    limit = int(_cfg("memory.prompt_max_chars", 4000))
    if limit <= 0 or len(core) <= limit:
        return core
    return core[:limit].rstrip() + "\n\n[...truncated. Call read_memory for the whole document.]"


def _steering(session_id: str, after_seq: int) -> Callable[[], Awaitable[list[str]]]:
    """Hand the loop whatever the human has said since the last hop.

    A message posted while a turn is running is appended and streamed by the
    endpoint — it appears in the transcript immediately — but the turn holds the
    message list the fold built and has no way to see the log. Without this it
    is invisible until the run is over, which is how "clone it onto your
    computer" got watched, logged, and never read.

    Reading from the fold's last seq and advancing past everything seen means a
    message that landed between the fold and the first hop is carried too, and
    nothing is delivered twice. Only what a human typed: the loop already knows
    about its own events, and the nudge it injects is its own business.

    This is delivery, never interruption. The message waits for the current hop
    to finish; stopping a run mid-step is `POST /cancel` (owner, 2026-08-18).
    """
    cursor = after_seq

    async def steer() -> list[str]:
        nonlocal cursor
        try:
            fresh = await slog.get_events(session_id, after_seq=cursor, limit=100)
        except Exception:
            # A read that fails costs this hop's steering, never the run.
            logger.exception("session %s: could not read steering messages", session_id)
            return []

        said: list[str] = []
        for stored in fresh:
            cursor = max(cursor, stored.seq)
            event = stored.event
            if isinstance(event, UserEvent) and event.source == "human":
                said.append(event.text)
        if said:
            logger.info("session %s: carrying %d steering message(s) into the run", session_id, len(said))
        return said

    return steer


def _clearable_refs(events: list[slog.StoredEvent]) -> list[str]:
    """Returns every stored result's ref, oldest first."""
    return [
        e.event.ref
        for e in events
        if isinstance(e.event, ToolResultEvent) and e.event.ref
    ]


def _assemble(
    session: Session,
    events: list[slog.StoredEvent],
    cleared: frozenset[str],
    memory: str = "",
    reach: Sequence[registry.ServerReach] = (),
    now: datetime | None = None,
    mounts: Sequence[workspace.Claim] = (),
) -> tuple[list[dict[str, Any]], int]:
    """Builds the message list from events, with `cleared` refs reduced to a pointer.

    Returns:
        The messages, and the hops spent in the current run (the log after the last
        `done`).
    """
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": prompts.system_prompt(
                session.mode,
                date=session.created_at.date().isoformat(),
                now=prompts.clock(now or datetime.now(UTC)),
                goal=session.goal,
                memory=memory,
                reach=reach,
                mounts=mounts,
            ),
        }
    ]
    hops_used = 0
    pending_text: list[str] = []
    pending_calls: list[dict[str, Any]] = []
    open_calls: set[str] = set()
    deferred_users: list[str] = []

    def flush_assistant() -> None:
        """Closes the assistant message being built, if there is one."""
        if not pending_text and not pending_calls:
            return
        message: dict[str, Any] = {"role": "assistant", "content": "".join(pending_text) or None}
        if pending_calls:
            message["tool_calls"] = list(pending_calls)
        messages.append(message)
        pending_text.clear()
        pending_calls.clear()

    def emit_user(text: str) -> None:
        flush_assistant()
        messages.append({"role": "user", "content": text})

    def drain_deferred() -> None:
        """Emits the user messages held back while tool calls were open."""
        if open_calls:
            return
        for text in deferred_users:
            emit_user(text)
        deferred_users.clear()

    for stored in events:
        event = stored.event
        if isinstance(event, UserEvent):
            # A message typed while a call was in flight sits between the call and its
            # result in the log. The chat template rejects a `tool` message that follows
            # a `user` one, so it is held here until the open calls close. The log keeps
            # its original order.
            if open_calls:
                deferred_users.append(event.text)
            else:
                emit_user(event.text)
        elif isinstance(event, ContentEvent):
            pending_text.append(event.text)
        elif isinstance(event, ToolCallEvent):
            open_calls.add(event.id)
            pending_calls.append(
                {
                    "id": event.id,
                    "type": "function",
                    "function": {"name": event.name, "arguments": _dumps(event.args)},
                }
            )
        elif isinstance(event, ToolResultEvent):
            # Every call of the hop is buffered by now, so each result lands directly
            # after the assistant message carrying its call.
            flush_assistant()
            body = _cleared_text(event.ref) if event.ref and event.ref in cleared else _result_text(event)
            messages.append(
                {"role": "tool", "tool_call_id": event.id, "content": _stamped(body, stored.ts)}
            )
            open_calls.discard(event.id)
            drain_deferred()
        elif isinstance(event, BudgetEvent):
            hops_used = event.hops_used
        elif isinstance(event, DoneEvent):
            flush_assistant()
            open_calls.clear()
            drain_deferred()
            hops_used = 0  # a done ends a run; the next one budgets from zero

    flush_assistant()
    open_calls.clear()
    drain_deferred()
    return messages, hops_used


async def _all_events(session_id: str, page: int = 500) -> list[slog.StoredEvent]:
    """Reads the session's whole log, one page at a time."""
    out: list[slog.StoredEvent] = []
    cursor = 0
    while True:
        batch = await slog.get_events(session_id, after_seq=cursor, limit=page)
        if not batch:
            return out
        out.extend(batch)
        cursor = batch[-1].seq


def _result_text(event: ToolResultEvent) -> str:
    """Returns the stored result as the model sees it, with a pointer to any stored tail."""
    if event.ref and event.total_chars:
        return (
            f"{event.content}\n\n[truncated at {len(event.content)} of {event.total_chars} chars. "
            f"read_result(ref={event.ref!r}) for the rest]"
        )
    return event.content


def _stamped(body: str, when: datetime) -> str:
    """Prefix a rendered result with when it was fetched.

    Presentation only. The stored event is untouched — this is the fold's view
    of it, the same place the ladder replaces a cleared result with a pointer.
    Without it the model reads a week-old inbox as the current one, because
    nothing in a `tool` message says when it was true.

    The stamp is ABSOLUTE, not an age. An age would rewrite every result on
    every fold, which changes the cached prefix each hop for no gain — the model
    has the current time in its system prompt and can subtract.
    """
    return f"[fetched {prompts.clock(when)}]\n{body}"


def _dumps(args: dict[str, Any]) -> str:
    return json.dumps(args, default=str)


# --- driving a turn ------------------------------------------------------------


# The live sink per session, so `stop` can reach the turn that is running. The
# task registry alone is not enough: cancelling the task is the old nuke, and
# what stop needs is the sink's dispatch bookkeeping.
_sinks: dict[str, _Sink] = {}


async def stop(session_id: str) -> bool:
    """Hold a running turn at its next hop boundary, without ending it.

    Not `cancel`. Cancel kills the whole turn, writes `done{cancelled}`, and
    flips the mode back to attended — which spends the plan the run was
    approved from. Stop closes the calls in flight as `cancelled_by_user`,
    refuses any further dispatch this hop, and lets the hop finish; the drive
    loop then parks on a `resume` row with the mode untouched.

    Returns:
        False when no turn of this session is running in this process, which is
        the caller's cue to fall back to `cancel`.
    """
    sink = _sinks.get(session_id)
    if sink is None or not is_running(session_id):
        return False
    cancelled = sink.request_stop()
    logger.info("session %s: stop requested, %d call(s) in flight", session_id, cancelled)
    _arm_stop_backstop(session_id)
    return True


# The grace timers in flight, held so the loop does not collect them mid-wait.
_stop_backstops: set[asyncio.Task[None]] = set()


def _arm_stop_backstop(session_id: str) -> None:
    """Degrade a stop that cannot land into the full cancel it replaced.

    Stop takes effect at a hop boundary, and a hop that never reaches one — a
    wedged tool, a stream that stopped arriving — would leave the button doing
    nothing at all. After the grace the old nuke runs, so the same press still
    kills a runaway run. This is the only path on which stopping spends a plan.
    """
    grace = float(_cfg("harness.stop_grace_s", 45))
    task = asyncio.create_task(_force_stop(session_id, grace), name=f"stopgrace:{session_id}")
    _stop_backstops.add(task)
    task.add_done_callback(_stop_backstops.discard)


async def _force_stop(session_id: str, grace: float) -> None:
    await asyncio.sleep(grace)
    if not is_running(session_id):
        return
    logger.warning("session %s: the stop did not reach a hop boundary in %.0fs; cancelling", session_id, grace)
    system_log.record("stop_degraded", level="warn", session_id=session_id, grace_s=grace)
    await cancel(session_id)


async def start(session_id: str, *, mode: lifecycle.Mode | None = None, reason: str = "woken") -> bool:
    """Moves a session to `running` and drives one turn in the background.

    Args:
        mode: set in the same UPDATE as the status when given; None keeps the session's
            current mode.
        reason: recorded on the lifecycle event.

    Returns:
        False if the session is already running, does not exist, or lost the status race
        to another writer.
    """
    live = _running.get(session_id)
    if live is not None and not live.done():
        # The running turn reads new user events at its next hop.
        return False

    session = await load(session_id)
    if session is None:
        return False
    if session.status == "running":
        # Running with no task in this process means the owning process died. The
        # startup sweep fails those sessions.
        logger.warning("session %s is running with no task in this process", session_id)
        return False
    if not await lifecycle.transition(session_id, session.status, "running", reason, mode=mode):
        return False

    task = asyncio.create_task(_drive(session_id), name=f"turn:{session_id}")
    _running[session_id] = task
    task.add_done_callback(lambda t: _running.pop(session_id, None))
    return True


def is_running(session_id: str) -> bool:
    """Returns True while this process is driving a turn for the session."""
    task = _running.get(session_id)
    return task is not None and not task.done()


async def cancel(session_id: str) -> bool:
    """Stops a session.

    A live turn is signalled and awaited; a session with no turn here is written straight
    to `cancelled`.
    """
    task = _running.get(session_id)
    if task is not None and not task.done():
        if session_id not in _cancelling:
            _cancelling.add(session_id)
            task.cancel()
        # asyncio.wait reports the task's completion without re-raising its
        # CancelledError in this caller.
        await asyncio.wait({task})
        return True

    session = await load(session_id)
    if session is None or session.status in lifecycle.TERMINAL:
        return False
    # `_ending` appends the done{cancelled} the transcript needs: the fold resets the
    # hop count at a `done`, so a restarted session budgets from zero.
    #
    # The mode goes back with it. Cancelling a STOPPED run is the path that made
    # this matter: the hold left the session unattended on purpose, so without
    # this a cancelled one stayed recorded unattended forever — holding a worker
    # slot in the quota, and lying about a session nobody is running.
    return await _ending(
        session_id,
        None,
        "cancelled",
        expected=session.status,
        mode="attended" if session.mode == "unattended" else None,
    )


async def _drive(session_id: str) -> None:
    """Runs one turn to its end.

    Every exit path writes a terminal, including a failure during setup.
    """
    sink: _Sink | None = None
    try:
        session = await load(session_id)
        if session is None:
            return
        sink = _Sink(session)
        _sinks[session_id] = sink

        # The manifest comes BEFORE the fold: the system prompt names the services
        # this request carries, and it cannot do that until the request is decided.
        shipped = await _manifest_for(session)
        _announce_benching(sink, shipped)

        dispatch = sink.write_ahead(
            registry.bind(sink.tool_context(), mcp_call=_mcp_call(), tools=shipped.specs),
            shipped.specs,
        )

        # A call this session parked on is settled BEFORE anything closes it as
        # dangling: it is open on purpose, and it is what the human answered.
        await _settle_gated_call(session, sink, dispatch)

        # Close any call the last run left open: the chat template rejects a tool_call
        # id with no matching tool message.
        stream.publish_all(session_id, await slog.close_dangling(session_id))

        started = time.monotonic()
        folded = await fold(session, shipped.servers)
        messages, hops_used = folded.messages, folded.hops_used
        system_log.record(
            "fold",
            session_id=session_id,
            user_id=session.user_id,
            ms=round((time.monotonic() - started) * 1000),
            messages=len(messages),
            hops_used=hops_used,
            cleared=len(folded.transform.dropped_refs) if folded.transform else 0,
        )
        if folded.transform is not None:
            # The clearing is recorded as an event; the log itself keeps every result.
            sink.emit(folded.transform)
        tools = shipped.specs
        async for event in run_turn(
            messages,
            tools,
            Budgets.load(session.mode),
            session.mode,
            dispatch=dispatch,
            hops_used=hops_used,
            options=_model_options(),
            store_blob=sink.store_blob,
            steer=_steering(session_id, folded.last_seq),
        ):
            if isinstance(event, DoneEvent) and sink.parked:
                # The run ended in the same hop that raised a question. The
                # terminal wins and no question is recorded: there is nothing
                # left for an answer to affect.
                logger.info("session %s ended before its question was recorded", session_id)
                sink.drop_park()
                await sink.close(event)
                return
            if (sink.parked or sink.stopping) and isinstance(event, BudgetEvent):
                # A hop boundary: every call of the parking hop has closed, so the
                # transcript folds cleanly. The loop stops before the next model call.
                # A stop holds here for the same reason, and takes the same exit.
                break
            if isinstance(event, DoneEvent):
                await sink.close(event)
                return
            sink.emit(event)
            if sink.failure is not None:
                # A failed append halts the run, so nothing executes off the record.
                raise RuntimeError("the session log could not be written") from sink.failure
        if sink.parked:
            await sink.park()
            return
        if sink.stopping:
            await sink.park_stopped()
            return
        await sink.close()
    except asyncio.CancelledError:
        await _shielded(_ending(session_id, sink, "cancelled"))
        raise
    except Exception:
        # Not `model_error`: nothing here is the model. A Postgres blip, a
        # sandbox that would not boot and an OpenAI outage were one label until
        # 11.8.5, so the pill could not tell an outage from a bad reply.
        logger.exception("session %s: the turn failed outside the loop", session_id)
        await _shielded(_ending(session_id, sink, "internal_error"))
    finally:
        _sinks.pop(session_id, None)
        _cancelling.discard(session_id)


async def _settle_gated_call(session: Session, sink: _Sink, dispatch: Dispatch) -> bool:
    """Close a call the session parked on, with the human's decision.

    This is where consent turns into execution, and the ordering is the whole
    guarantee. The call is still OPEN in the log — that is what the approvals row
    is bound to — so it is settled here, before `close_dangling` would abandon it
    as interrupted.

    Three outcomes:
      approved   -> claim the latch, run THAT call through normal dispatch, and
                    close it with the real result.
      declined   -> close it with the failure the model already knows how to
                    read, and it routes around rather than retrying.
      re-entered -> the row is already claimed and the call is still open, so a
                    previous wake died between claiming and appending. The tool
                    may well have run. Close it as interrupted and never repeat
                    it: sending a message twice is worse than not knowing whether
                    it sent once.

    Returns True when it settled something.
    """
    row = await approvals.grantable(session.id)
    if row is None or row.tool_name is None:
        return False
    if row.tool_call_id not in await slog.open_calls(session.id):
        # Already settled by an earlier wake; nothing is owed.
        return False

    if row.consumed_at is not None:
        logger.warning(
            "session %s: gated call %s was claimed but never closed; repairing without re-running it",
            session.id,
            row.tool_call_id,
        )
        sink.emit(
            ToolResultEvent(id=row.tool_call_id, ok=False, content=slog.INTERRUPTED, error_kind="interrupted")
        )
        await sink.barrier()
        return True

    if not row.approved:
        sink.emit(
            ToolResultEvent(
                id=row.tool_call_id,
                ok=False,
                content=f"The human declined {row.tool_name}. Do not retry it; choose another approach.",
                error_kind="upstream_error",
            )
        )
        await sink.barrier()
        return True

    claimed = await approvals.consume(row.id)
    if claimed is None:
        # Another wake got there first and is running it. Leave the call alone:
        # it is theirs to close.
        logger.info("session %s: gated call %s claimed by another wake", session.id, row.tool_call_id)
        return False

    envelope = await dispatch_granted(sink, dispatch, row.tool_name, row.tool_args or {})
    sink.emit(_result_event(row.tool_call_id, envelope))
    await sink.barrier()
    return True


async def dispatch_granted(sink: _Sink, dispatch: Dispatch, name: str, args: dict[str, Any]) -> ResultEnvelope:
    """Run one approved call through NORMAL dispatch, with a one-shot grant.

    Normal dispatch, not a side door: the schema check, the timeout, the lease
    and the write-ahead barrier all still apply. The grant only answers the
    approval gate, once — anything the model reaches for afterwards is gated
    again.
    """
    sink._grant_once = True
    try:
        return await dispatch(name, args)
    finally:
        sink._grant_once = False


def _result_event(call_id: str, envelope: ResultEnvelope) -> ToolResultEvent:
    """Build the result event for a call settled outside the loop."""
    cap = int(_cfg("tools.result_view_cap_chars", 4000))
    content = envelope.content
    total = len(content) if len(content) > cap else None
    return ToolResultEvent(
        id=call_id,
        ok=envelope.ok,
        content=content[:cap] if total else content,
        error_kind=envelope.error_kind if envelope.error_kind != "none" else None,
        total_chars=total,
        ref=envelope.ref,
    )


async def _manifest_for(session: Session) -> registry.Manifest:
    """Build the turn's tool list, degrading to ours alone rather than failing the turn.

    An unreachable MCP vendor is a smaller tool list, never a dead session. The
    fallback drops the session id with the MCP source, which is the honest
    degradation: with no way to read what a server offers there is no way to
    know whether it fits, and ours always do.
    """
    try:
        return await registry.manifest(session.user_id, mcp=hands.smithery(), session_id=session.id)
    except Exception:
        logger.exception("session %s: building the full manifest failed", session.id)
        return await registry.manifest(session.user_id)


def _announce_benching(sink: _Sink, shipped: registry.Manifest) -> None:
    """Say, in the transcript and in the operational log, that a server was left out.

    A benched server is the one case where what the human switched on and what
    the model was handed disagree, and it happens for a reason nobody typed —
    a server grew its tool list. Silence here is how 164 tool schemas appeared
    without anyone changing anything, so it gets a line the human can read and a
    row an operator can query.
    """
    benched = shipped.benched
    if not benched:
        return
    names = ", ".join(s.name for s in benched)
    sink.emit(
        StatusEvent(
            label=(
                f"{names} left out this turn: {shipped.used}/{shipped.budget} tool slots are already "
                "in use. Turn something off to make room."
            )
        )
    )
    system_log.record(
        "tools_benched",
        level="warn",
        session_id=sink.session.id,
        user_id=sink.session.user_id,
        benched=[s.label for s in benched],
        shipped=[s.label for s in shipped.servers if s.shipped],
        used=shipped.used,
        budget=shipped.budget,
    )


async def _shielded(work: Awaitable[None]) -> None:
    """Runs `work` to completion even if this task is cancelled again while it runs."""
    task = asyncio.ensure_future(work)
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # A further cancel reaches the shield, not the task, so keep waiting.
            if task.done():
                break
        except Exception:
            logger.exception("recording the end of the run failed")
            break


async def _ending(
    session_id: str,
    sink: _Sink | None,
    reason: str,
    expected: str = "running",
    mode: lifecycle.Mode | None = None,
) -> bool:
    """Records the end of a run: closes open calls, appends the `done`, moves the status.

    `sink` is None when the turn died before the sink was built, and when there is no turn
    at all (a cancel of a pending, idle or parked session).
    """
    if sink is not None:
        return await sink.abort(reason)
    try:
        status = "cancelled" if reason == "cancelled" else "failed"
        # The invariant refuses a `done` while a call is open.
        stream.publish_all(session_id, await slog.close_dangling(session_id))
        stored = await slog.append(session_id, DoneEvent(reason=reason))
        stream.publish(session_id, stored)
        # `transition` publishes its own event; this call wants only whether it moved.
        return await lifecycle.transition(session_id, expected, status, reason, mode=mode) is not None
    except Exception:
        logger.exception("session %s: could not record the %s ending", session_id, reason)
        return False


# What the human sees on the held row, and what the two card actions answer.
_STOP_PROMPT = "Run stopped. Resume the plan?"


def _stopped_envelope(name: str) -> ResultEnvelope:
    """The result a call gets when a human stopped the run.

    Written for the model to READ on resume, because that is what happens to it:
    the closed call plus whatever the human said is the next thing in its
    context. `cancelled_by_user` keeps it out of the per-tool failure streak —
    stopping the browser three times must not close the browser to the run.
    """
    return ResultEnvelope(
        ok=False,
        content=(
            f"The human stopped the run while {name} was running. Its outcome is unknown: "
            "check before assuming it did or did not happen. They may say what to do instead."
        ),
        error_kind="cancelled_by_user",
        retryable=False,
    )


# The `error_kind` the gate raises to mean "this call is parked, not failed".
# It never reaches the log: `emit` recognises it and suppresses the result, which
# is what leaves the tool_call open across the park.
_GATED = "approval_required"


def _park_prompt(name: str, args: dict[str, Any]) -> str:
    """Returns the text shown to the human, taken from the park tool's arguments."""
    if name == "ask":
        return str(args.get("question") or "").strip() or "(no question given)"
    if name == "propose_plan":
        # The prompt is the one-line summary; the card reads the plan itself off
        # `tool_args`, the same way a gated call's card reads the call.
        return str(args.get("goal") or "").strip() or "(no goal given)"
    action = str(args.get("action") or "").strip() or "(no action given)"
    detail = str(args.get("detail") or "").strip()
    return f"{action}\n\n{detail}" if detail else action


# --- the approved plan ---------------------------------------------------------

# What an approved plan is called. It lands at the root of the session's FIRST
# linked folder (11.9), because the run's first act is to read it and the model
# is told that path — and because a project links folders rather than owning
# one, so "the project root" is not a place any more.
PLAN_NAME = "plan.md"


def plan_markdown(args: dict[str, Any], version: int) -> str:
    """Render an approved plan as the file the run starts from.

    A pure function of the args that were approved: the human read those fields
    and nothing else is added, so the file cannot say something the card did
    not. The model is never asked to write this — a second generation would be a
    second chance to disagree with what was consented to.
    """
    lines = [f"# {str(args.get('goal') or '').strip()}", "", f"_plan v{version}_", ""]
    done_when = str(args.get("done_when") or "").strip()
    if done_when:
        lines += ["## done when", "", done_when, ""]
    steps = [str(s).strip() for s in (args.get("steps") or []) if str(s).strip()]
    if steps:
        lines += ["## steps", ""]
        lines += [f"{i}. {step}" for i, step in enumerate(steps, 1)]
        lines.append("")
    inputs = [i for i in (args.get("inputs") or []) if isinstance(i, dict)]
    if inputs:
        lines += ["## inputs", ""]
        for item in inputs:
            label = str(item.get("label") or "").strip()
            note = str(item.get("note") or "").strip()
            lines.append(f"- {label} — {note}" if note else f"- {label}")
        lines.append("")
    missing = [str(m).strip() for m in (args.get("missing") or []) if str(m).strip()]
    if missing:
        lines += ["## still open", ""]
        lines += [f"- {question}" for question in missing]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


async def plan_folder(session_id: str) -> str | None:
    """The folder an approved plan is written into: the session's FIRST claim.

    First LINKED, not first alphabetically: the order a project's folders were
    linked in is the order they were chosen in, and the first is the one the
    work is about. A session that claims nothing has nowhere durable to put a
    plan, and saying so before the approval is what keeps the promise "the run
    starts from plan.md" honest.
    """
    claims = await workspace.claims_for(session_id)
    writable = [c for c in claims if c.mode == "write"]
    return writable[0].folder if writable else None


async def save_plan(session_id: str, args: dict[str, Any], version: int) -> str | None:
    """Write an approved plan into the session's first linked folder, and return its path.

    Through the store, not the sandbox: the plan is approved while the session is
    parked, and a parked session's box is hibernated. The next materialize copies
    the file in like any other, which is what makes "starting from plan.md" true
    rather than a phrase in the prompt.

    Returns None when the session links no folder to write into.
    """
    row = await pool.fetchrow("SELECT user_id FROM sessions WHERE id = $1", _uuid(session_id))
    folder = await plan_folder(session_id)
    if row is None or folder is None:
        logger.warning("session %s: an approved plan has no folder to be saved in", session_id)
        return None
    path = f"{folder}/{PLAN_NAME}"
    await store.put_file(str(row["user_id"]), path, plan_markdown(args, version).encode())
    return path


def _model_options() -> dict[str, Any] | None:
    """Returns the per-call model parameters from config, or None when unset."""
    options = config.get("llm.options")
    return dict(options) if options else None


def _mcp_call():
    """Adapts the shared Smithery client to the registry's mcp_call shape, or None if unconfigured."""
    client = hands.smithery()
    if client is None:
        return None

    async def call(name: str, args: dict[str, Any], ctx: ToolContext):
        return await client.call(name, args, ctx)

    return call


# --- translating events into a record -----------------------------------------

# End-of-queue marker for the writer's queue.
_STOP = object()


class _Barrier:
    """A queue marker the writer resolves once everything ahead of it is written."""

    __slots__ = ("future",)

    def __init__(self, future: asyncio.Future[None]):
        self.future = future


class _Sink:
    """Writes the loop's events to the session log and moves the session at the end of a turn.

    `emit` queues an event and returns; a writer task performs the appends, and
    consecutive text events still queued are written as a single row.

    `write_ahead` wraps dispatch so a non-readonly tool waits, via `barrier()`,
    until its own `tool_call` event is committed before it runs.
    """

    def __init__(self, session: Session):
        self.session = session
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        # asyncio.Queue has no un-get, so a merge that reads one event too many parks
        # it here for the next iteration.
        self._pushback: list[Any] = []
        self._writer = asyncio.create_task(self._write_loop(), name=f"log:{session.id}")
        # Set when an append fails. The drive loop halts the run on it.
        self.failure: BaseException | None = None
        # Separate flags, so a finish interrupted between the two steps can be retried
        # without appending a second `done`.
        self._done_appended = False
        self._terminal_written = False
        self._reaping = False
        # The park tool call whose result arrived, and the arguments it carried. Set on
        # the result, at which point the call is closed in the transcript.
        self._park: tuple[str, str, dict[str, Any]] | None = None
        self._park_calls: dict[str, tuple[str, dict[str, Any]]] = {}
        # Every tool call of this turn, so a parked one can be bound to its own
        # id and args without threading the id through the approval gate.
        self._calls: dict[str, tuple[str, dict[str, Any]]] = {}
        # True for exactly one gated call: the one the human approved.
        self._grant_once = False
        # Dispatches in flight, so Stop can close exactly those and nothing else.
        # The drive task is NOT among them: cancelling that is the old nuke.
        self._dispatching: set[asyncio.Task[Any]] = set()
        # Set by `request_stop`. Read at the hop boundary by the drive loop, and
        # by `guarded` to refuse anything the model asks for after it.
        self._stopping = False
        # Set when the park left a call open, so `abort` knows to close it.
        self._gated_call: str | None = None
        # Resource keys this session holds, so a second call skips the database.
        self._leases: set[str] = set()
        # Set once this turn has a slot in the user's sandbox pool, so a second
        # tool call skips the database. The row is what releasing consults.
        self._sandbox_slot = False
        # The claims materialized into the sandbox, and the tree they came from.
        self._workspace: tuple[list[workspace.Claim], dict[str, str]] | None = None
        # The ending already on the record; a later abort completes this one.
        self._pending_done: DoneEvent | None = None
        self._last_seq = 0
        self._hops = 0

    # --- what tools and the loop see -------------------------------------------

    async def store_blob(self, content: str) -> str:
        """Stores the full text of an oversized result and returns its ref."""
        return await slog.save_blob(self.session.id, content)

    def tool_context(self) -> ToolContext:
        """Builds the context tools receive alongside their arguments."""
        return ToolContext(
            user_id=self.session.user_id,
            session_id=self.session.id,
            emit_status=lambda label, url=None: self.emit(StatusEvent(label=label, url=url)),
            store_blob=self.store_blob,
            read_blob=lambda ref, offset, limit: slog.read_blob(ref, offset, limit, user_id=self.session.user_id),
            approve=self._approve,
            lease=self._lease,
        )

    async def _approve(self, name: str, args: dict[str, Any]) -> bool:
        """Answer a `requires_approval` call, or park the turn on it.

        Three outcomes, and only two of them return.

        A grant issued by the resume path returns True once. It is one-shot on
        purpose: the resumed turn runs exactly the call the human approved, and
        anything the model reaches for afterwards is gated again.

        Otherwise there is no grant, so the turn PARKS on this call. The gate
        cannot park by itself — it runs inside dispatch, which must return an
        envelope — so it raises the marker that `emit` recognises: the loop turns
        it into a result, `emit` suppresses that result so the call stays OPEN in
        the log, and `park()` writes the approvals row bound to that call id
        carrying the real (name, args). The human then approves the thing that
        will run rather than a sentence the model wrote about it.

        The refusal this replaces told the model to call `request_approval` and
        promised "you may call {name} once they agree" — a promise nothing kept,
        because this gate never read the approvals table. It also logged asking
        correctly as `invalid_args`, spending the per-tool failure cap on it.

        Only one call may park per hop: the transcript permits exactly one open
        tool call across a park. A second gated call in the same hop is told so
        and closes normally; it is re-issued after the first is answered.

        `approvals.attended_auto_approve` remains the escape hatch, still OFF by
        default: it turns every gated call into a silent yes.
        """
        if self._grant_once:
            self._grant_once = False
            return True
        if self.session.mode == "attended" and bool(_cfg("approvals.attended_auto_approve", False)):
            return True
        if self._park is not None:
            raise ToolUnavailable(
                "approval_required",
                f"{name} needs the human's approval and another call is already waiting for it. "
                "Wait for that answer; this one has not been asked yet.",
                retryable=False,
            )
        raise ToolUnavailable(_GATED, f"{name} is waiting for the human to approve it.", retryable=False)

    def drop_park(self) -> None:
        """Discards a pending park. The question is never written."""
        self._park = None

    @property
    def parked(self) -> bool:
        """True once a park tool has returned a result."""
        return self._park is not None

    async def _lease(self, resource: str) -> None:
        """Claim what the session needs to use a shared resource, and fill its cache.

        The sandbox is this session's own box, so it is capacity rather than a
        lease: the wait is for a free slot in the user's pool. Each write claim
        leases the FOLDER it names, so two sessions writing different folders do
        not wait on each other even when they belong to the same project, and
        two writing the same folder still serialize even when they do not. The
        claimed folders are materialized once the box and the leases are held,
        and nothing unclaimed is put there.

        Raises:
            ToolUnavailable: a box or a lease did not free up. The model routes
                around it.
        """
        if resource != "sandbox":
            await self._acquire(leases.key(resource, self.session.user_id), f"the {resource}")
            return
        if self._workspace is not None:
            return

        claims = await workspace.claims_for(self.session.id)
        await self._claim_sandbox()
        for claim in claims:
            key = workspace.lease_key(claim)
            if key is not None:
                await self._acquire(key, f"{claim.folder}/")

        materialized = await workspace.materialize(sandbox_manager.manager(), self.session.id, claims)
        self._workspace = (claims, materialized.manifest)
        logger.info(
            "session %s mounted %d file(s) across %d claim(s)",
            self.session.id,
            len(materialized.manifest),
            len(claims),
        )

    async def _claim_sandbox(self) -> None:
        """Take a slot in the user's sandbox pool, waiting and saying so while it is full."""
        if self._sandbox_slot:
            return
        await self._wait_for(
            lambda: sandbox_manager.claim_slot(self.session.id),
            resource="sandbox_pool",
            label="a computer",
            busy=(
                f"No computer was free: this account already runs {sandbox_manager.max_per_user()} at "
                "once. The call never ran, so it is safe to retry later, or do something else first."
            ),
        )
        self._sandbox_slot = True

    async def _acquire(self, resource_key: str, label: str) -> None:
        """Take one lease, waiting and saying so while another session holds it."""
        if resource_key in self._leases:
            return

        ttl = float(_cfg("leases.ttl_s", 900))
        await self._wait_for(
            lambda: leases.acquire(resource_key, self.session.id, ttl),
            resource=resource_key,
            label=label,
            busy=(
                f"{label} is held by another session and did not free up. The call never ran, so it "
                "is safe to retry later, or do something else first."
            ),
        )
        self._leases.add(resource_key)

    async def _wait_for(
        self,
        take: Callable[[], Awaitable[bool]],
        *,
        resource: str,
        label: str,
        busy: str,
    ) -> None:
        """Retry `take` until it succeeds, saying once in the transcript that it is waiting.

        A wait is not a park: the session stays `running` and the wall clock is
        the only thing spent.

        Raises:
            ToolUnavailable: nothing freed up before the timeout. The model
                routes around it.
        """
        poll = float(_cfg("leases.poll_s", 2))
        deadline = time.monotonic() + float(_cfg("leases.wait_timeout_s", 120))
        waiting_since = time.monotonic()
        announced = False

        while True:
            if await take():
                if announced:
                    system_log.record(
                        "lease_wait",
                        session_id=self.session.id,
                        resource=resource,
                        waited_ms=round((time.monotonic() - waiting_since) * 1000),
                    )
                return
            if not announced:
                self.emit(StatusEvent(label=f"waiting for {label}"))
                announced = True
            if time.monotonic() >= deadline:
                system_log.record(
                    "lease_timeout",
                    level="warn",
                    session_id=self.session.id,
                    resource=resource,
                    waited_ms=round((time.monotonic() - waiting_since) * 1000),
                )
                raise ToolUnavailable("timeout", busy)
            await asyncio.sleep(poll)

    async def _release_leases(self, *, keep_box: bool = False) -> None:
        """Commit what the sandbox changed, then give up the box and every resource held.

        `keep_box` is the park: the session is not acting, so it holds no lease,
        but it is not over either, so its box is hibernated rather than
        destroyed and the work outside the claimed mounts survives the wait.
        """
        await self._flush_workspace()
        if keep_box:
            await self._pause_sandbox()
        else:
            await self._release_sandbox()
        if not self._leases:
            return
        with contextlib.suppress(Exception):
            await leases.release_all(self.session.id)
        self._leases.clear()

    async def _pause_sandbox(self) -> None:
        """Hibernate the session's box, keeping its slot for the turn that resumes it."""
        try:
            await sandbox_manager.manager().pause(self.session.id)
        except Exception:  # noqa: BLE001 - a box left running is not worth failing a park for
            logger.exception("session %s: pausing the sandbox failed", self.session.id)

    async def _release_sandbox(self) -> None:
        """Destroy the session's box and free its slot in the user's pool.

        Reached only after `_flush_workspace` returns, and that raises when the
        commit did not land, so the cache is never destroyed while it holds the
        only copy of an edit. The row is the authority, not this object: a turn
        that inherited a slot from a predecessor gives it back too.
        """
        try:
            await sandbox_manager.manager().reap(self.session.id)
        except Exception:  # noqa: BLE001 - a box outliving its run is not worth failing a terminal for
            logger.exception("session %s: reaping the sandbox failed", self.session.id)
        self._sandbox_slot = False

    async def _flush_workspace(self) -> None:
        """Write the sandbox's changes back to the store before the box goes.

        A failure here is loud and nothing is given up: the edits are still on
        the sandbox disk, and reaping the box or releasing the project leases
        would lose them or let another session materialize over them.
        """
        if self._workspace is None:
            return
        claims, manifest = self._workspace
        try:
            flushed = await workspace.flush(sandbox_manager.manager(), self.session.id, claims, manifest)
        except Exception as e:  # noqa: BLE001 - recorded, retried by the reaper
            logger.exception("session %s: flushing the workspace failed", self.session.id)
            system_log.record(
                "flush_failed", level="error", session_id=self.session.id, error=type(e).__name__
            )
            raise
        self._workspace = None
        system_log.record(
            "flush",
            session_id=self.session.id,
            committed=flushed.committed,
            uploaded=flushed.uploaded,
            discarded=len(flushed.discarded),
        )
        if flushed.discarded:
            # The person who watched these edits happen is reading the
            # transcript, not system_events.
            names = ", ".join(posixpath.basename(p) for p in flushed.discarded[:5])
            more = f" and {len(flushed.discarded) - 5} more" if len(flushed.discarded) > 5 else ""
            self.emit(StatusEvent(label=f"discarded edits to read-only files: {names}{more}"))

    def emit(self, event: Event) -> None:
        """Queues one event for the writer. Never blocks.

        With one exception, which is the whole of 11.7: the result of a call the
        gate parked on is DROPPED rather than queued. A queued result closes the
        call, and the call has to stay open — it is what the approval row is
        bound to, and what the resumed turn executes. The loop's own view of the
        turn is about to be abandoned at the hop boundary, so nothing downstream
        reads the result we are discarding.
        """
        if isinstance(event, BudgetEvent):
            self._hops = event.hops_used
        elif isinstance(event, ToolCallEvent):
            self._calls[event.id] = (event.name, event.args)
            if event.name in PARK_KINDS:
                self._park_calls[event.id] = (event.name, event.args)
        elif isinstance(event, ToolResultEvent):
            if event.error_kind == _GATED and self._park is None:
                name, args = self._calls.get(event.id, ("", {}))
                self._park = (event.id, name, args)
                self._gated_call = event.id
                logger.info("session %s: parking on gated call %s (%s)", self.session.id, event.id, name)
                return  # dropped on purpose: the call stays open across the park
            if event.ok and event.id in self._park_calls:
                name, args = self._park_calls[event.id]
                self._park = (event.id, name, args)
        self._queue.put_nowait(event)

    # --- write-ahead for anything that acts --------------------------------------

    async def barrier(self) -> None:
        """Waits until everything queued so far is committed.

        Raises:
            Exception: the error the writer failed on, so a call that cannot be recorded
                does not run.
        """
        if self.failure is not None:
            raise self.failure
        if self._writer.done():
            return
        waiting: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._queue.put_nowait(_Barrier(waiting))
        await waiting

    def write_ahead(self, dispatch: Dispatch, tools: Sequence[ToolSpec]) -> Dispatch:
        """Wraps dispatch so a non-readonly tool waits for its `tool_call` to be committed.

        It is also where a call becomes STOPPABLE. Every dispatch registers its
        own task here on the way in, so Stop can cancel the calls in flight
        without touching the turn — the loop stays pure and knows nothing about
        any of it, because a stopped call comes back as an ordinary envelope.
        """
        readonly = {t.name: t.readonly for t in tools}

        async def guarded(name: str, args: dict[str, Any]) -> ResultEnvelope:
            if self._stopping:
                # Stop refuses the rest of the hop too. A fan-out of five whose
                # first two were cancelled must not have the other three start.
                return _stopped_envelope(name)
            task = asyncio.current_task()
            if task is not None:
                self._dispatching.add(task)
            try:
                # An unknown name counts as non-readonly.
                if not readonly.get(name, False):
                    await self.barrier()
                return await dispatch(name, args)
            except asyncio.CancelledError:
                if not self._stopping:
                    # The whole turn is going down; this is not ours to swallow.
                    raise
                return _stopped_envelope(name)
            finally:
                if task is not None:
                    self._dispatching.discard(task)

        return guarded

    def request_stop(self) -> int:
        """Close the calls in flight and refuse the rest of the hop.

        Returns:
            How many in-flight calls were cancelled. Zero is ordinary — the stop
            may have landed between calls, or mid-generation.
        """
        self._stopping = True
        live = [task for task in self._dispatching if not task.done()]
        for task in live:
            task.cancel()
        return len(live)

    @property
    def stopping(self) -> bool:
        """True once a human has pressed Stop on this turn."""
        return self._stopping

    async def park_stopped(self) -> bool:
        """Hold the run on a `resume` row, with everything about it preserved.

        The same shape as `park`, and deliberately so: leases released, box
        hibernated, `running -> awaiting_approval`. What it does NOT do is the
        part that matters — no `done`, so nothing terminal is written, and no
        mode flip, so the plan this run was approved from is still approved.
        The hop budget carries for free: the fold counts hops from the last
        `done` and a park appends none.
        """
        await self._release_leases(keep_box=True)
        await self._drain()
        # Synthetic: no tool call parks here, and nothing in the log bears this
        # id, so `close_dangling` has nothing to find across the hold.
        call_id = f"stop_{uuid.uuid4().hex[:12]}"
        approval = await approvals.create(self.session.id, call_id, "resume", _STOP_PROMPT)
        await self._save_cursor()
        moved = await lifecycle.transition(self.session.id, "running", "awaiting_approval", "stopped") is not None
        if moved:
            logger.info("session %s stopped at hop %s (%s)", self.session.id, self._hops, approval.id)
        return moved

    # --- the writer -------------------------------------------------------------

    async def _write_loop(self) -> None:
        """Appends queued events in order, merging consecutive text into one row."""
        while True:
            event = self._pushback.pop() if self._pushback else await self._queue.get()
            if event is _STOP:
                self._release_barriers(None)
                return
            if isinstance(event, _Barrier):
                if not event.future.done():
                    event.future.set_result(None)
                continue
            if isinstance(event, (ContentEvent, ReasoningEvent)):
                event = self._merge_text(event)
            try:
                await self._append(event)
            except Exception as e:  # noqa: BLE001 - recorded here, acted on by the drive loop
                logger.exception("session %s: append failed; halting the run", self.session.id)
                self.failure = e
                self._release_barriers(e)
                return

    def _release_barriers(self, error: BaseException | None) -> None:
        """Settles every queued barrier, so nothing waits on a writer that has stopped."""
        pending = self._pushback + [self._queue.get_nowait() for _ in range(self._queue.qsize())]
        self._pushback.clear()
        unwritten = 0
        for item in pending:
            if isinstance(item, _Barrier):
                if item.future.done():
                    continue
                if error is None:
                    item.future.set_result(None)
                else:
                    item.future.set_exception(error)
            elif item is not _STOP:
                unwritten += 1
        if unwritten:
            logger.error("session %s: %d event(s) queued after the writer stopped", self.session.id, unwritten)

    def _merge_text(self, first: ContentEvent | ReasoningEvent) -> Event:
        """Merges the run of same-kind text events already queued, without waiting for more."""
        parts = [first.text]
        while True:
            try:
                nxt = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if type(nxt) is not type(first):
                self._pushback.append(nxt)
                break
            parts.append(nxt.text)
        return type(first)(text="".join(parts))

    async def _append(self, event: Event) -> None:
        stored = await slog.append(self.session.id, event)
        stream.publish(self.session.id, stored)
        self._last_seq = stored.seq

    async def _drain(self) -> None:
        """Stops the writer once everything queued is written."""
        if self._writer.done():
            return
        self._queue.put_nowait(_STOP)
        await self._writer

    # --- ending -----------------------------------------------------------------

    async def _finish(self, done: DoneEvent) -> None:
        """Appends the terminal event and moves the session.

        `_done_appended` guards the append and `_terminal_written` the transition, so an
        interrupted finish can be retried without a second `done` row. The session ends on
        the first reason to reach here.
        """
        if self._terminal_written:
            return
        # A partly written finish keeps its reason, so the `done` on the transcript and
        # the status the session lands in agree.
        done = self._pending_done or done
        self._pending_done = done
        try:
            if not self._done_appended:
                # Before the drain: the flush may have something to say, and a
                # status event queued after the writer stops is a status event
                # nobody sees.
                await self._release_leases()
                await self._drain()
                # The invariant refuses a `done` while a call is open.
                closed = await slog.close_dangling(self.session.id)
                stream.publish_all(self.session.id, closed)
                if closed:
                    self._last_seq = closed[-1].seq
                await self._append(done)
                self._done_appended = True

            await self._save_cursor()
            new_status = lifecycle.status_for(done)
            # An unattended run that reaches a terminal hands the session back attended.
            mode = "attended" if new_status in lifecycle.TERMINAL and self.session.mode == "unattended" else None
            await lifecycle.transition(self.session.id, "running", new_status, done.reason, mode=mode)
            self._terminal_written = True
        except Exception:
            self._reap_later(done)
            raise

    def _reap_later(self, done: DoneEvent) -> None:
        """Retries this terminal in the background until it lands or the attempts run out."""
        if self._reaping:
            return
        self._reaping = True
        task = asyncio.create_task(self._reap(done), name=f"reap:{self.session.id}")
        _reapers.add(task)
        task.add_done_callback(_reapers.discard)

    async def _reap(self, done: DoneEvent) -> None:
        """Calls `_finish` on a doubling backoff, capped at `harness.terminal_retry_max_s`."""
        attempts = int(_cfg("harness.terminal_retry_max", 8))
        base = float(_cfg("harness.terminal_retry_s", 2))
        ceiling = float(_cfg("harness.terminal_retry_max_s", 60))
        for attempt in range(1, attempts + 1):
            await asyncio.sleep(min(base * (2 ** (attempt - 1)), ceiling))
            try:
                # `_reaping` stays set, so a failing `_finish` re-enters `_reap_later`
                # as a no-op and no second reaper starts.
                await self._finish(done)
            except Exception as e:  # noqa: BLE001 - recorded, then retried
                logger.warning("session %s: terminal retry %d/%d failed", self.session.id, attempt, attempts)
                system_log.record(
                    "terminal_retry",
                    level="warn",
                    session_id=self.session.id,
                    attempt=attempt,
                    of=attempts,
                    reason=done.reason,
                    error=type(e).__name__,
                )
                continue
            if self._terminal_written:
                logger.warning("session %s: terminal written on retry %d", self.session.id, attempt)
                return
        logger.error(
            "session %s: could not write done{%s} after %d retries; it stays running until the next sweep",
            self.session.id,
            done.reason,
            attempts,
        )
        system_log.record(
            "terminal_abandoned", level="error", session_id=self.session.id, reason=done.reason, attempts=attempts
        )

    async def park(self) -> bool:
        """Suspends the session on its open question.

        No `done` is appended: the run is not over. The approval row and the
        `awaiting_approval` status carry the wait; nothing is held in memory.

        Returns:
            True if the status moved to `awaiting_approval`.
        """
        if self._park is None:
            return False
        # A parked session is not acting, so it holds no lease. Released before
        # the drain, so anything the flush reports is still recorded; the box is
        # kept, hibernated, because the session resumes into it.
        await self._release_leases(keep_box=True)
        await self._drain()
        call_id, name, args = self._park
        kind = PARK_KINDS.get(name)
        if kind == "plan":
            # Each proposal is a version, and only the newest is live: the older
            # row stays for the card's "changed since v{n-1}" diff but stops
            # waiting on anybody. Superseded before the insert, so the two are
            # never open at once.
            superseded = await approvals.supersede_plans(self.session.id)
            if superseded:
                logger.info("session %s: plan superseded by a newer proposal", self.session.id)
            approval = await approvals.create(
                self.session.id,
                call_id,
                "plan",
                _park_prompt(name, args),
                tool_name=name,
                tool_args=args,
            )
        elif self._gated_call == call_id:
            # A gated call: the row carries the call itself, and the call is
            # still open in the log for the answer to close.
            approval = await approvals.create(
                self.session.id,
                call_id,
                "call",
                f"Run {name}?",
                tool_name=name,
                tool_args=args,
            )
        else:
            approval = await approvals.create(
                self.session.id, call_id, PARK_KINDS[name], _park_prompt(name, args)
            )
        await self._save_cursor()
        moved = await lifecycle.transition(self.session.id, "running", "awaiting_approval", name) is not None
        if moved:
            logger.info("session %s parked on %s (%s)", self.session.id, name, approval.id)
        return moved

    async def close(self, done: DoneEvent | None = None) -> None:
        """Finishes the turn, synthesizing a terminal if the loop ended without one."""
        if done is not None:
            await self._finish(done)
            return
        await self._drain()
        if not self._terminal_written:
            logger.error("session %s: the loop ended with no done event", self.session.id)
            await self._finish(DoneEvent(reason="internal_error"))

    async def abort(self, reason: str) -> bool:
        """Ends the run from outside the loop, on the error path. Failures are logged, not raised.

        Returns:
            True if the terminal reached the database.
        """
        try:
            await self._finish(DoneEvent(reason=reason))
        except Exception:
            logger.exception("session %s: could not record the %s ending", self.session.id, reason)
        return self._terminal_written

    async def _save_cursor(self) -> None:
        """Writes back the cursor and hop count, both caches of what the log holds."""
        with contextlib.suppress(Exception):
            await pool.execute(
                "UPDATE sessions SET cursor_seq = $2, hops_used = $3 WHERE id = $1",
                _uuid(self.session.id),
                self._last_seq,
                self._hops,
            )


