"""Drives one turn of a session."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import posixpath
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
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
    TodoEvent,
    TodoTracker,
    ToolCallEvent,
    ToolResultEvent,
    UserEvent,
    ViewTransformEvent,
)
from agent_module.loop import Budgets, Dispatch, cap_view, run_turn
from cancellation import run_to_completion
from config_module.loader import cfg as _cfg
from config_module.loader import config
from db import pool
from db.ids import as_uuid as _uuid
from harness_module import approvals, hands, leases, lifecycle, memory, store, system_log, workspace
from harness_module import session_log as slog
from harness_module.stream import stream
from tool_module import registry
from tool_module.envelope import ResultEnvelope, ToolContext, ToolSpec, ToolUnavailable
from tool_module.sandbox import manager as sandbox_manager
from tool_module.tools.control import PARK_KINDS

logger = logging.getLogger(__name__)


# The prefix the transcript renders as an AUTO badge rather than as status prose;
# the frontend carries the same literal.
_AUTO_BADGE = "auto-approved "


def _destructive(name: str) -> bool:
    """Whether a tool is one autopilot refuses to answer for itself; absent from config means auto-answerable."""
    named = {str(n) for n in (_cfg("tools.destructive", []) or [])}
    # The gate sees the name the MODEL sees, which for a connector tool carries the
    # registry's `mcp_` prefix; config names tools as the vendor does.
    bare = name[len(registry.MCP_PREFIX) :] if name.startswith(registry.MCP_PREFIX) else name
    return name in named or bare in named


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


# Field names ARE the column names, so the SELECT cannot drift from the dataclass.
_SESSION_COLUMNS = ", ".join(Session.__dataclass_fields__)


# The live turn per session. At most one; a second start() is a no-op.
_running: dict[str, asyncio.Task[None]] = {}

# How a signalled turn should LAND: `stopped` or `cancelled`. Set by the endpoint
# before the task is cancelled, read where the ending is recorded.
_teardown: dict[str, str] = {}

# Background terminal retries, held so they are not garbage collected.
_reapers: set[asyncio.Task[None]] = set()


async def load(session_id: str) -> Session | None:
    """Returns the session, or None if there is no such row."""
    row = await pool.fetchrow(
        f"SELECT {_SESSION_COLUMNS} FROM sessions WHERE id = $1",
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
    # The last event this view contains; steering reads from here, so a message
    # that landed between the fold and the first hop is carried, not skipped.
    last_seq: int = 0
    # The checklist the log ends on; the terminal sweep needs it across a resume,
    # where the runner's own copy starts empty.
    todo: list[dict[str, Any]] = field(default_factory=list)


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

    The output is a function of (log, config, mode, memory, reach, mounts, now); `now`
    defaults to reading one ONCE, so a caller comparing two folds must pass both the same
    instant. `reach` is this turn's manifest, so the caller builds it before folding.
    """
    now = now or datetime.now(UTC)
    events = await _all_events(session.id)
    todo = next(
        (list(e.event.items) for e in reversed(events) if isinstance(e.event, TodoEvent)),
        [],
    )
    core = _capped_memory(await memory.read_memory(session.user_id))
    mounts = await workspace.claims_for(session.id)
    messages, hops_used = _assemble(session, events, frozenset(), core, reach, now, mounts)
    last_seq = events[-1].seq if events else 0

    # Rung 0 measures the view; rung 1 clears the oldest results holding a blob ref
    # until it fits. A result with no ref stays, since nothing can read it back.
    budget = _input_budget()
    threshold = float(_cfg("context.recovery_threshold", 0.8))
    ceiling = int(budget * threshold)
    if ceiling <= 0 or _estimate_tokens(messages) <= ceiling:
        return Folded(messages, hops_used, last_seq=last_seq, todo=todo)

    cleared: list[str] = []
    for ref in _clearable_refs(events):
        cleared.append(ref)
        messages, hops_used = _assemble(session, events, frozenset(cleared), core, reach, now, mounts)
        if _estimate_tokens(messages) <= ceiling:
            break

    if not cleared:
        logger.warning("session %s: the view is over budget and nothing holds a ref to clear", session.id)
        return Folded(messages, hops_used, last_seq=last_seq, todo=todo)

    if _estimate_tokens(messages) > ceiling:
        # Rung 1 clears results and nothing else, so a view dominated by the system
        # prompt and the conversation stays over budget and comes back context_overflow.
        logger.warning(
            "session %s: cleared every stored result and the view is still over budget",
            session.id,
        )
    logger.info("session %s: cleared %d result(s) from the view", session.id, len(cleared))
    return Folded(messages, hops_used, ViewTransformEvent(rung=1, dropped_refs=cleared), last_seq=last_seq, todo=todo)


def _capped_memory(core: str) -> str:
    """Cut the memory document to what the system prompt will carry, leaving a marker."""
    limit = int(_cfg("memory.prompt_max_chars", 4000))
    if limit <= 0 or len(core) <= limit:
        return core
    return core[:limit].rstrip() + "\n\n[...truncated. Call read_memory for the whole document.]"


def _steering(session_id: str, after_seq: int) -> Callable[[], Awaitable[list[str]]]:
    """Hand the loop whatever the human has said since the last hop.

    Delivery, never interruption: the message waits for the current hop to finish. Reads
    from the fold's last seq and advances past everything seen, so nothing is delivered
    twice; human `user` events only.
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
    return [e.event.ref for e in events if isinstance(e.event, ToolResultEvent) and e.event.ref]


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
            # The chat template rejects a `tool` message that follows a `user` one, so a
            # message typed mid-call is held until the open calls close. The log keeps its order.
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
            messages.append({"role": "tool", "tool_call_id": event.id, "content": _stamped(body, stored.ts)})
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
    """Prefix a rendered result with when it was fetched; presentation only, the stored event is untouched.

    The stamp is ABSOLUTE, not an age: an age would rewrite every result on every fold and
    break the cached prefix.
    """
    return f"[fetched {prompts.clock(when)}]\n{body}"


def _dumps(args: dict[str, Any]) -> str:
    return json.dumps(args, default=str)


# --- driving a turn ------------------------------------------------------------


async def stop(session_id: str) -> bool:
    """Hold a running turn without ending it: `done{stopped}`, `running -> idle`, mode KEPT.

    Same path as cancel (`task.cancel()` on the turn), differing only in where it lands.
    Immediate: there is no hop boundary to reach and no grace timer.

    Returns:
        False when no turn of this session is running in this process.
    """
    return await _teardown_turn(session_id, "stopped")


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
    """End a run for good: a live turn is signalled and awaited, otherwise `cancelled` is written directly."""
    if await _teardown_turn(session_id, "cancelled"):
        return True

    session = await load(session_id)
    if session is None or session.status in lifecycle.TERMINAL:
        return False
    # `_ending` appends the done{cancelled} the fold needs to reset the hop count, and
    # hands an unattended session's mode back so it stops holding a worker slot.
    return await _ending(
        session_id,
        None,
        "cancelled",
        expected=session.status,
        mode="attended" if session.mode == "unattended" else None,
    )


async def _teardown_turn(session_id: str, intent: str) -> bool:
    """Signal the live turn and wait for it to land as `intent`.

    Returns:
        False when there is no live turn here — the caller decides what that
        means. Stop has nothing to hold; cancel writes the terminal directly.
    """
    task = _running.get(session_id)
    if task is None or task.done():
        return False
    if session_id not in _teardown:
        # First press wins the landing: a cancel after a stop does not overwrite `stopped`.
        _teardown[session_id] = intent
        task.cancel()
    # asyncio.wait reports the task's completion without re-raising its
    # CancelledError in this caller.
    await asyncio.wait({task})
    return True


async def _drive(session_id: str) -> None:
    """Runs one turn to its end. Every exit path writes a terminal, including a failure during setup."""
    sink: _Sink | None = None
    try:
        session = await load(session_id)
        if session is None:
            return
        sink = _Sink(session)

        # The manifest comes BEFORE the fold: the system prompt names the services
        # this request carries, and it cannot do that until the request is decided.
        shipped = await _manifest_for(session)
        _announce_benching(sink, shipped)

        dispatch = sink.write_ahead(
            registry.bind(sink.tool_context(), mcp_call=_mcp_call(), tools=shipped.specs),
            shipped.specs,
        )

        # A call this session parked on is settled BEFORE anything closes it as dangling.
        await _settle_gated_call(session, sink, dispatch)

        # Close any call the last run left open: the chat template rejects a tool_call
        # id with no matching tool message.
        stream.publish_all(session_id, await slog.close_dangling(session_id))

        started = time.monotonic()
        folded = await fold(session, shipped.servers)
        messages, hops_used = folded.messages, folded.hops_used
        # Seeded so a resumed run can still sweep: `emit` has seen nothing yet.
        sink._todo.items = list(folded.todo)
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
            # The list the log ends on, so a RESUMED run's system prompt states the
            # checklist it has rather than claiming it has none.
            todo_items=folded.todo,
            # The loop cannot tell a stop from a cancel — both arrive as a
            # CancelledError — so it asks what the presser recorded.
            teardown_intent=lambda: _teardown.get(session_id),
        ):
            if isinstance(event, DoneEvent) and sink.parked:
                # The run ended in the same hop that raised a question: the terminal
                # wins and no question is recorded.
                logger.info("session %s ended before its question was recorded", session_id)
                sink.drop_park()
                await sink.close(event)
                return
            if sink.parked and isinstance(event, BudgetEvent):
                # A hop boundary: every call of the parking hop has closed, so the
                # transcript folds cleanly. The loop stops before the next model call.
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
        await sink.close()
    except asyncio.CancelledError:
        # With no recorded intent — a cancellation from elsewhere, a process coming
        # down — it lands as a cancel, which is the safe reading.
        await _shielded(_ending(session_id, sink, _teardown.get(session_id, "cancelled")))
        raise
    except Exception:
        # Not `model_error`: nothing on this path is the model.
        logger.exception("session %s: the turn failed outside the loop", session_id)
        await _shielded(_ending(session_id, sink, "internal_error"))
    finally:
        _teardown.pop(session_id, None)


async def _settle_gated_call(session: Session, sink: _Sink, dispatch: Dispatch) -> bool:
    """Close a call the session parked on, with the human's decision.

    The call is still OPEN in the log — that is what the approvals row is bound to — so it
    is settled here, before `close_dangling` would abandon it as interrupted.

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
        sink.emit(ToolResultEvent(id=row.tool_call_id, ok=False, content=slog.INTERRUPTED, error_kind="interrupted"))
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
        # Another wake got there first and is running it; the call is theirs to close.
        logger.info("session %s: gated call %s claimed by another wake", session.id, row.tool_call_id)
        return False

    envelope = await dispatch_granted(sink, dispatch, row.tool_name, row.tool_args or {})
    sink.emit(_result_event(row.tool_call_id, envelope))
    await sink.barrier()
    return True


async def dispatch_granted(sink: _Sink, dispatch: Dispatch, name: str, args: dict[str, Any]) -> ResultEnvelope:
    """Run one approved call through NORMAL dispatch, with a grant that answers the gate exactly once."""
    sink._grant_once = True
    try:
        return await dispatch(name, args)
    finally:
        sink._grant_once = False


def _result_event(call_id: str, envelope: ResultEnvelope) -> ToolResultEvent:
    """Build the result event for a call settled outside the loop, capped by `loop.cap_view`."""
    content, total = cap_view(envelope.content)
    return ToolResultEvent(
        id=call_id,
        ok=envelope.ok,
        content=content,
        error_kind=envelope.error_kind if envelope.error_kind != "none" else None,
        total_chars=total,
        ref=envelope.ref,
    )


async def _manifest_for(session: Session) -> registry.Manifest:
    """Build the turn's tool list, degrading to ours alone rather than failing the turn."""
    try:
        return await registry.manifest(session.user_id, mcp=hands.connectors(), session_id=session.id)
    except Exception:
        logger.exception("session %s: building the full manifest failed", session.id)
        return await registry.manifest(session.user_id)


def _announce_benching(sink: _Sink, shipped: registry.Manifest) -> None:
    """Say, in the transcript and in the operational log, that a server was left out."""
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
    """Record an ending to completion however often this turn is cancelled; a cancel still ends the turn."""
    task = asyncio.ensure_future(work)
    try:
        await run_to_completion(task)
    finally:
        if task.done() and not task.cancelled() and task.exception() is not None:
            logger.error("recording the end of the run failed", exc_info=task.exception())


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
        done = DoneEvent(reason=reason)
        if mode is None and done.is_terminal():
            # Every direct-write terminal hands an unattended mode back, so it stops
            # holding a quota slot. Not for `stopped`: it is not terminal and keeps the mode.
            current = await pool.fetchval("SELECT mode FROM sessions WHERE id = $1", _uuid(session_id))
            mode = "attended" if current == "unattended" else None
        # ONE mapping from reason to status, the same one the sink uses.
        status = lifecycle.status_for(done)
        # The invariant refuses a `done` while a call is open.
        stream.publish_all(session_id, await slog.close_dangling(session_id))
        stored = await slog.append(session_id, done)
        stream.publish(session_id, stored)
        # `transition` publishes its own event; this call wants only whether it moved.
        return await lifecycle.transition(session_id, expected, status, reason, mode=mode) is not None
    except Exception:
        logger.exception("session %s: could not record the %s ending", session_id, reason)
        return False


# The `error_kind` the gate raises to mean "this call is parked, not failed". It never
# reaches the log: `emit` suppresses the result, which leaves the tool_call open.
_GATED = "approval_required"


def _park_prompt(name: str, args: dict[str, Any]) -> str:
    """Returns the text shown to the human, taken from the park tool's arguments."""
    if name == "ask":
        return str(args.get("question") or "").strip() or "(no question given)"
    if name == "propose_plan":
        # The prompt is the one-line summary; the card reads the plan itself off `tool_args`.
        return str(args.get("goal") or "").strip() or "(no goal given)"
    action = str(args.get("action") or "").strip() or "(no action given)"
    detail = str(args.get("detail") or "").strip()
    return f"{action}\n\n{detail}" if detail else action


# --- the approved plan ---------------------------------------------------------

# An approved plan lands at the root of the session's FIRST linked folder: the run's
# first act is to read it, and the model is told that path.
PLAN_NAME = "plan.md"


def plan_markdown(args: dict[str, Any], version: int) -> str:
    """Render an approved plan as the file the run starts from: a pure function of the approved args."""
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
    """The folder an approved plan is written into: the session's first LINKED writable claim, or None."""
    claims = await workspace.claims_for(session_id)
    writable = [c for c in claims if c.mode == "write"]
    return writable[0].folder if writable else None


async def read_plan(session_id: str) -> str | None:
    """`plan.md`'s content for this session, read from the STORE, or None when no run has happened here."""
    row = await pool.fetchrow("SELECT user_id FROM sessions WHERE id = $1", _uuid(session_id))
    folder = await plan_folder(session_id)
    if row is None or folder is None:
        return None
    entries = await store.read_tree(str(row["user_id"]), f"{folder}/{PLAN_NAME}")
    if not entries:
        return None
    blob = await store.get_blob(entries[0].content_hash)
    return blob.decode(errors="replace") if blob else None


async def save_plan(session_id: str, args: dict[str, Any], version: int) -> str | None:
    """Write an approved plan into the session's first linked folder, and return its path.

    Through the store, not the sandbox: a parked session's box is hibernated, and the next
    materialize copies the file in. Returns None when the session links no folder to write into.
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
    """Adapts the shared connector client to the registry's mcp_call shape, or None if unconfigured."""
    client = hands.connectors()
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

    `emit` queues an event and returns; a writer task performs the appends, and consecutive
    text events still queued are written as a single row. `write_ahead` wraps dispatch so a
    non-readonly tool waits until its own `tool_call` event is committed before it runs.
    """

    def __init__(self, session: Session):
        self.session = session
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        # The checklist as the model last wrote it; seeded from the log on resume.
        self._todo = TodoTracker()
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
        # Every tool call of this turn, so a parked one can be bound to its own id and args.
        self._calls: dict[str, tuple[str, dict[str, Any]]] = {}
        # True for exactly one gated call: the one the human approved.
        self._grant_once = False
        # Set when the park left a call open, so `abort` knows to close it.
        self._gated_call: str | None = None
        # Resource keys this session holds, so a second call skips the database.
        self._leases: set[str] = set()
        # Set once this turn holds a slot in the user's sandbox pool; the row, not this
        # flag, is what releasing consults.
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

        Parking raises `_GATED` rather than returning: `emit` suppresses that result so the
        call stays OPEN, and `park()` binds the approvals row to it. Only one call may park
        per hop. An unattended run auto-answers non-destructive calls; destructive ones
        (`tools.destructive`) park in every mode.
        """
        if self._grant_once:
            self._grant_once = False
            return True
        if self.session.mode == "attended" and bool(_cfg("approvals.attended_auto_approve", False)):
            return True
        if self.session.mode == "unattended" and not _destructive(name):
            # Answered through the gate rather than around it, so the transcript and the
            # approvals history read like a manual approval that happened fast.
            await self._auto_answer(name, args)
            return True
        if self._park is not None:
            raise ToolUnavailable(
                "approval_required",
                f"{name} needs the human's approval and another call is already waiting for it. "
                "Wait for that answer; this one has not been asked yet.",
                retryable=False,
            )
        raise ToolUnavailable(_GATED, f"{name} is waiting for the human to approve it.", retryable=False)

    async def _auto_answer(self, name: str, args: dict[str, Any]) -> None:
        """Write and answer one approvals row as the harness, for the audit trail.

        The call is NOT parked, so there is no open call to bind to: the row is created and
        answered in the same breath.
        """
        try:
            row = await approvals.create(
                self.session.id,
                self._gated_call or f"auto-{name}",
                "call",
                _park_prompt(name, args),
                tool_name=name,
                tool_args=args,
            )
            answered = await approvals.answer_auto(row.id, approvals.APPROVE)
            # Driven by the row's `auto_answered`, so the badge in the transcript and the
            # column in the database are the same fact.
            if answered is not None and answered.auto_answered:
                self.emit(StatusEvent(label=f"{_AUTO_BADGE}{name}"))
        except Exception:  # noqa: BLE001 - the audit trail is not worth the run
            logger.warning("session %s: could not record the auto-approval of %s", self.session.id, name, exc_info=True)

    def drop_park(self) -> None:
        """Discards a pending park. The question is never written."""
        self._park = None

    @property
    def parked(self) -> bool:
        """True once a park tool has returned a result."""
        return self._park is not None

    async def _lease(self, resource: str) -> None:
        """Claim what the session needs to use a shared resource, and fill its cache.

        The sandbox is capacity rather than a lease: the wait is for a free slot in the
        user's pool. Each write claim leases the FOLDER it names, and the claimed folders
        are materialized once the box and the leases are held.

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

        A wait is not a park: the session stays `running`.

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

        `keep_box` is the park: the box is hibernated rather than destroyed, so work outside
        the claimed mounts survives the wait.
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

        Reached only after `_flush_workspace` returns, so the cache is never destroyed while
        it holds the only copy of an edit. The row is the authority, not this object.
        """
        try:
            await sandbox_manager.manager().reap(self.session.id)
        except Exception:  # noqa: BLE001 - a box outliving its run is not worth failing a terminal for
            logger.exception("session %s: reaping the sandbox failed", self.session.id)
        self._sandbox_slot = False

    async def _flush_workspace(self) -> None:
        """Write the sandbox's changes back to the store before the box goes.

        A failure re-raises and nothing is given up: the edits are still on the sandbox disk,
        and reaping the box or releasing the leases would lose them.
        """
        if self._workspace is None:
            return
        claims, manifest = self._workspace
        try:
            flushed = await workspace.flush(sandbox_manager.manager(), self.session.id, claims, manifest)
        except Exception as e:  # noqa: BLE001 - recorded, retried by the reaper
            logger.exception("session %s: flushing the workspace failed", self.session.id)
            system_log.record("flush_failed", level="error", session_id=self.session.id, error=type(e).__name__)
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
            names = ", ".join(posixpath.basename(p) for p in flushed.discarded[:5])
            more = f" and {len(flushed.discarded) - 5} more" if len(flushed.discarded) > 5 else ""
            self.emit(StatusEvent(label=f"discarded edits to read-only files: {names}{more}"))

    def emit(self, event: Event) -> None:
        """Queues one event for the writer. Never blocks.

        One exception: the result of a call the gate parked on is DROPPED rather than queued,
        because a queued result would close a call that has to stay open.
        """
        if isinstance(event, BudgetEvent):
            self._hops = event.hops_used
        elif isinstance(event, ToolCallEvent):
            self._calls[event.id] = (event.name, event.args)
        elif isinstance(event, ToolResultEvent):
            if event.error_kind == _GATED and self._park is None:
                name, args = self._calls.get(event.id, ("", {}))
                self._park = (event.id, name, args)
                self._gated_call = event.id
                logger.info("session %s: parking on gated call %s (%s)", self.session.id, event.id, name)
                return  # dropped on purpose: the call stays open across the park
            name, args = self._calls.get(event.id, ("", {}))
            self._todo.saw_call(event.id, name, args)
            if self._todo.saw_result(event.id, bool(event.ok)):
                self._queue.put_nowait(TodoEvent(items=self._todo.items))
            if event.ok and name in PARK_KINDS:
                # A park tool's own result: the call is closed, and THIS is the
                # moment the session parks.
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
        """Wraps dispatch so a non-readonly tool waits for its `tool_call` to be committed."""
        readonly = {t.name: t.readonly for t in tools}

        async def guarded(name: str, args: dict[str, Any]) -> ResultEnvelope:
            # An unknown name counts as non-readonly.
            if not readonly.get(name, False):
                await self.barrier()
            return await dispatch(name, args)

        return guarded

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
        interrupted finish can be retried without a second `done` row.
        """
        if self._terminal_written:
            return
        # A partly written finish keeps its reason, so the `done` on the transcript and
        # the status the session lands in agree.
        done = self._pending_done or done
        self._pending_done = done
        try:
            if not self._done_appended:
                # Before the drain: a status event queued after the writer stops is a
                # status event nobody sees. A STOP hibernates the box rather than reaping
                # it; leases go either way, since a session that is not acting holds none.
                await self._release_leases(keep_box=done.reason == "stopped")
                await self._sweep_checklist(done)
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
            # An unattended run that reaches a TERMINAL hands the session back attended;
            # a stop reaches `idle` instead and keeps the mode.
            mode = "attended" if new_status in lifecycle.TERMINAL and self.session.mode == "unattended" else None
            await lifecycle.transition(self.session.id, "running", new_status, done.reason, mode=mode)
            self._terminal_written = True
        except Exception:
            self._reap_later(done)
            raise

    async def _sweep_checklist(self, done: DoneEvent) -> None:
        """On a COMPLETED run, resolve the checklist. On any other ending, leave it.

        Queued rather than written directly, so it lands before the terminal and after the work.
        """
        if done.reason != "completed":
            return
        if self._todo.resolve():
            self.emit(TodoEvent(items=self._todo.items))

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
        """Suspends the session on its open question. No `done` is appended: the run is not over.

        Returns:
            True if the status moved to `awaiting_approval`.
        """
        if self._park is None:
            return False
        # A parked session holds no lease. Released before the drain, so anything the flush
        # reports is still recorded; the box is kept, hibernated, for the resuming turn.
        await self._release_leases(keep_box=True)
        await self._drain()
        call_id, name, args = self._park
        kind = PARK_KINDS.get(name)
        if kind == "plan":
            # Each proposal is a version and only the newest is live; superseded before the
            # insert, so the two are never open at once.
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
            # A gated call: the row carries the call itself, and the call is still open in
            # the log for the answer to close.
            approval = await approvals.create(
                self.session.id,
                call_id,
                "call",
                f"Run {name}?",
                tool_name=name,
                tool_args=args,
            )
        else:
            approval = await approvals.create(self.session.id, call_id, PARK_KINDS[name], _park_prompt(name, args))
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
