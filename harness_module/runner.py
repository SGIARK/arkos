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
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from datetime import datetime
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
from config_module.loader import config
from db import pool
from harness_module import approvals, hands, leases, lifecycle, system_log, workspace
from harness_module import session_log as slog
from harness_module.stream import stream
from tool_module import registry
from tool_module.envelope import ResultEnvelope, ToolContext, ToolSpec, ToolUnavailable
from tool_module.sandbox import manager as sandbox_manager
from tool_module.tools.control import PARK_KINDS

logger = logging.getLogger(__name__)


def _cfg(key: str, default: Any) -> Any:
    value = config.get(key)
    return default if value is None else value


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


async def fold(session: Session) -> Folded:
    """Rebuilds the model's message list from the session's log.

    `user` and `content` events become messages, `tool_call` and `tool_result` become the
    paired assistant and tool messages, `reasoning` is dropped, and the remaining kinds are
    UI-only.

    The output is a function of (log, config, mode): no clock is read, and the date in the
    system prompt comes from `sessions.created_at`.
    """
    events = await _all_events(session.id)
    messages, hops_used = _assemble(session, events, frozenset())

    # Rung 0 measures the view; rung 1 clears the oldest results holding a blob ref
    # until it fits. A result with no ref stays, since nothing can read it back.
    budget = _input_budget()
    threshold = float(_cfg("context.recovery_threshold", 0.8))
    ceiling = int(budget * threshold)
    if ceiling <= 0 or _estimate_tokens(messages) <= ceiling:
        return Folded(messages, hops_used)

    cleared: list[str] = []
    for ref in _clearable_refs(events):
        cleared.append(ref)
        messages, hops_used = _assemble(session, events, frozenset(cleared))
        if _estimate_tokens(messages) <= ceiling:
            break

    if not cleared:
        logger.warning("session %s: the view is over budget and nothing holds a ref to clear", session.id)
        return Folded(messages, hops_used)

    if _estimate_tokens(messages) > ceiling:
        # Rung 1 clears results and nothing else, so a view dominated by the system
        # prompt and the conversation stays over budget and the hop can come back
        # done{context_overflow}.
        logger.warning(
            "session %s: cleared every stored result and the view is still over budget",
            session.id,
        )
    logger.info("session %s: cleared %d result(s) from the view", session.id, len(cleared))
    return Folded(messages, hops_used, ViewTransformEvent(rung=1, dropped_refs=cleared))


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
                goal=session.goal,
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
            messages.append({"role": "tool", "tool_call_id": event.id, "content": body})
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


def _dumps(args: dict[str, Any]) -> str:
    return json.dumps(args, default=str)


# --- driving a turn ------------------------------------------------------------


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
    return await _ending(session_id, None, "cancelled", expected=session.status)


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

        # Close any call the last run left open: the chat template rejects a tool_call
        # id with no matching tool message.
        for closed in await slog.close_dangling(session_id):
            stream.publish(session_id, closed)

        started = time.monotonic()
        folded = await fold(session)
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
        try:
            tools = await registry.manifest(session.user_id, mcp=hands.smithery())
        except Exception:
            # An unreachable MCP server leaves the manifest without its remote tools.
            logger.exception("session %s: building the full manifest failed", session_id)
            tools = await registry.manifest(session.user_id)

        dispatch = sink.write_ahead(
            registry.bind(sink.tool_context(), mcp_call=_mcp_call(), tools=tools),
            tools,
        )
        async for event in run_turn(
            messages,
            tools,
            Budgets.load(session.mode),
            session.mode,
            dispatch=dispatch,
            hops_used=hops_used,
            options=_model_options(),
            store_blob=sink.store_blob,
        ):
            if isinstance(event, DoneEvent) and sink.parked:
                # The run ended in the same hop that raised a question. The
                # terminal wins and no question is recorded: there is nothing
                # left for an answer to affect.
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
        await _shielded(_ending(session_id, sink, "cancelled"))
        raise
    except Exception:
        logger.exception("session %s: the turn failed outside the loop", session_id)
        await _shielded(_ending(session_id, sink, "model_error"))
    finally:
        _cancelling.discard(session_id)


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
        for closed in await slog.close_dangling(session_id):
            stream.publish(session_id, closed)
        stored = await slog.append(session_id, DoneEvent(reason=reason))
        stream.publish(session_id, stored)
        return await lifecycle.transition(session_id, expected, status, reason)
    except Exception:
        logger.exception("session %s: could not record the %s ending", session_id, reason)
        return False


def _park_prompt(name: str, args: dict[str, Any]) -> str:
    """Returns the text shown to the human, taken from the park tool's arguments."""
    if name == "ask":
        return str(args.get("question") or "").strip() or "(no question given)"
    action = str(args.get("action") or "").strip() or "(no action given)"
    detail = str(args.get("detail") or "").strip()
    return f"{action}\n\n{detail}" if detail else action


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
        # Resource keys this session holds, so a second call skips the database.
        self._leases: set[str] = set()
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
            emit_status=lambda label: self.emit(StatusEvent(label=label)),
            store_blob=self.store_blob,
            read_blob=lambda ref, offset, limit: slog.read_blob(ref, offset, limit, user_id=self.session.user_id),
            approve=self._approve,
            lease=self._lease,
        )

    async def _approve(self, name: str, args: dict[str, Any]) -> bool:
        """Answers a `requires_approval` tool.

        Attended sessions approve automatically while `approvals.attended_auto_approve`
        is set; unattended sessions refuse.
        """
        if self.session.mode == "attended" and bool(_cfg("approvals.attended_auto_approve", True)):
            return True
        logger.warning("session %s: %s needs approval and nothing can ask for it yet", self.session.id, name)
        return False

    def drop_park(self) -> None:
        """Discards a pending park. The question is never written."""
        self._park = None

    @property
    def parked(self) -> bool:
        """True once a park tool has returned a result."""
        return self._park is not None

    async def _lease(self, resource: str) -> None:
        """Claim what the session needs to use a shared resource, and fill its cache.

        The sandbox is leased for the box itself; each write claim is leased for
        the project it names, so two sessions touching different projects do not
        wait on each other. The claimed subtrees are materialized once the
        leases are held, and nothing unclaimed is put there.

        Raises:
            ToolUnavailable: a lease did not free up. The model routes around it.
        """
        if resource != "sandbox":
            await self._acquire(leases.key(resource, self.session.user_id), f"the {resource}")
            return
        if self._workspace is not None:
            return

        claims = await workspace.claims_for(self.session.id)
        await self._acquire(leases.key("sandbox", self.session.user_id), "the sandbox")
        for claim in claims:
            if claim.mode == "write":
                await self._acquire(f"project:{claim.project_id}", claim.slug)

        materialized = await workspace.materialize(sandbox_manager.manager(), self.session.user_id, claims)
        self._workspace = (claims, materialized.manifest)
        logger.info(
            "session %s mounted %d file(s) across %d claim(s)",
            self.session.id,
            len(materialized.manifest),
            len(claims),
        )

    async def _acquire(self, resource_key: str, label: str) -> None:
        """Take one lease, waiting and saying so while another session holds it."""
        if resource_key in self._leases:
            return

        ttl = float(_cfg("leases.ttl_s", 900))
        poll = float(_cfg("leases.poll_s", 2))
        deadline = time.monotonic() + float(_cfg("leases.wait_timeout_s", 120))
        waiting_since = time.monotonic()
        announced = False

        while True:
            if await leases.acquire(resource_key, self.session.id, ttl):
                self._leases.add(resource_key)
                if announced:
                    system_log.record(
                        "lease_wait",
                        session_id=self.session.id,
                        resource=resource_key,
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
                    resource=resource_key,
                    waited_ms=round((time.monotonic() - waiting_since) * 1000),
                )
                raise ToolUnavailable(
                    "timeout",
                    f"{label} is in use by another session and did not free up. Try again, or do "
                    "something else first.",
                )
            await asyncio.sleep(poll)

    async def _release_leases(self) -> None:
        """Commit what the sandbox changed, then give up every resource held."""
        await self._flush_workspace()
        if not self._leases:
            return
        with contextlib.suppress(Exception):
            await leases.release_all(self.session.id)
        self._leases.clear()

    async def _flush_workspace(self) -> None:
        """Write the sandbox's changes back to the store before the lease goes.

        A failure here is loud and the leases are kept: the edits are still on
        the sandbox disk, and releasing would let another session materialize
        over them.
        """
        if self._workspace is None:
            return
        claims, manifest = self._workspace
        try:
            flushed = await workspace.flush(sandbox_manager.manager(), self.session.user_id, claims, manifest)
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
        """Queues one event for the writer. Never blocks."""
        if isinstance(event, BudgetEvent):
            self._hops = event.hops_used
        elif isinstance(event, ToolCallEvent) and event.name in PARK_KINDS:
            self._park_calls[event.id] = (event.name, event.args)
        elif isinstance(event, ToolResultEvent) and event.ok and event.id in self._park_calls:
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
                for closed in await slog.close_dangling(self.session.id):
                    stream.publish(self.session.id, closed)
                    self._last_seq = closed.seq
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
        # A parked session is not acting, so it holds nothing. Released before
        # the drain, so anything the flush reports is still recorded.
        await self._release_leases()
        await self._drain()
        call_id, name, args = self._park
        approval = await approvals.create(self.session.id, call_id, PARK_KINDS[name], _park_prompt(name, args))
        await self._save_cursor()
        moved = await lifecycle.transition(self.session.id, "running", "awaiting_approval", name)
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
            await self._finish(DoneEvent(reason="model_error"))

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


def _uuid(value: str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))
