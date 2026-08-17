"""
Drives one turn: fold a session's events into messages, run `run_turn`, and
translate what it yields into log appends and status transitions.

Covers attended sessions. Leases, the approve path, wake-at-cursor resume and
the context ladder are not implemented here.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
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
)
from agent_module.loop import Budgets, Dispatch, run_turn
from config_module.loader import config
from db import pool
from harness_module import hands, lifecycle
from harness_module import session_log as slog
from harness_module.stream import stream
from tool_module import registry
from tool_module.envelope import ResultEnvelope, ToolContext, ToolSpec

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

# Sessions already signalled. A second cancel waits instead of cancelling again.
_cancelling: set[str] = set()

# Background terminal retries, held so they are not garbage collected.
_reapers: set[asyncio.Task[None]] = set()


async def load(session_id: str) -> Session | None:
    """Return the session, or None if there is no such row."""
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


async def fold(session: Session) -> tuple[list[dict[str, Any]], int]:
    """
    Rebuild the model's message list from the session's log.

    Deterministic given (log, config, mode): no clock is read, and the date in
    the system prompt comes from `sessions.created_at`.

    `user` and `content` become messages, `tool_call` and `tool_result` become
    the paired assistant and tool messages, `reasoning` is dropped, and the
    remaining kinds are UI-only.

    Returns:
        The messages, and the hops spent in the current run (the log after the
        last `done`).
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
        """Close the assistant message being built, if there is one."""
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
        """Emit user messages held back while tool calls were open."""
        if open_calls:
            return
        for text in deferred_users:
            emit_user(text)
        deferred_users.clear()

    for stored in await _all_events(session.id):
        event = stored.event
        if isinstance(event, UserEvent):
            # A message typed while a call was in flight sits between the call
            # and its result in the log. Emitting it there would put a `tool`
            # message after a `user` one, which the chat template rejects, so it
            # is held until the results close. View-only: the log is untouched.
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
            # Every call of the hop is buffered by now, so each result lands
            # directly after the assistant message carrying its call.
            flush_assistant()
            messages.append({"role": "tool", "tool_call_id": event.id, "content": _result_text(event)})
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
    """Read the session's whole log, one page at a time."""
    out: list[slog.StoredEvent] = []
    cursor = 0
    while True:
        batch = await slog.get_events(session_id, after_seq=cursor, limit=page)
        if not batch:
            return out
        out.extend(batch)
        cursor = batch[-1].seq


def _result_text(event: ToolResultEvent) -> str:
    """Return the stored result as the model should see it, with a pointer to any stored tail."""
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
    """
    Move a session to `running` and drive one turn in the background.

    Args:
        mode: flipped in the same UPDATE as the status. The approve endpoint
            passes `unattended`; a plain wake passes nothing and keeps the mode
            the session already has.
        reason: recorded on the lifecycle event.

    Returns:
        False if the session is already running, does not exist, or lost the
        status race to another writer.
    """
    live = _running.get(session_id)
    if live is not None and not live.done():
        # The running turn reads new user events at its next hop.
        return False

    session = await load(session_id)
    if session is None:
        return False
    if session.status == "running":
        # Running with no task here means the owning process died. The startup
        # sweep fails those; racing it would double-write the terminal.
        logger.warning("session %s is running with no task in this process", session_id)
        return False
    if not await lifecycle.transition(session_id, session.status, "running", reason, mode=mode):
        return False

    task = asyncio.create_task(_drive(session_id), name=f"turn:{session_id}")
    _running[session_id] = task
    task.add_done_callback(lambda t: _running.pop(session_id, None))
    return True


def is_running(session_id: str) -> bool:
    """Return True if this process is driving a turn for the session right now."""
    task = _running.get(session_id)
    return task is not None and not task.done()


async def cancel(session_id: str) -> bool:
    """Stop a session. A live turn is signalled; anything else is written straight to `cancelled`."""
    task = _running.get(session_id)
    if task is not None and not task.done():
        if session_id not in _cancelling:
            _cancelling.add(session_id)
            task.cancel()
        # asyncio.wait rather than `await task`: it reports completion without
        # re-raising the task's CancelledError at this caller.
        await asyncio.wait({task})
        return True

    session = await load(session_id)
    if session is None or session.status in lifecycle.TERMINAL:
        return False
    # Through the same helper as a live turn, so the transcript gets its
    # done{cancelled}. A bare transition leaves no `done`, and the fold resets
    # hops at a `done`, so a restarted session would inherit the old count and
    # hit max_hops before calling the model.
    return await _ending(session_id, None, "cancelled", expected=session.status)


async def _drive(session_id: str) -> None:
    """Run one turn to its end. Every exit path writes a terminal, setup included."""
    sink: _Sink | None = None
    try:
        session = await load(session_id)
        if session is None:
            return
        sink = _Sink(session)

        # Close any call the last run left open. The chat template rejects a
        # tool_call id with no matching tool message, so an unclosed call makes
        # the session unloadable rather than merely interrupted.
        for closed in await slog.close_dangling(session_id):
            stream.publish(session_id, closed)

        messages, hops_used = await fold(session)
        try:
            tools = await registry.manifest(session.user_id, mcp=hands.smithery())
        except Exception:
            # An unreachable MCP costs the model its remote tools, not its turn.
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
        ):
            if isinstance(event, DoneEvent):
                await sink.close(event)
                return
            sink.emit(event)
            if sink.failure is not None:
                # Nothing executes off the record: a failed append halts the run.
                raise RuntimeError("the session log could not be written") from sink.failure
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
    """Run `work` to completion even if this task is cancelled again while it runs."""
    task = asyncio.ensure_future(work)
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # A further cancel. The shielded task is untouched, so keep waiting.
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
    """
    Record the end of a run: close open calls, append the `done`, move the status.

    `sink` is None when the turn died before it was built, and when there is no
    turn at all — a cancel of a pending, idle or parked session.
    """
    if sink is not None:
        await sink.abort(reason)
        return True
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


def _model_options() -> dict[str, Any] | None:
    """Per-call model params, read from config only."""
    options = config.get("llm.options")
    return dict(options) if options else None


def _mcp_call():
    """Adapt the shared Smithery client to the registry's mcp_call shape, or None if unconfigured."""
    client = hands.smithery()
    if client is None:
        return None

    async def call(name: str, args: dict[str, Any], ctx: ToolContext):
        return await client.call(name, args, ctx)

    return call


# --- translating events into a record -----------------------------------------

# End-of-queue marker. Not None, which is a legal value to find in a queue.
_STOP = object()


class _Barrier:
    """A queue marker the writer resolves once everything ahead of it is written."""

    __slots__ = ("future",)

    def __init__(self, future: asyncio.Future[None]):
        self.future = future


class _Sink:
    """
    Appends what the loop yields, and moves the session when the turn ends.

    `emit` is synchronous: it queues the event for a writer task, so the loop is
    never suspended on a database round trip while the model is streaming. Text
    coalescing follows from that — whatever queues up behind an in-flight append
    is written as one row.

    Mutating tool calls are the exception. `write_ahead` makes them await
    `barrier()`, so a side effect cannot happen before the `tool_call` recording
    its intent is committed. Readonly calls skip the barrier.
    """

    def __init__(self, session: Session):
        self.session = session
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        # asyncio.Queue cannot un-get, so a merge that overshoots parks the
        # event it read here for the next iteration.
        self._pushback: list[Any] = []
        self._writer = asyncio.create_task(self._write_loop(), name=f"log:{session.id}")
        # Set when an append fails. The drive loop halts the run on it.
        self.failure: BaseException | None = None
        # Separate flags so a finish interrupted between the two steps can be
        # retried without appending a second `done`.
        self._done_appended = False
        self._terminal_written = False
        self._reaping = False
        # The ending already on the record. A later abort completes it rather
        # than contradicting it.
        self._pending_done: DoneEvent | None = None
        self._last_seq = 0
        self._hops = 0

    # --- what tools and the loop see -------------------------------------------

    def tool_context(self) -> ToolContext:
        """Build the context tools receive alongside their arguments."""
        return ToolContext(
            user_id=self.session.user_id,
            session_id=self.session.id,
            emit_status=lambda label: self.emit(StatusEvent(label=label)),
            store_blob=lambda content: slog.save_blob(self.session.id, content),
            read_blob=lambda ref, offset, limit: slog.read_blob(ref, offset, limit, user_id=self.session.user_id),
            approve=self._approve,
        )

    async def _approve(self, name: str, args: dict[str, Any]) -> bool:
        """
        Answer a `requires_approval` tool.

        Attended sessions approve automatically: a human is watching the stream
        and can cancel. Unattended sessions have no one to ask, so they refuse.
        """
        if self.session.mode == "attended" and bool(_cfg("approvals.attended_auto_approve", True)):
            return True
        logger.warning("session %s: %s needs approval and nothing can ask for it yet", self.session.id, name)
        return False

    def emit(self, event: Event) -> None:
        """Queue one event for the writer. Never blocks."""
        if isinstance(event, BudgetEvent):
            self._hops = event.hops_used
        self._queue.put_nowait(event)

    # --- write-ahead for anything that acts --------------------------------------

    async def barrier(self) -> None:
        """
        Wait until everything queued so far is committed.

        Raises:
            Exception: whatever the writer failed on, so a caller that cannot be
                recorded does not proceed to act.
        """
        if self.failure is not None:
            raise self.failure
        if self._writer.done():
            return
        waiting: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._queue.put_nowait(_Barrier(waiting))
        await waiting

    def write_ahead(self, dispatch: Dispatch, tools: Sequence[ToolSpec]) -> Dispatch:
        """Wrap dispatch so a mutating tool waits for its `tool_call` to be committed."""
        readonly = {t.name: t.readonly for t in tools}

        async def guarded(name: str, args: dict[str, Any]) -> ResultEnvelope:
            # An unknown name counts as mutating.
            if not readonly.get(name, False):
                await self.barrier()
            return await dispatch(name, args)

        return guarded

    # --- the writer -------------------------------------------------------------

    async def _write_loop(self) -> None:
        """Append queued events in order, merging consecutive text into one row."""
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
        """Settle every queued barrier, so nothing waits on a writer that has stopped."""
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
        """Merge the run of same-kind text events already queued, without waiting for more."""
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
        """Stop the writer once everything queued is written."""
        if self._writer.done():
            return
        self._queue.put_nowait(_STOP)
        await self._writer

    # --- ending -----------------------------------------------------------------

    async def _finish(self, done: DoneEvent) -> None:
        """
        Append the terminal event and move the session.

        Both steps are guarded separately so an interrupted finish can be
        retried: `_done_appended` prevents a second `done` row, and
        `_terminal_written` is set only once the status transition has landed.
        The first reason to reach here is the one the session ends on.
        """
        if self._terminal_written:
            return
        # A partially written finish owns the reason. Without this, an abort
        # arriving after the `done` was appended but before the transition
        # landed would move the session on ITS reason, leaving a transcript
        # saying turn_end next to a status saying failed.
        done = self._pending_done or done
        self._pending_done = done
        try:
            if not self._done_appended:
                await self._drain()
                # The invariant refuses a `done` while a call is open.
                for closed in await slog.close_dangling(self.session.id):
                    stream.publish(self.session.id, closed)
                    self._last_seq = closed.seq
                await self._append(done)
                self._done_appended = True

            await self._save_cursor()
            new_status = lifecycle.status_for(done)
            # An unattended run that ends hands the session back to its human.
            mode = "attended" if new_status in lifecycle.TERMINAL and self.session.mode == "unattended" else None
            await lifecycle.transition(self.session.id, "running", new_status, done.reason, mode=mode)
            self._terminal_written = True
        except Exception:
            self._reap_later(done)
            raise

    def _reap_later(self, done: DoneEvent) -> None:
        """Retry this terminal in the background until it lands or the attempts run out."""
        if self._reaping:
            return
        self._reaping = True
        task = asyncio.create_task(self._reap(done), name=f"reap:{self.session.id}")
        _reapers.add(task)
        task.add_done_callback(_reapers.discard)

    async def _reap(self, done: DoneEvent) -> None:
        attempts = int(_cfg("harness.terminal_retry_max", 8))
        base = float(_cfg("harness.terminal_retry_s", 2))
        ceiling = float(_cfg("harness.terminal_retry_max_s", 60))
        for attempt in range(1, attempts + 1):
            await asyncio.sleep(min(base * (2 ** (attempt - 1)), ceiling))
            try:
                # `_reaping` stays set here, so a failing `_finish` re-enters
                # `_reap_later` as a no-op instead of spawning another reaper.
                await self._finish(done)
            except Exception:
                logger.warning("session %s: terminal retry %d/%d failed", self.session.id, attempt, attempts)
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

    async def close(self, done: DoneEvent | None = None) -> None:
        """Finish the turn, synthesizing a terminal if the loop ended without one."""
        if done is not None:
            await self._finish(done)
            return
        await self._drain()
        if not self._terminal_written:
            logger.error("session %s: the loop ended with no done event", self.session.id)
            await self._finish(DoneEvent(reason="model_error"))

    async def abort(self, reason: str) -> None:
        """End the run from outside the loop. Failures are logged, not raised: this is the error path."""
        try:
            await self._finish(DoneEvent(reason=reason))
        except Exception:
            logger.exception("session %s: could not record the %s ending", self.session.id, reason)

    async def _save_cursor(self) -> None:
        """Write back the cursor and hop count. Both cache what the log already holds."""
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
