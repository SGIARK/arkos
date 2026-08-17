"""
The seam between the log and the loop.

Three jobs and nothing else: fold a session's events back into messages, drive
`run_turn`, and translate what it yields into appends and status transitions.
The loop stays pure — it never learns what a session is — and the harness never
learns what a hop is.

**Scope.** This is the attended half, which is what Task 4 needs to put chat on
the new loop. Leases, the unattended approve path, the wake-at-cursor resume and
the context-recovery ladder are Task 5, and each has a marked seam below.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
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
from agent_module.loop import Budgets, run_turn
from config_module.loader import config
from db import pool
from harness_module import hands, lifecycle
from harness_module import session_log as slog
from harness_module.stream import stream
from tool_module import registry
from tool_module.envelope import ToolContext

logger = logging.getLogger(__name__)


def _cfg(key: str, default: Any) -> Any:
    value = config.get(key)
    return default if value is None else value


@dataclass(slots=True)
class Session:
    """The columns a turn needs. Read once; the loop never re-reads Postgres."""

    id: str
    user_id: str
    project_id: str | None
    mode: str
    status: str
    goal: str | None
    created_at: datetime
    cursor_seq: int
    hops_used: int


# One turn per session, at most. A second `start` while one is live is a no-op
# rather than two loops appending to one transcript.
_running: dict[str, asyncio.Task[None]] = {}


async def load(session_id: str) -> Session | None:
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
    Rebuild the model's view of this session from its log, and count the hops spent.

    Nothing is held in memory between turns, so this runs on every wake. It is
    deterministic given (log, config, mode): no clock is read, and the date in
    the system prompt is the session's own `created_at`.

    Kinds map three ways, which is why there are three of them: `user` and
    `content` become messages, `reasoning` is dropped (replaying a model's
    thinking is both wrong for the chat template and a context tax), and
    `status`/`todo`/`budget`/`lifecycle`/`view_transform` are for the screen only.

    Returns:
        The message list, and hops already spent in the CURRENT run — the log
        after the last `done`. A new turn starts its budget fresh; a resume
        inside one does not.
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

    for stored in await _all_events(session.id):
        event = stored.event
        if isinstance(event, UserEvent):
            flush_assistant()
            messages.append({"role": "user", "content": event.text})
        elif isinstance(event, ContentEvent):
            pending_text.append(event.text)
        elif isinstance(event, ToolCallEvent):
            pending_calls.append(
                {
                    "id": event.id,
                    "type": "function",
                    "function": {"name": event.name, "arguments": _dumps(event.args)},
                }
            )
        elif isinstance(event, ToolResultEvent):
            # Every call of this hop is already buffered, so closing the
            # assistant message here keeps each result adjacent to its call.
            flush_assistant()
            messages.append({"role": "tool", "tool_call_id": event.id, "content": _result_text(event)})
        elif isinstance(event, BudgetEvent):
            hops_used = event.hops_used
        elif isinstance(event, DoneEvent):
            flush_assistant()
            # A `done` ends a run, so the next one budgets from zero.
            hops_used = 0

    flush_assistant()
    return messages, hops_used


async def _all_events(session_id: str, page: int = 500) -> list[slog.StoredEvent]:
    """Read the whole log, paged. Bounding this view is the context ladder's job (Task 5)."""
    out: list[slog.StoredEvent] = []
    cursor = 0
    while True:
        batch = await slog.get_events(session_id, after_seq=cursor, limit=page)
        if not batch:
            return out
        out.extend(batch)
        cursor = batch[-1].seq


def _result_text(event: ToolResultEvent) -> str:
    """
    Return what the model should see for a stored result.

    The event holds the view-capped text, so a resumed run sees the preview
    rather than the full output the live run saw. Saying so is what makes it
    recoverable: the ref is right there and `read_result` pages the rest.
    """
    if event.ref and event.total_chars:
        return (
            f"{event.content}\n\n[truncated at {len(event.content)} of {event.total_chars} chars. "
            f"read_result(ref={event.ref!r}) for the rest]"
        )
    return event.content


def _dumps(args: dict[str, Any]) -> str:
    return json.dumps(args, default=str)


# --- driving a turn ------------------------------------------------------------


async def start(session_id: str) -> bool:
    """
    Move a session to `running` and drive one turn in the background.

    Returns False when it is already running, when it does not exist, or when it
    lost the race to another writer (a cancel, most likely).
    """
    live = _running.get(session_id)
    if live is not None and not live.done():
        # Steering: the message is already in the log and the running turn will
        # read it at the next hop. Nothing to start.
        return False

    session = await load(session_id)
    if session is None:
        return False
    if session.status == "running":
        # Marked running with no task here: the process that owned it died. The
        # startup sweep fails those; racing it would double-write the terminal.
        logger.warning("session %s is running with no task in this process", session_id)
        return False
    if not await lifecycle.transition(session_id, session.status, "running", "woken"):
        return False

    task = asyncio.create_task(_drive(session_id), name=f"turn:{session_id}")
    _running[session_id] = task
    task.add_done_callback(lambda t: _running.pop(session_id, None))
    return True


async def cancel(session_id: str) -> bool:
    """
    Stop a session, whatever it is doing.

    A live turn is signalled and closes its own transcript; anything else is
    written straight to `cancelled`, because there is no loop to signal.
    """
    task = _running.get(session_id)
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return True

    session = await load(session_id)
    if session is None or session.status in lifecycle.TERMINAL:
        return False
    return await lifecycle.transition(session_id, session.status, "cancelled", "cancelled")


async def _drive(session_id: str) -> None:
    """
    Run one turn to its end, recording everything it does as it happens.

    Every exit path writes a terminal, setup included. Fold, manifest build and
    the wake repair are all awaits against the network, so a cancel is at least
    as likely to land there as inside the loop — and one that landed there used
    to leave the row `running` with no task behind it, which nothing but a
    restart could clear.
    """
    sink: _Sink | None = None
    try:
        session = await load(session_id)
        if session is None:
            return
        sink = _Sink(session)

        # Verify-on-wake: a call the last run left open is closed before anything
        # else, or the chat template rejects the request and the session is
        # permanently unloadable rather than merely interrupted.
        for closed in await slog.close_dangling(session_id):
            stream.publish(session_id, closed)

        messages, hops_used = await fold(session)
        try:
            tools = await registry.manifest(session.user_id, mcp=hands.smithery())
        except Exception:
            # MCP being unreachable costs the model its remote tools, not its turn.
            logger.exception("session %s: building the full manifest failed", session_id)
            tools = await registry.manifest(session.user_id)

        dispatch = registry.bind(sink.tool_context(), mcp_call=_mcp_call())
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
                # A failed append halts the run rather than letting it continue
                # unrecorded. One event of lag, which is the cost of not putting
                # the database on the token path.
                raise RuntimeError("the session log could not be written") from sink.failure
        await sink.close()
    except asyncio.CancelledError:
        # Safe to await here: a caught CancelledError is not re-delivered, and
        # the whole point of this path is to leave a transcript saying why the
        # run stopped. `cancel()` waits for that to finish before returning.
        await _ending(session_id, sink, "cancelled")
        raise
    except Exception:
        logger.exception("session %s: the turn failed outside the loop", session_id)
        await _ending(session_id, sink, "model_error")


async def _ending(session_id: str, sink: _Sink | None, reason: str) -> None:
    """
    Record the end of a run, with or without a sink.

    A cancel that lands on the very first read has no sink to write through, and
    without this the row stays `running` with no task behind it — a state only a
    restart clears, because `start` refuses to touch a running session.
    """
    if sink is not None:
        await sink.abort(reason)
        return
    try:
        status = "cancelled" if reason == "cancelled" else "failed"
        await slog.append(session_id, DoneEvent(reason=reason))
        await lifecycle.transition(session_id, "running", status, reason)
    except Exception:
        logger.exception("session %s: could not record the %s ending", session_id, reason)


def _model_options() -> dict[str, Any] | None:
    """Per-call model params, from config only — never from model output."""
    options = config.get("llm.options")
    return dict(options) if options else None


def _mcp_call():
    """Adapt the shared Smithery client to the registry's mcp_call shape, if there is one."""
    client = hands.smithery()
    if client is None:
        return None

    async def call(name: str, args: dict[str, Any], ctx: ToolContext):
        return await client.call(name, args, ctx)

    return call


# --- translating events into a record -----------------------------------------

# Tells the writer there is nothing more coming. Not None: None is a legal thing
# to find in a queue and this must never be mistaken for one.
_STOP = object()


class _Sink:
    """
    Appends what the loop yields, and moves the session when it ends.

    Write-behind, and that is the whole point: `emit` is SYNCHRONOUS. It hands
    the event to a writer task and returns, so the loop is never suspended on a
    database round trip while the model is mid-sentence. A remote Postgres at
    150ms would otherwise cost a 200-chunk reply thirty seconds of pure latency,
    because the loop only produces the next chunk once the consumer returns.

    Coalescing falls out of that rather than needing a timer: whatever piled up
    behind an in-flight append is merged into one row. A fast database writes
    fine-grained rows, a slow one writes fewer and larger, and neither changes
    when the first chunk reaches the screen.
    """

    def __init__(self, session: Session):
        self.session = session
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        # One-slot push-back, because asyncio.Queue cannot un-get: the merge
        # below stops on the first event of another kind and leaves it here.
        self._pushback: list[Any] = []
        self._writer = asyncio.create_task(self._write_loop(), name=f"log:{session.id}")
        # An append that fails must halt the run: nothing executes off the record.
        self.failure: BaseException | None = None
        self._done = False
        self._last_seq = 0
        self._hops = 0

    # --- what tools and the loop see -------------------------------------------

    def tool_context(self) -> ToolContext:
        """What tools may reach that is not an argument."""
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
        Decide a `requires_approval` tool.

        Attended, the human is watching this stream and can cancel mid-call, so
        the call proceeds — which is what makes "every mcp_* tool requires
        approval" affordable in conversation. Unattended has nobody to ask until
        Task 6 builds the park, and guessing yes there is exactly the fifteen
        hops of unsupervised Gmail that default was written against.
        """
        if self.session.mode == "attended" and bool(_cfg("approvals.attended_auto_approve", True)):
            return True
        logger.warning("session %s: %s needs approval and nothing can ask for it yet", self.session.id, name)
        return False

    def emit(self, event: Event) -> None:
        """Queue one event. Never blocks, so the model's stream never waits on us."""
        if isinstance(event, BudgetEvent):
            self._hops = event.hops_used
        self._queue.put_nowait(event)

    # --- the writer -------------------------------------------------------------

    async def _write_loop(self) -> None:
        """Drain the queue in order, merging any run of same-kind text into one row."""
        while True:
            event = self._pushback.pop() if self._pushback else await self._queue.get()
            if event is _STOP:
                return
            if isinstance(event, (ContentEvent, ReasoningEvent)):
                event = self._merge_text(event)
            try:
                await self._append(event)
            except Exception as e:  # noqa: BLE001 - recorded here, acted on by the loop
                logger.exception("session %s: append failed; halting the run", self.session.id)
                self.failure = e
                return

    def _merge_text(self, first: ContentEvent | ReasoningEvent) -> Event:
        """Take everything already queued of the same kind, without waiting for more."""
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
        """Write the terminal event and move the session, exactly once."""
        if self._done:
            return
        self._done = True
        await self._drain()

        # A run that was cut short may still hold an open call; the invariant
        # refuses a `done` while it does, and this is the last chance to close it.
        for closed in await slog.close_dangling(self.session.id):
            stream.publish(self.session.id, closed)
            self._last_seq = closed.seq

        await self._append(done)
        await self._save_cursor()

        new_status = lifecycle.status_for(done)
        # An unattended run that ends hands the session back to its human.
        mode = "attended" if new_status in lifecycle.TERMINAL and self.session.mode == "unattended" else None
        await lifecycle.transition(self.session.id, "running", new_status, done.reason, mode=mode)

    async def close(self, done: DoneEvent | None = None) -> None:
        """Finish the turn. A loop that ended without a `done` is a bug in the loop."""
        if done is not None:
            await self._finish(done)
            return
        await self._drain()
        if not self._done:
            logger.error("session %s: the loop ended with no done event", self.session.id)
            await self._finish(DoneEvent(reason="model_error"))

    async def abort(self, reason: str) -> None:
        """
        End the run from outside the loop, leaving a transcript that says why.

        Failures here are logged rather than raised — this IS the error path, and
        a raise would replace one unexplained session with two — but never
        swallowed silently, or a session stuck `running` looks like a mystery
        rather than the database error it was.
        """
        try:
            await self._finish(DoneEvent(reason=reason))
        except Exception:
            logger.exception("session %s: could not record the %s ending", self.session.id, reason)

    async def _save_cursor(self) -> None:
        """
        Write the cursor and the hop count back.

        Both are caches of what the log already says — the log is authoritative
        (G49) — so they are written once at the end of a turn rather than on
        every event, where they would double the writes on the token path.
        """
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
