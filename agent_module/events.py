"""
The event vocabulary. Store-shape equals wire-shape: the object saved is the
object pushed, so db, SSE and tests share one shape.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, ClassVar, Literal

EventKind = Literal[
    "user",
    "content",
    "reasoning",
    "tool_call",
    "tool_result",
    "status",
    "todo",
    "budget",
    "lifecycle",
    "view_transform",
    "done",
]

# turn_end and stopped are NON-terminal: they are the two triggers for
# running -> idle, and they differ only in who ended the hop.
DoneReason = Literal[
    "turn_end",
    # A human pressed Stop. Not a terminal and not a failure: the run is held,
    # the mode is kept, and a message or a plain start picks it up again.
    "stopped",
    "completed",
    "max_hops",
    "wall_clock",
    # The model produced no work: an empty reply, or a third consecutive
    # bare-text hop after the continuation and the finish nudge. Split out of
    # `model_error` in 11.8.5, because nothing errored.
    "stalled_progress",
    "model_error",
    # The harness failed, not the model: `_drive`'s catch-all, or a loop that
    # ended with no `done`. Also split out of `model_error` in 11.8.5.
    "internal_error",
    "context_overflow",
    "cancelled",
    "interrupted",
]

TERMINAL_REASONS: frozenset[str] = frozenset(
    {
        "completed",
        "max_hops",
        "wall_clock",
        "stalled_progress",
        "model_error",
        "internal_error",
        "context_overflow",
        "cancelled",
        "interrupted",
    }
)


@dataclass(slots=True)
class Event:
    """Base for every event. Rows are never rewritten, so readers upcast by `version`."""

    kind: ClassVar[EventKind]
    version: int = field(default=1, kw_only=True)

    def payload(self) -> dict[str, Any]:
        """Return the jsonb column and SSE data: everything but kind and version."""
        return {k: v for k, v in asdict(self).items() if k != "version"}

    def to_row(self) -> dict[str, Any]:
        return {"kind": self.kind, "version": self.version, "payload": self.payload()}


@dataclass(slots=True)
class UserEvent(Event):
    """User input; `source` separates the human from system injections."""

    kind: ClassVar[EventKind] = "user"
    text: str
    source: Literal["human", "system"] = "human"


@dataclass(slots=True)
class ContentEvent(Event):
    kind: ClassVar[EventKind] = "content"
    text: str


@dataclass(slots=True)
class ReasoningEvent(Event):
    """Model reasoning; never folded back into the message list."""

    kind: ClassVar[EventKind] = "reasoning"
    text: str


@dataclass(slots=True)
class ToolCallEvent(Event):
    kind: ClassVar[EventKind] = "tool_call"
    id: str
    name: str
    args: dict[str, Any]


@dataclass(slots=True)
class ToolResultEvent(Event):
    """`content` is view-capped; `ref` pages the full blob."""

    kind: ClassVar[EventKind] = "tool_result"
    id: str
    ok: bool
    content: str
    error_kind: str | None = None
    total_chars: int | None = None
    ref: str | None = None


@dataclass(slots=True)
class StatusEvent(Event):
    """Progress label; `url` is an ephemeral frame stream, not stored for replay."""

    kind: ClassVar[EventKind] = "status"
    label: str
    url: str | None = None


# --- the checklist item, defined ONCE ------------------------------------------
#
# This shape used to be spelled out in five places — the tool's validator, the
# session-create seed, the terminal sweep, and twice in the frontend — and it had
# already drifted: the UI tested for a status `"completed"` that `todo_write`
# never accepted, so a list the model dutifully marked done rendered entirely
# unchecked. One definition, and the vocabulary lives with it.

# The tool that writes it. `tool_module.tools.control` re-exports this as
# TODO_TOOL so the spec and the trackers cannot name different tools.
TODO_TOOL = "todo_write"

TodoStatus = Literal["pending", "in_progress", "done"]

TODO_STATUSES: frozenset[str] = frozenset(("pending", "in_progress", "done"))
TODO_PENDING: TodoStatus = "pending"
TODO_IN_PROGRESS: TodoStatus = "in_progress"
TODO_DONE: TodoStatus = "done"


def todo_item(text: str, status: str = TODO_PENDING) -> dict[str, Any]:
    """One checklist item, built the one way it is built."""
    return {"text": str(text), "status": status}


def todo_is_done(item: dict[str, Any]) -> bool:
    """Whether an item counts as finished, for anyone counting."""
    return str(item.get("status", "")) == TODO_DONE


@dataclass(slots=True)
class TodoEvent(Event):
    kind: ClassVar[EventKind] = "todo"
    items: list[dict[str, Any]]


class TodoTracker:
    """Follows `todo_write` calls past and reports the list as it stands.

    The list is in the CALL's arguments and whether it was accepted is in the
    RESULT, so neither alone is enough — which is why this is a small object
    rather than a function. `todo_write` is latest-wins: the whole list every
    time, so a successful call replaces rather than merges.

    Written once because it was written twice: the loop kept a copy for its hop
    scaffold and the runner kept another for its terminal sweep, and the two
    could disagree about what the model had last said.
    """

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self._pending: dict[str, list[dict[str, Any]]] = {}

    def saw_call(self, call_id: str, name: str, args: dict[str, Any]) -> None:
        """Remember what a `todo_write` asked for; other tools are ignored."""
        if name != TODO_TOOL:
            return
        items = args.get("items")
        if isinstance(items, list):
            self._pending[call_id] = [item for item in items if isinstance(item, dict)]

    def saw_result(self, call_id: str, ok: bool) -> bool:
        """Commit that call if it succeeded. True when the list changed."""
        written = self._pending.pop(call_id, None)
        if written is None or not ok:
            return False
        self.items = written
        return True

    def resolve(self) -> bool:
        """Mark everything done. True when that changed anything."""
        if not self.items or all(todo_is_done(item) for item in self.items):
            return False
        self.items = [{**item, "status": TODO_DONE} for item in self.items]
        return True


@dataclass(slots=True)
class BudgetEvent(Event):
    kind: ClassVar[EventKind] = "budget"
    hops_used: int
    hops_max: int


@dataclass(slots=True)
class LifecycleEvent(Event):
    """Appended by transition() on every status change."""

    kind: ClassVar[EventKind] = "lifecycle"
    from_: str
    to: str
    reason: str | None = None

    def payload(self) -> dict[str, Any]:
        return {"from": self.from_, "to": self.to, "reason": self.reason}


@dataclass(slots=True)
class ViewTransformEvent(Event):
    """A context-ladder drop. View-only: the log is never rewritten."""

    kind: ClassVar[EventKind] = "view_transform"
    rung: int
    dropped_refs: list[str]


@dataclass(slots=True)
class DoneEvent(Event):
    kind: ClassVar[EventKind] = "done"
    reason: DoneReason

    def is_terminal(self) -> bool:
        return self.reason in TERMINAL_REASONS


_BY_KIND: dict[str, type[Event]] = {
    cls.kind: cls
    for cls in (
        UserEvent,
        ContentEvent,
        ReasoningEvent,
        ToolCallEvent,
        ToolResultEvent,
        StatusEvent,
        TodoEvent,
        BudgetEvent,
        LifecycleEvent,
        ViewTransformEvent,
        DoneEvent,
    )
}


def parse_event(kind: str, payload: dict[str, Any], version: int = 1) -> Event:
    """
    Rebuild an event from a stored row, dropping payload keys this reader does not know.

    Raises:
        ValueError: on an unknown kind, or a payload missing a required field.
    """
    cls = _BY_KIND.get(kind)
    if cls is None:
        raise ValueError(f"unknown event kind: {kind!r}")
    data = dict(payload)
    if cls is LifecycleEvent:
        data["from_"] = data.pop("from", None)
    known = {f.name for f in fields(cls)}
    data = {k: v for k, v in data.items() if k in known}
    try:
        return cls(**data, version=version)
    except TypeError as e:
        raise ValueError(f"bad payload for {kind!r}: {e}") from e
