"""
The event vocabulary. Store-shape = wire-shape: the object saved is the object
pushed, so there is no translation layer between db, SSE and tests.

Consumed by session_log (store), SSE (wire), the runner (lifecycle), and the
frontend. `to_row()` and `parse_event()` are the only serialisation boundary.
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

# turn_end is NON-terminal: the attended "I have said my piece", and the only
# trigger for running -> idle. interrupted is written by the startup sweep for a
# session the process died underneath.
DoneReason = Literal[
    "turn_end",
    "completed",
    "max_hops",
    "wall_clock",
    "model_error",
    "context_overflow",
    "cancelled",
    "interrupted",
]

TERMINAL_REASONS: frozenset[str] = frozenset(
    {"completed", "max_hops", "wall_clock", "model_error", "context_overflow", "cancelled", "interrupted"}
)


@dataclass(slots=True)
class Event:
    """
    Base for every event. `kind` identifies the row; `version` lets readers
    upcast, since rows are never rewritten.
    """

    kind: ClassVar[EventKind]
    version: int = field(default=1, kw_only=True)

    def payload(self) -> dict[str, Any]:
        """The jsonb column, and the SSE data. Everything but kind and version."""
        return {k: v for k, v in asdict(self).items() if k != "version"}

    def to_row(self) -> dict[str, Any]:
        return {"kind": self.kind, "version": self.version, "payload": self.payload()}


@dataclass(slots=True)
class UserEvent(Event):
    """`source` distinguishes the human from the nudge and system injections."""

    kind: ClassVar[EventKind] = "user"
    text: str
    source: Literal["human", "system"] = "human"


@dataclass(slots=True)
class ContentEvent(Event):
    kind: ClassVar[EventKind] = "content"
    text: str


@dataclass(slots=True)
class ReasoningEvent(Event):
    """Separate from content because the fold drops it and replays content."""

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
    """`url` is the ephemeral browser frame stream: announced, not stored for replay."""

    kind: ClassVar[EventKind] = "status"
    label: str
    url: str | None = None


@dataclass(slots=True)
class TodoEvent(Event):
    kind: ClassVar[EventKind] = "todo"
    items: list[dict[str, Any]]


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
    Rebuild an event from a stored row.

    A newer writer's extra payload keys are dropped rather than raising, which
    is what `version` is for: an old reader must still render a v2 row.

    Raises:
        ValueError: on an unknown kind, or a payload missing a required field.
            An out-of-vocabulary row is loud rather than silently absent.
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
