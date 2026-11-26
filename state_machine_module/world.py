"""arkos as it behaves today, written as pure functions the checker can explore.

`World` is one session and what it holds: its status and mode, the turn in flight,
its open questions, its sandbox and its folder lease. `events` lists what can happen
next, and `step` returns the world after arkos handles one of them. Each event is a
single atomic step, so races inside a step are not explored yet.

The budgets are scaled down from config.yaml so the search stays small. The logic
that reads them is the same.

The fields World groups under "records" are never read by arkos. They note what
happened, so that `rules.py` can judge it.
"""

from __future__ import annotations

from dataclasses import dataclass

from state_machine_module.machines import SESSION

ATTENDED_MAX_HOPS = 2  # budgets.attended.max_hops is 6
UNATTENDED_MAX_HOPS = 3  # budgets.unattended.max_hops is 15
STALL_STREAK = 3  # budgets.stall_streak
MODEL_RETRIES = 1  # budgets.model_retries is 3
MAX_CALLS_PER_HOP = 2

FINAL = frozenset(SESSION.final)
WAITING = "awaiting_approval"
CONSENT_KINDS = ("call", "approval", "plan")


@dataclass(frozen=True, slots=True)
class Tool:
    """What the world needs to know about a tool the model can call."""

    name: str
    park: str = ""  # the question kind its own result parks the turn on
    gated: bool = False  # requires_approval: the gate parks the call itself
    destructive: bool = False  # on tools.destructive: it parks even when unattended
    needs_box: bool = False
    finishes: bool = False
    malformed: bool = False  # its arguments do not parse


TOOLS = (
    Tool("run_command", needs_box=True),
    Tool("mcp_send_email", gated=True),
    Tool("mcp_trash_thread", gated=True, destructive=True),
    Tool("ask", park="ask"),
    Tool("request_approval", park="approval"),
    Tool("propose_plan", park="plan"),
    Tool("finish_task", finishes=True),
    Tool("bad_args", malformed=True),
)
TOOL = {tool.name: tool for tool in TOOLS}


@dataclass(frozen=True, slots=True)
class Question:
    """An open approvals row."""

    kind: str  # ask, approval, call or plan
    is_goal: bool = False  # Autopilot's opening ask, always bound to tool_call_id "autopilot-goal"
    tool: str = ""  # a call or plan row: the tool that raised it
    call_open: bool = False  # a call row: its tool_call is still open in the log


@dataclass(frozen=True, slots=True)
class Grant:
    """The newest answered call row, which the next turn's setup settles."""

    tool: str
    approved: bool
    consumed: bool = False
    call_open: bool = False


@dataclass(frozen=True, slots=True)
class Turn:
    """The turn in flight. Between events it rests while the model writes its current hop."""

    hops: int = 0
    repair_available: bool = True
    repair_pending: bool = False
    reattempts: int = 0
    reattempt_pending: bool = False
    stall: int = 0
    finished: bool = False
    park: str = ""  # the tool this turn parks on at its next hop boundary
    park_gated: bool = False  # that tool's own call is still open, held by the gate
    materialized: bool = False


@dataclass(frozen=True, slots=True)
class World:
    """One session and everything it holds."""

    status: str
    mode: str = "attended"
    has_folder: bool = True
    turn: Turn | None = None
    hops_since_done: int = 0
    questions: tuple[Question, ...] = ()
    grant: Grant | None = None
    box: str = "none"
    slot: bool = False
    unflushed: bool = False
    lease: bool = False
    lease_orphaned: bool = False
    # Records: what happened, for the rules. Nothing in arkos reads these.
    illegal_move: bool = False
    unattended_without_plan: bool = False
    ended_with_open_call: bool = False
    gated_ran_unapproved: bool = False
    destructive_ran_unapproved: bool = False
    model_called_while_parked: bool = False
    acted_after_question: bool = False
    lost_edits: bool = False


NEW_SESSION = World(status="pending")
HOME_CHAT = World(status="idle", has_folder=False)
ROOTS = (NEW_SESSION, HOME_CHAT)
ROOT_NAMES = {NEW_SESSION: "a new session (POST /sessions)", HOME_CHAT: "the home Chat"}


@dataclass(frozen=True, slots=True)
class Event:
    """Something that happens to the session, and who does it."""

    actor: str
    action: str
    tools: tuple[str, ...] = ()
    row: int | None = None
    verb: str = ""


def describe(event: Event) -> str:
    """The event in words, as a counterexample lists it."""
    if event.tools:
        return f"{event.actor}: {event.action} {', '.join(event.tools)}"
    if event.row is not None:
        return f"{event.actor}: {event.action} question {event.row + 1}: {event.verb}"
    return f"{event.actor}: {event.action}"


def max_hops(mode: str) -> int:
    """The turn budget for a session mode."""
    return UNATTENDED_MAX_HOPS if mode == "unattended" else ATTENDED_MAX_HOPS


def status_for(reason: str) -> str:
    """lifecycle.status_for: where done{reason} moves a running session."""
    if reason in ("turn_end", "stopped"):
        return "idle"
    if reason in ("completed", "cancelled"):
        return reason
    return "failed"


def turn_of(world: World) -> Turn:
    """The turn in flight. Only a running world has one."""
    if world.turn is None:
        raise ValueError(f"no turn is in flight while the session is {world.status}")
    return world.turn
