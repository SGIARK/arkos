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

from dataclasses import dataclass, replace

from state_machine_module.machines import SANDBOX, SESSION, allows

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


def move(world: World, target: str, reason: str, mode: str = "") -> World:
    """lifecycle.transition: move the status, and the mode in the same update."""
    sneaked = mode == "unattended" and reason != "plan_approved"
    return replace(
        world,
        status=target,
        mode=mode or world.mode,
        illegal_move=world.illegal_move or not allows(SESSION, world.status, target),
        unattended_without_plan=world.unattended_without_plan or sneaked,
    )


def box_to(world: World, target: str) -> World:
    """Move the sandbox, noting a move its machine does not declare."""
    if world.box == target:
        return world
    return replace(world, box=target, illegal_move=world.illegal_move or not allows(SANDBOX, world.box, target))


def has_open_call(world: World) -> bool:
    """Whether any tool_call in the log still lacks a result."""
    in_rows = any(question.call_open for question in world.questions)
    in_grant = world.grant is not None and world.grant.call_open
    in_turn = world.turn is not None and world.turn.park_gated
    return in_rows or in_grant or in_turn


def close_dangling(world: World) -> World:
    """session_log.close_dangling: close every open call with an interrupted result."""
    questions = tuple(replace(question, call_open=False) for question in world.questions)
    grant = replace(world.grant, call_open=False) if world.grant else None
    turn = replace(world.turn, park_gated=False) if world.turn else None
    return replace(world, questions=questions, grant=grant, turn=turn)


def release(world: World, keep_box: bool) -> World:
    """_Sink._release_leases: flush what this turn changed, pause or reap the box, drop the lease."""
    turn = world.turn
    if turn is not None and turn.materialized:
        world = replace(world, unflushed=False, turn=replace(turn, materialized=False))
    if keep_box and world.box == "live":
        world = box_to(world, "paused")
    elif not keep_box and world.slot:
        lost = world.lost_edits or world.unflushed
        world = replace(box_to(world, "none"), slot=False, unflushed=False, lost_edits=lost)
    return replace(world, lease=False, lease_orphaned=False)


def finish(world: World, reason: str) -> World:
    """_Sink._finish: release, close dangling calls, write done{reason}, move the status."""
    world = close_dangling(release(world, keep_box=reason == "stopped"))
    target = status_for(reason)
    mode = "attended" if target in FINAL and world.mode == "unattended" else ""
    open_call = world.ended_with_open_call or has_open_call(world)
    world = replace(world, turn=None, hops_since_done=0, ended_with_open_call=open_call)
    return move(world, target, reason, mode)


def park(world: World) -> World:
    """_Sink.park: release the lease keeping the box, open the question, wait. No done is written."""
    turn = turn_of(world)
    tool = TOOL[turn.park]
    world = release(world, keep_box=True)
    kind = tool.park or "call"
    kept = tuple(question for question in world.questions if not (kind == "plan" and question.kind == "plan"))
    question = Question(kind=kind, tool=tool.name if kind in ("call", "plan") else "", call_open=turn.park_gated)
    world = replace(world, questions=(*kept, question), turn=None)
    return move(world, WAITING, tool.name)


def next_hop(world: World) -> World:
    """The top of run_turn's loop: end on the budget, park at a hop boundary, or call the model."""
    turn = turn_of(world)
    if turn.hops >= max_hops(world.mode):
        return finish(world, "max_hops")
    charged = not (turn.repair_pending or turn.reattempt_pending)
    turn = replace(turn, repair_pending=False, reattempt_pending=False)
    if charged and turn.park:
        return park(replace(world, turn=turn))
    if charged:
        turn = replace(turn, hops=turn.hops + 1)
    called_while_parked = world.model_called_while_parked or bool(turn.park)
    return replace(world, turn=turn, hops_since_done=turn.hops, model_called_while_parked=called_while_parked)


def take_box(world: World) -> World:
    """_Sink._lease("sandbox"): a pool slot, the folder lease, the box materialized once a turn."""
    turn = turn_of(world)
    if turn.materialized:
        return world
    world = box_to(world, "live")
    lease = world.lease or world.has_folder
    return replace(world, slot=True, lease=lease, lease_orphaned=False, turn=replace(turn, materialized=True))


def act(world: World, tool: Tool, approval: str) -> World:
    """A tool that changes something runs. `approval` is "person", "auto", or "" for none."""
    turn = turn_of(world)
    if tool.needs_box:
        world = take_box(world)
    return replace(
        world,
        unflushed=world.unflushed or (tool.needs_box and world.has_folder),
        gated_ran_unapproved=world.gated_ran_unapproved or (tool.gated and not approval),
        destructive_ran_unapproved=world.destructive_ran_unapproved or (tool.destructive and approval != "person"),
        acted_after_question=world.acted_after_question or bool(turn.park),
    )


def settle_approved_call(world: World) -> World:
    """_settle_gated_call: close the call the session parked on, running it once if approved."""
    grant = world.grant
    if grant is None or not grant.call_open:
        return world
    if grant.consumed or not grant.approved:
        return replace(world, grant=replace(grant, call_open=False))
    world = replace(world, grant=replace(grant, consumed=True, call_open=False))
    return act(world, TOOL[grant.tool], approval="person")


def drive(world: World) -> World:
    """runner._drive: settle a parked call, close dangling ones, fold the log, take a hop."""
    world = close_dangling(settle_approved_call(replace(world, turn=Turn())))
    turn = replace(turn_of(world), hops=world.hops_since_done)
    return next_hop(replace(world, turn=turn))


def start(world: World, reason: str, mode: str = "") -> World:
    """runner.start: move to running and drive a turn. A live turn makes it a no-op."""
    if world.status == "running":
        return world
    return drive(move(world, "running", reason, mode))


def model_replies(world: World, said_something: bool) -> World:
    """A hop with no tool calls. It ends an attended turn; unattended, it is a stall."""
    if world.mode == "attended":
        return finish(world, "turn_end")
    turn = turn_of(world)
    stall = turn.stall + 1
    if not said_something or stall >= STALL_STREAK:
        return finish(world, "stalled_progress")
    return next_hop(replace(world, turn=replace(turn, stall=stall, reattempts=0)))


def model_fails(world: World) -> World:
    """A retryable model error: the loop re-attempts the hop uncharged, until the retries run out."""
    turn = turn_of(world)
    if turn.reattempts >= MODEL_RETRIES:
        return finish(world, "model_error")
    turn = replace(turn, reattempts=turn.reattempts + 1, reattempt_pending=True)
    return next_hop(replace(world, turn=turn))


def call_gated(world: World, tool: Tool) -> World:
    """_Sink._approve: autopilot answers safe calls, a waiting question refuses, else the call parks."""
    turn = turn_of(world)
    if world.mode == "unattended" and not tool.destructive:
        return act(world, tool, approval="auto")
    if turn.park:
        return world
    return replace(world, turn=replace(turn, park=tool.name, park_gated=True))


def call(world: World, tool: Tool) -> World:
    """One tool call inside a hop, as loop._run_batch and envelope.execute handle it."""
    turn = turn_of(world)
    if tool.malformed:
        repair = turn.repair_pending or turn.repair_available
        return replace(world, turn=replace(turn, repair_available=False, repair_pending=repair))
    if tool.gated:
        return call_gated(world, tool)
    if tool.park:
        return replace(world, turn=replace(turn, park=tool.name, park_gated=False))
    if tool.finishes:
        return replace(world, turn=replace(turn, finished=True))
    return act(world, tool, approval="")


def model_calls(world: World, names: tuple[str, ...]) -> World:
    """A hop with tool calls: run them in order, then end the run or take the next hop."""
    world = replace(world, turn=replace(turn_of(world), stall=0))
    for name in names:
        world = call(world, TOOL[name])
    turn = replace(turn_of(world), reattempts=0)
    world = replace(world, turn=turn)
    if turn.finished and world.mode == "unattended":
        return finish(world, "completed")
    return next_hop(world)


def answer_by_message(world: World) -> World:
    """_answer_by_message: the composer answers the OLDEST open row, and never one asking consent."""
    if not world.questions:
        return start(world, "answered")
    oldest, rest = world.questions[0], world.questions[1:]
    if oldest.kind in CONSENT_KINDS:
        return world
    world = replace(world, questions=rest)
    return start(world, "plan_requested" if oldest.is_goal else "answered")


def send_message(world: World) -> World:
    """POST /messages: a waiting session routes it to a question; otherwise it starts a turn."""
    if world.status == WAITING:
        return answer_by_message(world)
    return start(close_dangling(world), "woken")


def without(world: World, index: int) -> World:
    """The world with the row at `index` answered, so no longer open."""
    return replace(world, questions=world.questions[:index] + world.questions[index + 1 :])


def dismiss_goal(world: World) -> World:
    """Dismissing the goal question: back to idle, if the session is still waiting on it."""
    return move(world, "idle", "goal_cancelled") if world.status == WAITING else world


def answer_plan(world: World, index: int, verb: str) -> World:
    """_answer_plan: approve starts an unattended run, decline goes idle, a reply redrafts."""
    if verb == "approve":
        if not world.has_folder or world.status == "running":
            return world
        return start(without(world, index), "plan_approved", mode="unattended")
    world = without(world, index)
    if verb == "decline":
        return move(world, "idle", "plan_declined") if world.status == WAITING else world
    return start(world, "plan_reply")


def respond(world: World, index: int, verb: str) -> World:
    """POST /approvals/{id}/respond, for the open row at `index`."""
    row = world.questions[index]
    if row.kind == "plan":
        return answer_plan(world, index, verb)
    world = without(world, index)
    if row.kind == "call":
        grant = Grant(tool=row.tool, approved=verb == "approve", call_open=row.call_open)
        return start(replace(world, grant=grant), "answered")
    if row.is_goal and verb == "dismiss":
        return dismiss_goal(world)
    return start(world, "plan_requested" if row.is_goal else "answered")


def press_autopilot(world: World) -> World:
    """POST /sessions/{id}/approve: park on the goal question without running anything."""
    if world.mode == "unattended":
        return world
    world = move(world, WAITING, "goal_requested")
    if any(question.is_goal for question in world.questions):
        return move(world, "idle", "goal_failed")
    return replace(world, questions=(*world.questions, Question(kind="ask", is_goal=True)))


def cancel(world: World) -> World:
    """runner.cancel: a live turn lands as done{cancelled}; otherwise the ending is written directly."""
    if world.status == "running":
        return finish(world, "cancelled")
    world = replace(close_dangling(world), hops_since_done=0)
    return move(world, "cancelled", "cancelled", "attended" if world.mode == "unattended" else "")
