"""The state machines arkos is built from, declared as data.

Code that needs a machine reads it from here. The world model in `world.py` notes
any move a machine does not declare, and a test holds `SESSION` to
`harness_module.lifecycle.ALLOWED`, so the declaration cannot drift from the code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Move:
    """One legal move between two states, and what triggers it."""

    source: str
    target: str
    trigger: str


@dataclass(frozen=True, slots=True)
class Start:
    """A state a machine can begin in, and what puts it there."""

    state: str
    trigger: str


@dataclass(frozen=True, slots=True)
class Machine:
    """A state machine: what it tracks, its states, where it begins and ends, and its legal moves."""

    name: str
    about: str
    states: tuple[str, ...]
    starts: tuple[Start, ...]
    final: tuple[str, ...]
    moves: tuple[Move, ...]


def allows(machine: Machine, source: str, target: str) -> bool:
    """Whether `machine` declares a move from `source` to `target`."""
    return any(move.source == source and move.target == target for move in machine.moves)


def follows(machine: Machine, source: str, target: str) -> bool:
    """Whether going from `source` to `target` stays put or takes a move `machine` declares."""
    return source == target or allows(machine, source, target)


def pairs(machine: Machine) -> frozenset[tuple[str, str]]:
    """Every (source, target) pair a move connects, triggers set aside."""
    return frozenset((move.source, move.target) for move in machine.moves)


SESSION = Machine(
    name="session",
    about=(
        "sessions.status: where a session is in its life. lifecycle.transition is its one writer, and it"
        " refuses any move not listed here."
    ),
    states=("pending", "idle", "running", "awaiting_approval", "completed", "failed", "cancelled"),
    starts=(Start("pending", "POST /sessions creates it"), Start("idle", "a first login creates the home Chat")),
    final=("completed", "failed", "cancelled"),
    moves=(
        Move("pending", "running", "runner.start claims the new session"),
        Move("pending", "cancelled", "Cancel"),
        Move("pending", "awaiting_approval", "Autopilot asks for the goal"),
        Move("idle", "running", "a message, or Resume"),
        Move("idle", "cancelled", "Cancel"),
        Move("idle", "awaiting_approval", "Autopilot asks for the goal"),
        Move("running", "idle", "the turn ends, or Stop"),
        Move("running", "awaiting_approval", "the turn parks on a question"),
        Move("running", "completed", "finish_task ends an unattended run"),
        Move("running", "failed", "a budget, model or harness failure, or a restart"),
        Move("running", "cancelled", "Cancel"),
        Move("awaiting_approval", "running", "an answer wakes the session"),
        Move("awaiting_approval", "idle", "the plan is declined, or the goal question dismissed"),
        Move("awaiting_approval", "cancelled", "Cancel"),
        Move("completed", "running", "a message restarts it"),
        Move("failed", "running", "a message restarts it"),
        Move("cancelled", "running", "a message restarts it"),
        Move("completed", "awaiting_approval", "Autopilot asks for the goal"),
        Move("failed", "awaiting_approval", "Autopilot asks for the goal"),
        Move("cancelled", "awaiting_approval", "Autopilot asks for the goal"),
    ),
)

SANDBOX = Machine(
    name="sandbox",
    about=(
        "The session's e2b box. Its disk is a cache of the store: a turn's first sandbox call copies the"
        " session's folders in, and the turn's ending flushes them back."
    ),
    states=("none", "live", "paused"),
    starts=(Start("none", "a session begins without a box"),),
    final=(),
    moves=(
        Move("none", "live", "the turn's first sandbox call boots a box"),
        Move("paused", "live", "the turn's first sandbox call resumes the box"),
        Move("live", "paused", "a park or Stop flushes the box, then pauses it"),
        Move("live", "none", "the turn ends and flushes, then reaps it; or a restart kills it"),
        Move("paused", "none", "the next ending reaps it, or a later claim reclaims it"),
    ),
)
