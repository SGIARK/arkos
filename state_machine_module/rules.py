"""What arkos must never do, judged one world at a time.

The checker evaluates every rule on every world it reaches. Rules that need two
sessions at once, or races inside a single step, are not modeled yet.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from state_machine_module.world import FINAL, WAITING, World, max_hops


@dataclass(frozen=True, slots=True)
class Rule:
    """A named rule, the sentence it stands for, and the test that it holds."""

    name: str
    says: str
    holds: Callable[[World], bool]


def one_question_while_waiting(world: World) -> bool:
    """A waiting session has exactly one open question, and no other session has any."""
    if world.status == WAITING:
        return len(world.questions) == 1
    return not world.questions


def ended_runs_are_attended(world: World) -> bool:
    """An ended session holds no unattended run."""
    return world.status not in FINAL or world.mode == "attended"


def leases_only_while_running(world: World) -> bool:
    """Only a running session holds a lease."""
    return not world.lease or world.status == "running"


def hops_within_budget(world: World) -> bool:
    """A turn never takes more hops than its budget."""
    return world.turn is None or world.turn.hops <= max_hops(world.mode)
