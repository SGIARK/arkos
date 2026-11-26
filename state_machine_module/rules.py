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


RULES = (
    Rule(
        "legal_moves",
        "Every part of a session moves only as its declared machine allows.",
        lambda w: not w.illegal_move,
    ),
    Rule(
        "one_question_while_waiting",
        "A waiting session has exactly one open question; other sessions have none.",
        one_question_while_waiting,
    ),
    Rule("ended_runs_are_attended", "An ended session holds no unattended run.", ended_runs_are_attended),
    Rule(
        "unattended_only_from_a_plan",
        "Only approving a plan starts an unattended run.",
        lambda w: not w.unattended_without_plan,
    ),
    Rule(
        "no_ending_with_a_call_open", "No turn ends with a tool call still open.", lambda w: not w.ended_with_open_call
    ),
    Rule(
        "gated_calls_wait_for_approval",
        "A call that needs approval runs only once approved.",
        lambda w: not w.gated_ran_unapproved,
    ),
    Rule(
        "destructive_calls_need_a_person",
        "A destructive call never runs without a person's approval.",
        lambda w: not w.destructive_ran_unapproved,
    ),
    Rule(
        "parked_turns_stop",
        "Once a turn decides to park, the model is not called again before the answer.",
        lambda w: not w.model_called_while_parked,
    ),
    Rule(
        "nothing_acts_after_a_question",
        "No tool acts between a turn raising a question and its answer.",
        lambda w: not w.acted_after_question,
    ),
    Rule("leases_only_while_running", "Only a running session holds a lease.", leases_only_while_running),
    Rule(
        "edits_reach_the_store",
        "A box is never destroyed while it holds edits the store does not have.",
        lambda w: not w.lost_edits,
    ),
    Rule("hops_within_budget", "A turn never takes more hops than its budget.", hops_within_budget),
)


def broken(world: World) -> list[Rule]:
    """The rules `world` breaks."""
    return [rule for rule in RULES if not rule.holds(world)]
