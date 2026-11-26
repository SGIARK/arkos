"""The checker finds what today's arkos breaks, with paths that reproduce it."""

import pytest

from state_machine_module.check import Report, explore
from state_machine_module.world import ROOTS

# The longest of today's shortest counterexamples is four events.
DEPTH = 5

BROKEN_TODAY = {
    "one_question_while_waiting",
    "ended_runs_are_attended",
    "parked_turns_stop",
    "nothing_acts_after_a_question",
    "leases_only_while_running",
    "edits_reach_the_store",
}


@pytest.fixture(scope="module")
def report() -> Report:
    return explore(ROOTS, DEPTH)


def test_explore_finds_the_rules_today_breaks(report: Report) -> None:
    assert {finding.rule.name for finding in report.findings} == BROKEN_TODAY
