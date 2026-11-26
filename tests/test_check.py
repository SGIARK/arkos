"""The checker finds what today's arkos breaks, with paths that reproduce it."""

import pytest

from state_machine_module.check import Report, explore, main
from state_machine_module.rules import broken
from state_machine_module.world import ROOTS, step

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


def test_explore_paths_reproduce_their_rule(report: Report) -> None:
    for finding in report.findings:
        world = finding.root
        for event in finding.path:
            world = step(world, event)
        assert finding.rule in broken(world)


def test_explore_returns_the_shortest_path(report: Report) -> None:
    lengths = {finding.rule.name: len(finding.path) for finding in report.findings}
    assert lengths["one_question_while_waiting"] == 2
    assert lengths["ended_runs_are_attended"] == 4


def test_main_exits_nonzero_when_a_rule_breaks(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--depth", "3"]) == 1
    assert "BROKEN  one_question_while_waiting" in capsys.readouterr().out
