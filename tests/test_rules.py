"""Each rule flags the world it describes, and passes a world that keeps it."""

from state_machine_module.rules import broken
from state_machine_module.world import NEW_SESSION, Question, Turn, World, max_hops


def broken_names(world: World) -> set[str]:
    return {rule.name for rule in broken(world)}


def test_broken_finds_nothing_in_a_new_session() -> None:
    assert broken(NEW_SESSION) == []


def test_one_question_while_waiting_flags_an_ended_session_with_a_question() -> None:
    world = World(status="cancelled", questions=(Question(kind="ask"),))
    assert "one_question_while_waiting" in broken_names(world)


def test_one_question_while_waiting_flags_a_wait_with_no_question() -> None:
    assert "one_question_while_waiting" in broken_names(World(status="awaiting_approval"))


def test_ended_runs_are_attended_flags_a_failed_unattended_session() -> None:
    assert "ended_runs_are_attended" in broken_names(World(status="failed", mode="unattended"))


def test_leases_only_while_running_flags_an_idle_lease() -> None:
    assert "leases_only_while_running" in broken_names(World(status="idle", lease=True))


def test_hops_within_budget_flags_a_turn_over_budget() -> None:
    world = World(status="running", turn=Turn(hops=max_hops("attended") + 1))
    assert "hops_within_budget" in broken_names(world)
