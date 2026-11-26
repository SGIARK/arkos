"""Each rule flags the world it describes, and passes a world that keeps it."""

from state_machine_module.rules import broken
from state_machine_module.world import NEW_SESSION, Question, World


def broken_names(world: World) -> set[str]:
    return {rule.name for rule in broken(world)}


def test_broken_finds_nothing_in_a_new_session() -> None:
    assert broken(NEW_SESSION) == []


def test_one_question_while_waiting_flags_an_ended_session_with_a_question() -> None:
    world = World(status="cancelled", questions=(Question(kind="ask"),))
    assert "one_question_while_waiting" in broken_names(world)
