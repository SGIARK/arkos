"""The world model behaves the way arkos does, one event at a time."""

from state_machine_module.world import NEW_SESSION, Event, World, step

START = Event("arkos", "starts the session")
TEXT = Event("model", "replies with text")
CANCEL = Event("you", "cancel")
RESTART = Event("server", "restarts")


def calls(*tools: str) -> Event:
    return Event("model", "calls", tools=tools)


def answer(verb: str, row: int = 0) -> Event:
    return Event("you", "answer", row=row, verb=verb)


def play(world: World, *happenings: Event) -> World:
    for event in happenings:
        world = step(world, event)
    return world


def test_step_a_park_tool_parks_at_the_hop_boundary() -> None:
    world = play(NEW_SESSION, START, calls("ask"))
    assert world.status == "awaiting_approval"
    assert [question.kind for question in world.questions] == ["ask"]
