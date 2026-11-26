"""The world model behaves the way arkos does, one event at a time."""

from state_machine_module.world import HOME_CHAT, NEW_SESSION, Event, World, step

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


def test_step_an_approved_call_runs_once_during_setup() -> None:
    world = play(NEW_SESSION, START, calls("mcp_send_email"), answer("approve"))
    assert world.status == "running"
    assert world.grant is not None and world.grant.consumed and not world.grant.call_open


def test_step_a_declined_call_never_runs() -> None:
    world = play(NEW_SESSION, START, calls("mcp_send_email"), answer("decline"))
    assert world.grant is not None and not world.grant.consumed


def test_step_an_approved_plan_starts_an_unattended_run() -> None:
    world = play(NEW_SESSION, START, calls("propose_plan"), answer("approve"))
    assert (world.status, world.mode) == ("running", "unattended")


def test_step_the_home_chat_cannot_approve_a_plan() -> None:
    world = play(HOME_CHAT, Event("you", "send a message"), calls("propose_plan"))
    assert play(world, answer("approve")) == world


def test_step_an_attended_reply_flushes_and_reaps_the_box() -> None:
    world = play(NEW_SESSION, START, calls("run_command"), TEXT)
    assert (world.status, world.box, world.slot, world.unflushed) == ("idle", "none", False, False)


def test_step_stop_pauses_the_box_and_keeps_the_mode() -> None:
    world = play(NEW_SESSION, START, calls("propose_plan"), answer("approve"), calls("run_command"))
    world = play(world, Event("you", "stop"))
    assert (world.status, world.mode, world.box, world.unflushed) == ("idle", "unattended", "paused", False)
