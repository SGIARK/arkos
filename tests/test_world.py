"""The world model behaves the way arkos does, one event at a time."""

from state_machine_module.world import (
    HOME_CHAT,
    NEW_SESSION,
    Event,
    World,
    events,
    lease_state,
    step,
)

START = Event("arkos", "starts the session")
TEXT = Event("model", "replies with text")
CANCEL = Event("you", "cancel")
RESTART = Event("server", "restarts")
MESSAGE = Event("you", "send a message")


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


def test_step_finish_task_completes_only_an_unattended_run() -> None:
    attended = play(NEW_SESSION, START, calls("finish_task"))
    unattended = play(NEW_SESSION, START, calls("propose_plan"), answer("approve"), calls("finish_task"))
    assert attended.status == "running"
    assert (unattended.status, unattended.mode) == ("completed", "attended")


def test_step_cancel_while_waiting_leaves_the_question_open() -> None:
    """Today's behavior, which the rule one_question_while_waiting flags."""
    world = play(NEW_SESSION, START, calls("ask"), CANCEL)
    assert world.status == "cancelled"
    assert len(world.questions) == 1


def test_step_restart_fails_a_running_session() -> None:
    assert play(NEW_SESSION, START, RESTART).status == "failed"


def test_step_a_restart_mid_turn_orphans_the_lease() -> None:
    assert lease_state(play(NEW_SESSION, START, calls("run_command"), RESTART)) == "orphaned"


def test_step_a_later_turn_takes_its_orphaned_lease_back() -> None:
    world = play(NEW_SESSION, START, calls("run_command"), RESTART, MESSAGE, calls("run_command"))
    assert lease_state(world) == "held"


def test_events_offer_the_model_nothing_outside_a_turn() -> None:
    assert not [event for event in events(HOME_CHAT) if event.actor == "model"]
