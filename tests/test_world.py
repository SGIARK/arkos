"""The world model behaves the way arkos does, one event at a time."""

from state_machine_module.world import Event

START = Event("arkos", "starts the session")
TEXT = Event("model", "replies with text")
CANCEL = Event("you", "cancel")
RESTART = Event("server", "restarts")


def calls(*tools: str) -> Event:
    return Event("model", "calls", tools=tools)


def answer(verb: str, row: int = 0) -> Event:
    return Event("you", "answer", row=row, verb=verb)
