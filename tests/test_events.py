"""Store-shape = wire-shape: what we save is what we push, with no translation."""

import pytest

from agent_module import events as ev


def test_every_vocabulary_kind_is_implemented():
    """The table in contracts.md is the whole vocabulary."""
    assert set(ev._BY_KIND) == {
        "user",
        "content",
        "reasoning",
        "tool_call",
        "tool_result",
        "status",
        "todo",
        "budget",
        "lifecycle",
        "view_transform",
        "done",
    }


@pytest.mark.parametrize(
    "event",
    [
        ev.UserEvent(text="hi", source="system"),
        ev.ContentEvent(text="the answer"),
        ev.ReasoningEvent(text="thinking"),
        ev.ToolCallEvent(id="c1", name="grep", args={"q": "x"}),
        ev.ToolResultEvent(id="c1", ok=False, content="boom", error_kind="timeout", total_chars=9, ref="r1"),
        ev.StatusEvent(label="using the browser", url="/frames"),
        ev.TodoEvent(items=[{"text": "step", "done": False}]),
        ev.BudgetEvent(hops_used=7, hops_max=15),
        ev.LifecycleEvent(from_="running", to="idle", reason="turn_end"),
        ev.ViewTransformEvent(rung=1, dropped_refs=["r1"]),
        ev.DoneEvent(reason="completed"),
    ],
)
def test_round_trips_through_a_stored_row(event):
    row = event.to_row()
    assert ev.parse_event(row["kind"], row["payload"], row["version"]) == event


def test_lifecycle_uses_from_on_the_wire():
    """`from` is a Python keyword, so the wire name must not leak the underscore."""
    row = ev.LifecycleEvent(from_="pending", to="running").to_row()
    assert "from" in row["payload"] and "from_" not in row["payload"]


def test_unknown_kind_is_loud():
    with pytest.raises(ValueError):
        ev.parse_event("not_a_kind", {})


def test_turn_end_is_not_terminal():
    """It is the attended 'I have said my piece', the only trigger for idle."""
    assert not ev.DoneEvent(reason="turn_end").is_terminal()
    assert ev.DoneEvent(reason="max_hops").is_terminal()


def test_a_newer_writers_extra_keys_are_dropped_not_fatal():
    """An old reader must still render a v2 row."""
    event = ev.parse_event("content", {"text": "hi", "sentiment": "warm"}, version=2)
    assert event.text == "hi" and event.version == 2


def test_a_missing_required_field_is_loud():
    with pytest.raises(ValueError):
        ev.parse_event("tool_call", {"id": "c1"})
