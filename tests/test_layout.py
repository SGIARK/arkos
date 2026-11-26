"""Layouts keep states, frames and labels clear of each other, and every arrow meets its states."""

import itertools

import pytest

from state_machine_module.layout import Box, Layout, Point, drawn_moves, frames, layout, point_on, rows
from state_machine_module.machines import MACHINES, SESSION, Machine, Move

CASES = [(machine, labelled) for machine in MACHINES for labelled in (True, False)]
IDS = [f"{machine.name}-{'labelled' if labelled else 'bare'}" for machine, labelled in CASES]
EDGE_SLACK = 0.5  # how far off its box's edge an arrow's end may sit, for rounding


def on_edge(point: Point, box: Box) -> bool:
    """Whether `point` lies on the edge of `box`."""
    return box.grown(EDGE_SLACK).holds(point) and not box.grown(-EDGE_SLACK).holds(point)


def labels(drawing: Layout) -> list[Box]:
    """Every label box in a drawing."""
    return [drawn.label.box for drawn in (*drawing.arrows, *drawing.entries) if drawn.label is not None]


def test_rows_put_the_start_states_first_and_the_final_ones_last() -> None:
    grouped = rows(SESSION)
    assert set(grouped[0]) == {"pending", "idle"}
    assert grouped[-1] == list(SESSION.final)


def test_drawn_moves_draw_a_move_its_whole_frame_shares_once() -> None:
    cancels = [move for move, _ in drawn_moves(SESSION) if move.trigger == "Cancel"]
    assert len(cancels) == 1


def test_drawn_moves_cover_every_declared_move() -> None:
    for machine in MACHINES:
        framed = frames(machine)
        covered = set()
        for move, frame in drawn_moves(machine):
            sources = framed[frame] if frame else (move.source,)
            covered |= {Move(source, move.target, move.trigger) for source in sources}
        assert covered == set(machine.moves), machine.name


def test_point_on_runs_from_a_curve_start_to_its_end() -> None:
    curve = ((0.0, 0.0), (10.0, 0.0), (20.0, 10.0), (30.0, 10.0))
    assert point_on(curve, 0) == (0.0, 0.0)
    assert point_on(curve, 1) == (30.0, 10.0)


@pytest.mark.parametrize(("machine", "labelled"), CASES, ids=IDS)
def test_layout_keeps_states_apart(machine: Machine, labelled: bool) -> None:
    drawing = layout(machine, labelled)
    for first, second in itertools.combinations(drawing.states.values(), 2):
        assert not first.overlaps(second)


@pytest.mark.parametrize(("machine", "labelled"), CASES, ids=IDS)
def test_layout_frames_hold_their_states(machine: Machine, labelled: bool) -> None:
    drawing = layout(machine, labelled)
    for name, states in frames(machine).items():
        for state in states:
            assert drawing.frames[name].overlap(drawing.states[state]) == drawing.states[state].overlap(
                drawing.states[state]
            ), (name, state)


@pytest.mark.parametrize(("machine", "labelled"), CASES, ids=IDS)
def test_layout_runs_each_arrow_from_its_source_to_its_target(machine: Machine, labelled: bool) -> None:
    drawing = layout(machine, labelled)
    for arrow in drawing.arrows:
        source = drawing.frames[arrow.frame] if arrow.frame else drawing.states[arrow.move.source]
        assert on_edge(arrow.curve[0], source), arrow.move
        assert on_edge(arrow.curve[-1], drawing.states[arrow.move.target]), arrow.move


@pytest.mark.parametrize("machine", MACHINES, ids=[machine.name for machine in MACHINES])
def test_layout_keeps_labels_off_states_and_each_other(machine: Machine) -> None:
    drawing = layout(machine, labelled=True)
    placed = labels(drawing)
    assert len(placed) == len(drawing.arrows) + len(drawing.entries)
    for label in placed:
        assert not any(label.overlaps(state) for state in drawing.states.values())
    for first, second in itertools.combinations(placed, 2):
        assert not first.overlaps(second)
