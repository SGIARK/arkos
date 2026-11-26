"""Layouts keep states, frames and labels clear of each other, and every arrow meets its states."""

import itertools

import pytest

from state_machine_module.layout import drawn_moves, frames, layout, point_on, rows
from state_machine_module.machines import MACHINES, SESSION, Machine, Move

CASES = [(machine, labelled) for machine in MACHINES for labelled in (True, False)]
IDS = [f"{machine.name}-{'labelled' if labelled else 'bare'}" for machine, labelled in CASES]


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
