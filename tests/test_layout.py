"""Layouts keep states, frames and labels clear of each other, and every arrow meets its states."""

from state_machine_module.layout import drawn_moves, point_on
from state_machine_module.machines import SESSION


def test_drawn_moves_draw_a_move_its_whole_frame_shares_once() -> None:
    cancels = [move for move, _ in drawn_moves(SESSION) if move.trigger == "Cancel"]
    assert len(cancels) == 1


def test_point_on_runs_from_a_curve_start_to_its_end() -> None:
    curve = ((0.0, 0.0), (10.0, 0.0), (20.0, 10.0), (30.0, 10.0))
    assert point_on(curve, 0) == (0.0, 0.0)
    assert point_on(curve, 1) == (30.0, 10.0)
