"""Layouts keep states, frames and labels clear of each other, and every arrow meets its states."""

from state_machine_module.layout import point_on


def test_point_on_runs_from_a_curve_start_to_its_end() -> None:
    curve = ((0.0, 0.0), (10.0, 0.0), (20.0, 10.0), (30.0, 10.0))
    assert point_on(curve, 0) == (0.0, 0.0)
    assert point_on(curve, 1) == (30.0, 10.0)
