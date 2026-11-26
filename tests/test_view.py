"""The pages draw every declared machine, and every box links to the right page."""

from state_machine_module.machines import SESSION
from state_machine_module.view import (
    drawn_moves,
    quoted,
)


def test_quoted_escapes_quotes_and_keeps_line_breaks() -> None:
    assert quoted('a "b"\nc') == '"a \\"b\\"\\nc"'


def test_drawn_moves_draw_a_move_its_whole_frame_shares_once() -> None:
    cancels = [move for move, _ in drawn_moves(SESSION) if move.trigger == "Cancel"]
    assert len(cancels) == 1
