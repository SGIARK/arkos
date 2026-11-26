"""The pages draw every declared machine, and every box links to the right page."""

from state_machine_module.machines import MACHINES, SESSION, Move
from state_machine_module.view import (
    drawn_moves,
    frames,
    quoted,
)


def test_quoted_escapes_quotes_and_keeps_line_breaks() -> None:
    assert quoted('a "b"\nc') == '"a \\"b\\"\\nc"'


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
