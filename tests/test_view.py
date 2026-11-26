"""The pages draw every declared machine, and every box links to the right page."""

from state_machine_module.machines import LEASE, MACHINES, SESSION, Move
from state_machine_module.view import (
    drawn_moves,
    frames,
    legend,
    machine_dot,
    machine_page,
    overview_dot,
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


def test_machine_dot_double_borders_only_final_states() -> None:
    assert machine_dot(SESSION).count("peripheries=2") == len(SESSION.final)


def test_overview_dot_links_every_state_and_frame_to_its_page() -> None:
    source = overview_dot()
    for machine in MACHINES:
        assert source.count(f'href="/{machine.name}"') == len(machine.states) + 1, machine.name


def test_legend_leaves_out_what_a_drawing_does_not_use() -> None:
    assert "frame" in legend(SESSION)
    assert "frame" not in legend(LEASE) and "double border" not in legend(LEASE)


def test_machine_page_lists_every_start_and_move() -> None:
    page = machine_page(SESSION, svg="<svg></svg>")
    assert page.count("<tr>") == 1 + len(SESSION.starts) + len(SESSION.moves)
