"""Drawings parse as SVG, say what every arrow means, and link only where asked."""

import xml.etree.ElementTree as ElementTree

from state_machine_module.layout import layout
from state_machine_module.machines import MACHINES, SESSION
from state_machine_module.svg import drawing_svg, overview_svg

SVG = "{http://www.w3.org/2000/svg}"


def test_drawing_svg_parses_for_every_machine() -> None:
    for machine in MACHINES:
        root = ElementTree.fromstring(drawing_svg(layout(machine, labelled=True)))
        assert root.tag == f"{SVG}svg", machine.name


def test_drawing_svg_titles_every_arrow_with_its_trigger() -> None:
    drawing = layout(SESSION, labelled=True)
    root = ElementTree.fromstring(drawing_svg(drawing))
    moves = [group for group in root.iter(f"{SVG}g") if group.get("class") == "move"]
    assert [group.find(f"{SVG}title").text for group in moves] == [arrow.move.trigger for arrow in drawing.arrows]


def test_drawing_svg_links_nothing_on_a_machines_own_page() -> None:
    assert "<a " not in drawing_svg(layout(SESSION, labelled=True))


def test_overview_svg_links_every_state_and_name_to_its_machine() -> None:
    drawings = [layout(machine, labelled=False) for machine in MACHINES]
    svg = overview_svg(drawings, [f"/{machine.name}" for machine in MACHINES])
    ElementTree.fromstring(svg)
    for machine in MACHINES:
        assert svg.count(f'href="/{machine.name}"') == len(machine.states) + 1, machine.name
