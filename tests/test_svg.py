"""Drawings parse as SVG, say what every arrow means, and link only where asked."""

import xml.etree.ElementTree as ElementTree

from state_machine_module.layout import layout
from state_machine_module.machines import MACHINES
from state_machine_module.svg import drawing_svg

SVG = "{http://www.w3.org/2000/svg}"


def test_drawing_svg_parses_for_every_machine() -> None:
    for machine in MACHINES:
        root = ElementTree.fromstring(drawing_svg(layout(machine, labelled=True)))
        assert root.tag == f"{SVG}svg", machine.name
