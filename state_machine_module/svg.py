"""Draw a Layout as SVG: states, frames, arrows and labels, each state a link where asked.

Hovering over a state or an arrow shows what it means through SVG's own <title>,
and a link is a plain <a>, so the drawing needs no script. Colours and fonts come
from the page's stylesheet.
"""

from __future__ import annotations

from state_machine_module.layout import (
    Box,
)


def number(value: float) -> str:
    """A coordinate, to a tenth of a pixel."""
    return f"{value:.1f}"


def rect(box: Box, corner: float) -> str:
    """A rounded rectangle filling `box`."""
    return (
        f'<rect x="{number(box.x)}" y="{number(box.y)}" width="{number(box.width)}"'
        f' height="{number(box.height)}" rx="{corner}"/>'
    )
