"""Draw a Layout as SVG: states, frames, arrows and labels, each state a link where asked.

Hovering over a state or an arrow shows what it means through SVG's own <title>,
and a link is a plain <a>, so the drawing needs no script. Colours and fonts come
from the page's stylesheet.
"""

from __future__ import annotations

from state_machine_module.layout import (
    Box,
    Curve,
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


def path_data(curve: Curve) -> str:
    """A cubic curve as SVG path data."""
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = curve
    return f"M{number(x0)},{number(y0)} C{number(x1)},{number(y1)} {number(x2)},{number(y2)} {number(x3)},{number(y3)}"
