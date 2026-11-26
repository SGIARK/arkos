"""Draw a Layout as SVG: states, frames, arrows and labels, each state a link where asked.

Hovering over a state or an arrow shows what it means through SVG's own <title>,
and a link is a plain <a>, so the drawing needs no script. Colours and fonts come
from the page's stylesheet.
"""

from __future__ import annotations

import html

from state_machine_module.layout import (
    LABEL_PAD,
    LINE_HEIGHT,
    STATE_FONT,
    Box,
    Curve,
    Label,
)

CORNER = 9
LABEL_CORNER = 4
FINAL_INSET = 4  # the gap between a final state's two borders
BASELINE = 0.75  # of a line's height: where its text sits on it
STATE_BASELINE = 0.35  # of the state font, below a box's middle, to centre its name


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


def label_svg(label: Label) -> str:
    """A label's lines on a pale ground, so the arrows behind it stay out of the way."""
    box = label.box
    middle, first = number(box.x + box.width / 2), box.y + LABEL_PAD + BASELINE * LINE_HEIGHT
    lines = "".join(
        f'<tspan x="{middle}" y="{number(first + row * LINE_HEIGHT)}">{html.escape(line)}</tspan>'
        for row, line in enumerate(label.lines)
    )
    return f'<g class="label">{rect(box, LABEL_CORNER)}<text>{lines}</text></g>'


def state_svg(name: str, box: Box, final: bool, link: str, hint: str) -> str:
    """A state: a rounded box with its name, a second border if the machine can end there, and a link
    if one is given. `hint` shows on hover."""
    inner = rect(box.grown(-FINAL_INSET), CORNER - FINAL_INSET) if final else ""
    x, y = box.middle
    body = (
        f'<g class="state"><title>{html.escape(hint)}</title>{rect(box, CORNER)}{inner}'
        f'<text x="{number(x)}" y="{number(y + STATE_FONT * STATE_BASELINE)}">{html.escape(name)}</text></g>'
    )
    return f'<a href="{html.escape(link)}">{body}</a>' if link else body
