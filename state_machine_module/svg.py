"""Draw a Layout as SVG: states, frames, arrows and labels, each state a link where asked.

Hovering over a state or an arrow shows what it means through SVG's own <title>,
and a link is a plain <a>, so the drawing needs no script. Colours and fonts come
from the page's stylesheet.
"""

from __future__ import annotations

import html

from state_machine_module.layout import (
    DOT_RADIUS,
    LABEL_PAD,
    LINE_HEIGHT,
    STATE_FONT,
    Arrow,
    Box,
    Curve,
    Entry,
    Label,
    Layout,
)

CORNER = 9
FRAME_CORNER = 12
LABEL_CORNER = 4
FINAL_INSET = 4  # the gap between a final state's two borders
BASELINE = 0.75  # of a line's height: where its text sits on it
STATE_BASELINE = 0.35  # of the state font, below a box's middle, to centre its name
FRAME_NAME_DROP = 13  # from a frame's top edge to the baseline of its name
MACHINE_PAD = 16  # inside a machine's frame on the overview
MACHINE_TITLE = 30  # room for a machine's name at the top of its frame

MARKER = (
    '<defs><marker id="arrowhead" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"'
    ' orient="auto"><path class="head" d="M0,0 L10,5 L0,10 z"/></marker></defs>'
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


def frame_svg(name: str, box: Box) -> str:
    """A frame round states that move alike, its name in its top corner."""
    return (
        f'<g class="frame">{rect(box, FRAME_CORNER)}<text x="{number(box.x + FRAME_CORNER)}"'
        f' y="{number(box.y + FRAME_NAME_DROP)}">{html.escape(name)}</text></g>'
    )


def arrow_svg(arrow: Arrow) -> str:
    """An arrow, with a wide invisible stroke so that hovering near it shows its trigger."""
    data = path_data(arrow.curve)
    return (
        f'<g class="move"><title>{html.escape(arrow.move.trigger)}</title><path class="hit" d="{data}"/>'
        f'<path class="line" d="{data}" marker-end="url(#arrowhead)"/></g>'
    )


def entry_svg(entry: Entry) -> str:
    """A start dot and its arrow into the state the machine begins in."""
    x, y = entry.dot
    return (
        f'<g class="entry"><title>{html.escape(entry.trigger)}</title>'
        f'<circle cx="{number(x)}" cy="{number(y)}" r="{DOT_RADIUS}"/>'
        f'<path class="line" d="{path_data(entry.curve)}" marker-end="url(#arrowhead)"/></g>'
    )


def machine_svg(drawing: Layout, link: str = "") -> str:
    """Every part of one machine, frames first and states last, so each draws over what it should.
    With `link`, every state is a link to it, and hovering over one says what the machine tracks."""
    machine = drawing.machine
    labels = [drawn.label for drawn in (*drawing.arrows, *drawing.entries) if drawn.label is not None]
    parts = [frame_svg(name, box) for name, box in drawing.frames.items()]
    parts += [arrow_svg(arrow) for arrow in drawing.arrows]
    parts += [entry_svg(entry) for entry in drawing.entries]
    parts += [label_svg(label) for label in labels]
    parts += [
        state_svg(name, box, name in machine.final, link, machine.about if link else name)
        for name, box in drawing.states.items()
    ]
    return "".join(parts)


def document(body: str, bounds: Box, title: str) -> str:
    """A whole <svg> element the size of `bounds`, holding the arrowhead every arrow uses."""
    view = " ".join(number(value) for value in (bounds.x, bounds.y, bounds.width, bounds.height))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view}" width="{number(bounds.width)}"'
        f' height="{number(bounds.height)}" aria-label="{html.escape(title)}">{MARKER}{body}</svg>'
    )


def drawing_svg(drawing: Layout) -> str:
    """One machine's drawing, for its own page."""
    return document(machine_svg(drawing), drawing.bounds, f"The {drawing.machine.name} state machine")


def machine_frame_svg(name: str, box: Box, link: str) -> str:
    """A machine's dashed frame on the overview, its name a link to its page."""
    return (
        f'<g class="machine">{rect(box, FRAME_CORNER)}<a href="{html.escape(link)}"><text'
        f' x="{number(box.x + MACHINE_PAD)}" y="{number(box.y + MACHINE_TITLE - MACHINE_PAD / 2)}">'
        f"{html.escape(name)}</text></a></g>"
    )
