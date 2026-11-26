"""Where each part of a machine's drawing goes, worked out in plain Python.

A machine is drawn in rows. The states it starts in head the first row, each
other live state sits one row below the nearest state that moves to it, and the
states it can end in share the last row. Each row takes the order that tangles
its arrows least. An arrow between neighbouring rows is an S-curve, one that
would cross a row bends round it, and one within a row runs straight across.
Arrows that meet the same side of a state spread along it, so two opposite moves
run side by side. Sizes come from the length of each name, so no font is
measured. Nothing here draws: `svg.py` turns a Layout into markup.
"""

from __future__ import annotations

import itertools
import math
import textwrap
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass

from state_machine_module.machines import Machine, Move

Point = tuple[float, float]
Curve = tuple[Point, Point, Point, Point]

STATE_FONT = 13
LABEL_FONT = 11
CHAR_WIDTH = 0.6  # of the font size: a generous average for a sans-serif face
STATE_HEIGHT = 34
STATE_PAD = 16  # either side of a state's name
MIN_STATE_WIDTH = 64
LABELLED_MIN_STATE_WIDTH = 110  # wide enough for the arrows meeting a state to fan out
LINE_HEIGHT = 14
LABEL_PAD = 4
LABEL_CHARS = 26  # characters on each line of a label
COLUMN_GAP = 40
LABELLED_COLUMN_GAP = 190  # room between neighbours for the labels of the arrows that join them
ROW_GAP = 44
LABELLED_ROW_GAP = 160
FRAME_PAD = 14
FRAME_TITLE = 16  # room for a frame's name above its states
SIDE_GAP = 36  # how far an arrow bending round other rows clears them
ARC_RISE = 40  # how far an arrow over or under its row bulges from it
FRAME_PORT_INSET = 24  # how near a frame's corner an arrow may leave it
SPREAD_WEIGHT = 2.0  # cost of each state between two states an arrow joins within a row
CROSSING_WEIGHT = 3.0  # cost of two arrows crossing between rows
ORDER_LIMIT = 40_000  # orderings few enough to try them all; a bigger machine keeps its declared order


@dataclass(frozen=True, slots=True)
class Box:
    """A rectangle, by its top-left corner and its size."""

    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def middle(self) -> Point:
        return (self.x + self.width / 2, self.y + self.height / 2)

    def grown(self, by: float) -> Box:
        """This box with `by` added on every side; a negative `by` shrinks it."""
        return Box(self.x - by, self.y - by, self.width + 2 * by, self.height + 2 * by)

    def overlaps(self, other: Box) -> bool:
        """Whether the two boxes share any area."""
        return self.x < other.right and other.x < self.right and self.y < other.bottom and other.y < self.bottom

    def overlap(self, other: Box) -> float:
        """How much area the two boxes share."""
        wide = min(self.right, other.right) - max(self.x, other.x)
        high = min(self.bottom, other.bottom) - max(self.y, other.y)
        return max(wide, 0) * max(high, 0)

    def holds(self, point: Point) -> bool:
        """Whether `point` lies in this box or on its edge."""
        return self.x <= point[0] <= self.right and self.y <= point[1] <= self.bottom


@dataclass(frozen=True, slots=True)
class Label:
    """A trigger broken into lines, and the box they are written in."""

    lines: tuple[str, ...]
    box: Box


@dataclass(frozen=True, slots=True)
class Arrow:
    """One arrow as drawn: the move it stands for, the frame it speaks for if any, its curve and label."""

    move: Move
    frame: str
    curve: Curve
    label: Label | None


@dataclass(frozen=True, slots=True)
class Entry:
    """Where a machine begins: a dot, its arrow into the state, and what puts the machine there."""

    trigger: str
    dot: Point
    curve: Curve
    label: Label | None


@dataclass(frozen=True, slots=True)
class Leg:
    """An arrow before its ends are pinned: the box and side it leaves, the box and side it enters,
    and for an arrow bent round other rows, the line it runs along."""

    move: Move
    frame: str
    source: str  # a state, or a frame written [name]
    leaves: str  # top, bottom, left or right
    target: str
    enters: str
    bend: float | None = None


@dataclass(frozen=True, slots=True)
class Layout:
    """A machine measured for drawing: its states, frames, entries and arrows, and the box round them all."""

    machine: Machine
    states: dict[str, Box]
    frames: dict[str, Box]
    entries: tuple[Entry, ...]
    arrows: tuple[Arrow, ...]
    bounds: Box


def text_width(text: str, font: float) -> float:
    """About how wide `text` sets in a sans-serif face `font` pixels high."""
    return len(text) * font * CHAR_WIDTH


def wrapped(text: str) -> tuple[str, ...]:
    """A trigger broken into lines short enough for a label."""
    return tuple(textwrap.wrap(text, LABEL_CHARS)) or ("",)


def enclosing(boxes: Iterable[Box]) -> Box:
    """The smallest box round all of `boxes`."""
    every = list(boxes)
    left, top = min(box.x for box in every), min(box.y for box in every)
    right, bottom = max(box.right for box in every), max(box.bottom for box in every)
    return Box(left, top, right - left, bottom - top)


def point_on(curve: Curve, share: float) -> Point:
    """The point `share` of the way along a cubic curve: 0 at its start, 1 at its end."""
    rest = 1 - share
    weights = (rest**3, 3 * rest * rest * share, 3 * rest * share * share, share**3)
    x = sum(weight * point[0] for weight, point in zip(weights, curve, strict=True))
    y = sum(weight * point[1] for weight, point in zip(weights, curve, strict=True))
    return (x, y)


def frames(machine: Machine) -> dict[str, tuple[str, ...]]:
    """The states drawn framed together: those the machine ends in, and the rest, if two or more each."""
    if not machine.final:
        return {}
    live = tuple(state for state in machine.states if state not in machine.final)
    return {name: states for name, states in (("live", live), ("ended", machine.final)) if len(states) > 1}


def shared_moves(machine: Machine, states: tuple[str, ...]) -> list[Move]:
    """The moves every one of `states` makes alike, to a state outside them: one Move each, from the first."""
    return [
        move
        for move in machine.moves
        if move.source == states[0]
        and move.target not in states
        and all(Move(state, move.target, move.trigger) in machine.moves for state in states)
    ]


def drawn_moves(machine: Machine) -> list[tuple[Move, str]]:
    """Each arrow to draw, with the frame it leaves from: a move a whole frame shares is drawn once."""
    arrows: list[tuple[Move, str]] = []
    covered: set[Move] = set()
    for name, states in frames(machine).items():
        for move in shared_moves(machine, states):
            arrows.append((move, name))
            covered |= {Move(state, move.target, move.trigger) for state in states}
    return arrows + [(move, "") for move in machine.moves if move not in covered]


def rows(machine: Machine) -> list[list[str]]:
    """The states by row: where the machine starts, each live state one row below its nearest
    predecessor, live states nothing reaches after those, and last the states it ends in."""
    live = [state for state in machine.states if state not in machine.final]
    depth = {start.state: 0 for start in machine.starts if start.state in live}
    queue = deque(depth)
    while queue:
        state = queue.popleft()
        for move in machine.moves:
            if move.source == state and move.target in live and move.target not in depth:
                depth[move.target] = depth[state] + 1
                queue.append(move.target)
    levels = [
        [state for state in live if depth.get(state) == level] for level in range(max(depth.values(), default=-1) + 1)
    ]
    unreached = [state for state in live if state not in depth]
    return [row for row in (*levels, unreached, list(machine.final)) if row]


def tangle(order: list[tuple[str, ...]], pairs: list[tuple[str, str]]) -> float:
    """How tangled an ordering draws: arrows that cross between rows, how far each arrow leans, and
    states standing between two that an arrow joins within a row."""
    row_of = {state: number for number, row in enumerate(order) for state in row}
    place_of = {state: index for row in order for index, state in enumerate(row)}
    across = {state: (place_of[state] + 0.5) / len(order[row_of[state]]) for state in row_of}
    cost, spans = 0.0, []
    for first, second in pairs:
        if row_of[first] == row_of[second]:
            cost += SPREAD_WEIGHT * (abs(place_of[first] - place_of[second]) - 1)
            continue
        cost += abs(across[first] - across[second])
        upper, lower = sorted((first, second), key=row_of.__getitem__)
        if row_of[lower] - row_of[upper] == 1:
            spans.append((row_of[upper], across[upper], across[lower]))
    for (row, upper, lower), (other_row, other_upper, other_lower) in itertools.combinations(spans, 2):
        if row == other_row and (upper - other_upper) * (lower - other_lower) < 0:
            cost += CROSSING_WEIGHT
    return cost


def ordered(machine: Machine, grouped: list[list[str]]) -> list[tuple[str, ...]]:
    """Each row in the order that tangles the arrows least. The declared order wins a tie, and a
    machine with too many orderings to try keeps it."""
    pairs = sorted({tuple(sorted((move.source, move.target))) for move in machine.moves if move.source != move.target})
    options = [list(itertools.permutations(row)) for row in grouped]
    if math.prod(len(choices) for choices in options) > ORDER_LIMIT:
        return [tuple(row) for row in grouped]
    return list(min(itertools.product(*options), key=lambda order: tangle(list(order), pairs)))


def row_tops(order: list[tuple[str, ...]], framed: dict[str, tuple[str, ...]], labelled: bool) -> list[float]:
    """How far down each row starts. Where one frame ends and another begins, rows stand further apart."""
    gap = LABELLED_ROW_GAP if labelled else ROW_GAP
    frame_of = {state: name for name, states in framed.items() for state in states}
    tops = [0.0]
    for above, row in itertools.pairwise(order):
        parted = frame_of.get(above[0]) != frame_of.get(row[0])
        tops.append(tops[-1] + STATE_HEIGHT + gap + (2 * FRAME_PAD + FRAME_TITLE if parted else 0))
    return tops


def place(order: list[tuple[str, ...]], tops: list[float], labelled: bool) -> dict[str, Box]:
    """A box for every state, sized to its name, each row centred under the widest."""
    gap = LABELLED_COLUMN_GAP if labelled else COLUMN_GAP
    least = LABELLED_MIN_STATE_WIDTH if labelled else MIN_STATE_WIDTH
    width = {state: max(text_width(state, STATE_FONT) + 2 * STATE_PAD, least) for row in order for state in row}
    spans = [sum(width[state] for state in row) + gap * (len(row) - 1) for row in order]
    boxes = {}
    for row, top, span in zip(order, tops, spans, strict=True):
        x = (max(spans) - span) / 2
        for state in row:
            boxes[state] = Box(x, top, width[state], STATE_HEIGHT)
            x += width[state] + gap
    return boxes


def frame_boxes(machine: Machine, boxes: dict[str, Box]) -> dict[str, Box]:
    """A frame round each group of framed states, with room for the group's name at its top."""
    found = {}
    for name, states in frames(machine).items():
        inner = enclosing(boxes[state] for state in states).grown(FRAME_PAD)
        found[name] = Box(inner.x, inner.y - FRAME_TITLE, inner.width, inner.height + FRAME_TITLE)
    return found


def leg(move: Move, frame: str, boxes: dict[str, Box], row_of: dict[str, int], order: list[tuple[str, ...]]) -> Leg:
    """How an arrow runs: down or up between rows, across its row, or round the rows it would cross."""
    source = f"[{frame}]" if frame else move.source
    if frame:
        down = boxes[move.target].y > boxes[source].bottom
    else:
        rise = row_of[move.target] - row_of[move.source]
        if rise == 0:
            return across(move, boxes, order[row_of[move.source]])
        if abs(rise) > 1:
            return around(move, boxes, row_of, order)
        down = rise > 0
    return Leg(move, frame, source, "bottom" if down else "top", move.target, "top" if down else "bottom")


def across(move: Move, boxes: dict[str, Box], row: tuple[str, ...]) -> Leg:
    """An arrow within a row: straight across to a neighbour, or arched over or under the states between."""
    rightward = boxes[move.target].x > boxes[move.source].x
    if abs(row.index(move.target) - row.index(move.source)) > 1:
        side = "top" if rightward else "bottom"
        return Leg(move, "", move.source, side, move.target, side)
    leaves, enters = ("right", "left") if rightward else ("left", "right")
    return Leg(move, "", move.source, leaves, move.target, enters)


def around(move: Move, boxes: dict[str, Box], row_of: dict[str, int], order: list[tuple[str, ...]]) -> Leg:
    """An arrow that would cross other rows, bent round them on whichever side is nearer."""
    low, high = sorted((row_of[move.source], row_of[move.target]))
    passed = [boxes[state] for row in order[low : high + 1] for state in row]
    right = max(box.right for box in passed) + SIDE_GAP
    left = min(box.x for box in passed) - SIDE_GAP
    ends = (boxes[move.source], boxes[move.target])
    if sum(right - box.right for box in ends) <= sum(box.x - left for box in ends):
        return Leg(move, "", move.source, "right", move.target, "right", right)
    return Leg(move, "", move.source, "left", move.target, "left", left)


def on_side(box: Box, side: str, share: float) -> Point:
    """The point `share` of the way along one side of `box`, left to right or top to bottom."""
    if side == "top":
        return (box.x + box.width * share, box.y)
    if side == "bottom":
        return (box.x + box.width * share, box.bottom)
    if side == "left":
        return (box.x, box.y + box.height * share)
    return (box.right, box.y + box.height * share)


def pin(box: Box, side: str, share: float, other: Box, framed: bool) -> Point:
    """Where an arrow meets `box`: `share` of the way along the side, or for a frame, level with `other`."""
    if framed and side in ("top", "bottom"):
        inset = min(FRAME_PORT_INSET, box.width / 2)
        x = min(max(other.middle[0], box.x + inset), box.right - inset)
        return (x, box.y if side == "top" else box.bottom)
    return on_side(box, side, share)


def pins(legs: list[Leg], boxes: dict[str, Box], starts: list[str]) -> dict[tuple[int, str], Point]:
    """Where each arrow meets its boxes, keyed (arrow number, "from" or "to"); the arrow from the
    nth start dot is keyed (-1 - n, "to"). Arrows meeting one side of a box spread along it in the
    order of their other ends, so they do not cross where they meet."""
    sides: dict[tuple[str, str], list[tuple[tuple[float, int], tuple[int, str], str]]] = defaultdict(list)
    for number, drawn in enumerate(legs):
        flag = 0 if drawn.leaves in ("bottom", "right") else 1
        ends = ((drawn.source, drawn.leaves, drawn.target, "from"), (drawn.target, drawn.enters, drawn.source, "to"))
        for key, side, other, end in ends:
            along = boxes[other].middle[0 if side in ("top", "bottom") else 1]
            sides[(key, side)].append(((along, flag), (number, end), other))
    for number, state in enumerate(starts):
        sides[(state, "top")].append(((boxes[state].middle[0], -1), (-1 - number, "to"), state))
    pinned = {}
    for (key, side), attached in sides.items():
        attached.sort(key=lambda entry: entry[0])
        for slot, (_, end, other) in enumerate(attached, 1):
            share = slot / (len(attached) + 1)
            pinned[end] = pin(boxes[key], side, share, boxes[other], key.startswith("["))
    return pinned


def curve_of(drawn: Leg, start: Point, end: Point) -> Curve:
    """An arrow's curve: bent round other rows, straight across a row, arched over or under it, or an
    S between rows."""
    (x0, y0), (x3, y3) = start, end
    if drawn.bend is not None:
        return (start, (drawn.bend, y0), (drawn.bend, y3), end)
    if drawn.leaves in ("left", "right"):
        third = (x3 - x0) / 3
        return (start, (x0 + third, y0), (x3 - third, y3), end)
    if drawn.leaves == drawn.enters:
        rise = -ARC_RISE if drawn.leaves == "top" else ARC_RISE
        return (start, (x0, y0 + rise), (x3, y3 + rise), end)
    half = (y3 - y0) / 2
    return (start, (x0, y0 + half), (x3, y3 - half), end)


def label_box(lines: tuple[str, ...], centre: Point) -> Box:
    """The box `lines` fill, centred on `centre`."""
    width = max(text_width(line, LABEL_FONT) for line in lines) + 2 * LABEL_PAD
    height = len(lines) * LINE_HEIGHT + 2 * LABEL_PAD
    return Box(centre[0] - width / 2, centre[1] - height / 2, width, height)
