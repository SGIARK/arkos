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
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from state_machine_module.machines import Machine, Move

Point = tuple[float, float]
Curve = tuple[Point, Point, Point, Point]
CHAR_WIDTH = 0.6  # of the font size: a generous average for a sans-serif face
STATE_HEIGHT = 34
LABEL_CHARS = 26  # characters on each line of a label
ROW_GAP = 44
LABELLED_ROW_GAP = 160
FRAME_PAD = 14
FRAME_TITLE = 16  # room for a frame's name above its states
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
