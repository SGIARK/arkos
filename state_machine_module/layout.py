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

from dataclasses import dataclass

from state_machine_module.machines import Move

Point = tuple[float, float]
Curve = tuple[Point, Point, Point, Point]


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
