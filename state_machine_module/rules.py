"""What arkos must never do, judged one world at a time.

The checker evaluates every rule on every world it reaches. Rules that need two
sessions at once, or races inside a single step, are not modeled yet.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from state_machine_module.world import World


@dataclass(frozen=True, slots=True)
class Rule:
    """A named rule, the sentence it stands for, and the test that it holds."""

    name: str
    says: str
    holds: Callable[[World], bool]
