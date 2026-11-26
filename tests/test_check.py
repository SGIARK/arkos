"""The checker finds what today's arkos breaks, with paths that reproduce it."""

import pytest

from state_machine_module.check import Report, explore
from state_machine_module.world import ROOTS

# The longest of today's shortest counterexamples is four events.
DEPTH = 5


@pytest.fixture(scope="module")
def report() -> Report:
    return explore(ROOTS, DEPTH)
