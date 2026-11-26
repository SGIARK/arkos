"""Each rule flags the world it describes, and passes a world that keeps it."""

from state_machine_module.rules import broken
from state_machine_module.world import World


def broken_names(world: World) -> set[str]:
    return {rule.name for rule in broken(world)}
