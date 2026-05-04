import importlib
import pathlib

STATE_REGISTRY = {}


def auto_register_states():
    """Discover all agent_* subfolders under state_module/ and import every
    state_*.py file found inside them. This triggers @register_state decorators
    and populates STATE_REGISTRY. Adding a new agent folder named agent_<name>/
    is all that is needed for its states to be picked up automatically."""
    pkg_root = pathlib.Path(__file__).parent.parent  # state_module/
    for folder in sorted(pkg_root.iterdir()):
        if folder.is_dir() and folder.name.startswith("agent_"):
            for py_file in sorted(folder.glob("state_*.py")):
                module_name = py_file.stem
                importlib.import_module(f"state_module.{folder.name}.{module_name}")


def register_state(cls):
    """Decorator that registers a State subclass in STATE_REGISTRY by its `type` attribute."""
    state_type = getattr(cls, "type", None)
    print(f"Registering state: {cls.type}")
    if not state_type:
        raise ValueError(f"State class {cls.__name__} must have a `type` attribute.")
    STATE_REGISTRY[state_type] = cls
    return cls
