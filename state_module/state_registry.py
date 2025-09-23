import importlib
import pathlib

STATE_REGISTRY = {}


def auto_register_states(package_name: str):
    """
    Dynamically imports all Python modules in the given package to trigger decorators like @register_state.
    Example: auto_register_states("state_module")
    """
    pkg_path = pathlib.Path(__file__).parent
    for py_file in pkg_path.glob("state_*.py"):
        module_name = py_file.stem
        importlib.import_module(f"{package_name}.{module_name}")


def register_state(cls):
    state_type = getattr(cls, "type", None)
    print(f"Registering state: {cls.type}")
    if not state_type:
        raise ValueError(f"State class {cls.__name__} must have a `type` attribute.")
    STATE_REGISTRY[state_type] = cls
    return cls
