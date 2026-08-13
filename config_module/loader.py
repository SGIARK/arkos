import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


class ConfigLoader:
    """Load YAML config and substitute ${VAR} with environment variables."""

    def __init__(self, config_path: str | None = None):
        """Initialize the loader, defaulting to config_module/config.yaml."""
        if config_path is None:
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config_module" / "config.yaml"

        self.config_path = Path(config_path)
        self._config: dict[str, Any] | None = None

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}\nPlease create config_module/arkos.yaml"
            )

    def load(self) -> dict[str, Any]:
        """Load and cache the config file with environment variables substituted."""
        if self._config is not None:
            return self._config

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        self._config = self._substitute_env_vars(config)
        return self._config

    def _substitute_env_vars(self, obj: Any) -> Any:
        """Recursively substitute ${VAR} patterns in `obj` with os.environ values."""
        if isinstance(obj, dict):
            return {key: self._substitute_env_vars(val) for key, val in obj.items()}

        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]

        elif isinstance(obj, str):
            pattern = r"\$\{([^}]+)\}"

            def replace_var(match):
                var_name = match.group(1)
                var_value = os.environ.get(var_name)

                if var_value is None:
                    raise OSError(
                        f"Environment variable '{var_name}' not found.\n"
                        f"Required by: {self.config_path}\n"
                        f"Please set it in .env file or export it."
                    )

                return var_value

            return re.sub(pattern, replace_var, obj)

        else:
            return obj

    def get(self, key_path: str, default: Any = None) -> Any:
        """Return the value at a dot-separated `key_path`, or `default` if absent."""
        config = self.load()
        keys = key_path.split(".")
        value = config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default

        return value

    def validate_required(self, required_keys: list[str]) -> None:
        """Raise RuntimeError if any of the given dot-notation keys is missing or None."""
        missing = [k for k in required_keys if self.get(k) is None]
        if missing:
            raise RuntimeError(f"Missing required config keys (check config.yaml and .env): {missing}")

    def reload(self) -> dict[str, Any]:
        """Discard the cache and reload the config from disk."""
        self._config = None
        return self.load()


project_root = Path(__file__).parent.parent
env_path = project_root / ".env"

load_dotenv(dotenv_path=env_path, override=False)

config = ConfigLoader()
