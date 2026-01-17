import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


class ConfigLoader:
    """
    Loads YAML config and substitutes ${VAR} with environment variables.

    Example:
        config = ConfigLoader()
        db_url = config.get('database.url')  # Gets from arkos.yaml
        port = config.get('app.port', default=8080)
    """

    def __init__(self, config_path: str | Path | None = None):
        """
        Initialize config loader.

        Args:
            config_path: Path to YAML config. If None, uses default location.
        """
        if config_path is None:
            # Find project root (parent of config_module)
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config_module" / "config.yaml"

        self.config_path = Path(config_path)
        self._config: dict[str, Any] | None = None

        # Check if config file exists
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}\nPlease create config_module/arkos.yaml"
            )

    def load(self) -> dict[str, Any]:
        """
        Load config file and substitute environment variables.

        Returns:
            Complete config dictionary
        """
        # Cache the loaded config
        if self._config is not None:
            return self._config

        # Load YAML
        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        # Substitute environment variables
        self._config = self._substitute_env_vars(config)
        return self._config

    def _substitute_env_vars(self, obj: Any) -> Any:
        """
        Recursively substitute ${VAR} patterns with os.environ['VAR'].

        Args:
            obj: Config value (dict, list, str, etc.)

        Returns:
            Same structure with variables substituted
        """
        if isinstance(obj, dict):
            return {key: self._substitute_env_vars(val) for key, val in obj.items()}

        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]

        elif isinstance(obj, str):
            # Match ${VAR} pattern
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
        """
        Get nested config value using dot notation.

        Args:
            key_path: Dot-separated path like 'llm.base_url' or 'app.port'
            default: Value to return if key not found

        Returns:
            Config value or default

        Example:
            >>> config.get('llm.base_url')
            'http://localhost:30000/v1'
            >>> config.get('app.port')
            1112
            >>> config.get('nonexistent.key', default=999)
            999
        """
        config = self.load()
        keys = key_path.split(".")
        value = config

        for key in keys:
            if isinstance(value, dict):
                next_value = value.get(key)
                if next_value is None:
                    return default
                value = next_value
            else:
                return default

        return value

    def reload(self) -> dict[str, Any]:
        """Force reload config from disk (useful for testing)."""
        self._config = None
        return self.load()


# Find project root and load .env
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"

# Load .env file - python-dotenv handles everything
load_dotenv(dotenv_path=env_path, override=False)

# Create global config instance
config = ConfigLoader()
