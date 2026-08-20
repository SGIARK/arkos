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

    def assert_coherent(self) -> None:
        """Raise RuntimeError on settings that are each valid and wrong together.

        Run at startup, because the failures they cause are invisible at the
        point of the mistake: a wait that outlives the call it is inside times
        out as the wrong thing, and a cap below the session quota starves
        sessions the quota promised.
        """
        problems = []

        waiting = float(self.get("leases.wait_timeout_s") or 0)
        call = float(self.get("tools.call_timeout_s") or 0)
        if waiting + _WAIT_MARGIN_S > call:
            problems.append(
                f"leases.wait_timeout_s ({waiting}) leaves less than {_WAIT_MARGIN_S}s of "
                f"tools.call_timeout_s ({call}): a contended call would be cut off as a tool "
                "timeout instead of reporting that it never ran"
            )

        boxes = int(self.get("sandbox.max_concurrent_per_user") or 0)
        sessions = int(self.get("quotas.max_unattended_sessions") or 0)
        if boxes < sessions:
            problems.append(
                f"sandbox.max_concurrent_per_user ({boxes}) is below "
                f"quotas.max_unattended_sessions ({sessions}): {sessions - boxes} session(s) the "
                "quota permits could never get a computer"
            )

        # Imported here, not at module scope: `registry` reads this loader, and a
        # config module that imports the tool registry at import time would be a
        # cycle. By the time anything calls this, both are loaded.
        from tool_module.registry import local_tools

        ours = len(local_tools())
        cap = int(self.get("llm.max_tools") or 0)
        if cap and ours >= cap:
            problems.append(
                f"llm.max_tools ({cap}) is not above the {ours} tools we author ourselves: "
                "the session's own allowance would be zero or negative, so no connected "
                "service could ever be reached and the meter would read out of a budget of nothing"
            )

        asked = float(self.get("browser.wall_clock_s") or 0)
        forced = float(self.get("browser.hard_timeout_s") or 0)
        if asked and forced <= asked:
            problems.append(
                f"browser.hard_timeout_s ({forced}) is not above browser.wall_clock_s ({asked}): "
                "the backstop would fire before the graceful stop could return partial results"
            )

        if problems:
            raise RuntimeError("incoherent configuration: " + "; ".join(problems))


# Seconds a contended call must have left after its wait gives up, so the tool
# returns its own answer rather than being cut off mid-report.
_WAIT_MARGIN_S = 10

project_root = Path(__file__).parent.parent
env_path = project_root / ".env"

load_dotenv(dotenv_path=env_path, override=False)

config = ConfigLoader()
