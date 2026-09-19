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
                f"Config file not found: {self.config_path}\nPlease create config_module/config.yaml"
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

    def _coherence_number(self, key: str, missing: list[str]) -> float:
        """One number a coherence rule compares, recording the key if it is not there.

        A RULE ABOUT A KEY NOBODY SETS IS NOT A RULE. `self.get(key) or 0` reads an
        absent key as zero, and every rule here is written to skip a zero, so a rule
        whose keys have been renamed or deleted goes on passing and says nothing.
        """
        value = self.get(key)
        if value is None:
            missing.append(key)
            return 0.0
        return float(value)

    def assert_coherent(self) -> None:
        """Raise RuntimeError on settings that are each valid and wrong together.

        AND ON A RULE THAT HAS LOST ITS SUBJECT. Every key these comparisons read
        must exist, because the alternative is a check that cannot fail, which is
        the same artifact as a test that cannot fail and is harder to notice: it is
        the fail-closed control, so its silence reads as health.
        """
        problems = []
        missing: list[str] = []

        waiting = self._coherence_number("leases.wait_timeout_s", missing)
        call = self._coherence_number("tools.call_timeout_s", missing)
        if waiting + _WAIT_MARGIN_S > call:
            problems.append(
                f"leases.wait_timeout_s ({waiting}) leaves less than {_WAIT_MARGIN_S}s of "
                f"tools.call_timeout_s ({call}): a contended call would be cut off as a tool "
                "timeout instead of reporting that it never ran"
            )

        boxes = int(self._coherence_number("sandbox.max_concurrent_per_user", missing))
        sessions = int(self._coherence_number("quotas.max_unattended_sessions", missing))
        if boxes < sessions:
            problems.append(
                f"sandbox.max_concurrent_per_user ({boxes}) is below "
                f"quotas.max_unattended_sessions ({sessions}): {sessions - boxes} session(s) the "
                "quota permits could never get a computer"
            )

        # Imported here, not at module scope: `registry` reads this loader, so a
        # top-level import of the tool registry would be a cycle.
        from tool_module.registry import local_tools

        ours = len(local_tools())
        cap = int(self._coherence_number("llm.max_tools", missing))
        if cap and ours >= cap:
            problems.append(
                f"llm.max_tools ({cap}) is not above the {ours} tools we author ourselves: "
                "the session's own allowance would be zero or negative, so no connected "
                "service could ever be reached and the meter would read out of a budget of nothing"
            )

        asked = self._coherence_number("browser.wall_clock_s", missing)
        forced = self._coherence_number("browser.hard_timeout_s", missing)
        if asked and forced <= asked:
            problems.append(
                f"browser.hard_timeout_s ({forced}) is not above browser.wall_clock_s ({asked}): "
                "the backstop would fire before the graceful stop could return partial results"
            )

        if missing:
            problems.append(
                f"a coherence rule reads {sorted(set(missing))}, which config.yaml does not define: "
                "an absent key reads as zero and every rule here skips a zero, so the check would "
                "pass without comparing anything"
            )
        if problems:
            raise RuntimeError("incoherent configuration: " + "; ".join(problems))


# Seconds a contended call must have left after its wait for a lease gives up.
_WAIT_MARGIN_S = 10

project_root = Path(__file__).parent.parent
env_path = project_root / ".env"

load_dotenv(dotenv_path=env_path, override=False)

config = ConfigLoader()


def cfg(key: str, default: Any = None) -> Any:
    """Read one config value, falling back to `default`.

    A plain function, not the bound method, so `import cfg as _cfg` keeps each
    module's attribute a separate monkeypatch seam.
    """
    return config.get(key, default)
