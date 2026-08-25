"""ConfigLoader in config_module/loader.py."""

from pathlib import Path

import pytest
import yaml

from config_module.loader import ConfigLoader


@pytest.fixture
def simple_config(tmp_path):
    """Create a simple YAML config file."""
    config_data = {
        "app": {"port": 8080, "host": "localhost", "debug": True},
        "llm": {"base_url": "http://localhost:30000/v1", "max_tokens": 1024},
        "database": {"name": "testdb"},
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config_data))
    return str(config_file)


@pytest.fixture
def env_var_config(tmp_path):
    """Create a config file with environment variable references."""
    config_data = {
        "database": {"url": "${DB_URL}", "password": "${DB_PASS}"},
        "app": {"name": "arkos"},
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config_data))
    return str(config_file)


@pytest.fixture
def nested_config(tmp_path):
    """Create a deeply nested config."""
    config_data = {
        "level1": {
            "level2": {
                "level3": {"value": "deep"},
            }
        }
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config_data))
    return str(config_file)


class TestConfigLoaderInit:
    def test_init_with_valid_path(self, simple_config):
        loader = ConfigLoader(simple_config)
        assert loader.config_path == Path(simple_config)

    def test_init_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ConfigLoader(str(tmp_path / "nonexistent.yaml"))

    def test_config_starts_unloaded(self, simple_config):
        loader = ConfigLoader(simple_config)
        assert loader._config is None


class TestConfigLoaderLoad:
    def test_load_returns_dict(self, simple_config):
        loader = ConfigLoader(simple_config)
        config = loader.load()
        assert isinstance(config, dict)

    def test_load_correct_values(self, simple_config):
        loader = ConfigLoader(simple_config)
        config = loader.load()
        assert config["app"]["port"] == 8080
        assert config["llm"]["base_url"] == "http://localhost:30000/v1"

    def test_load_caches_result(self, simple_config):
        loader = ConfigLoader(simple_config)
        first = loader.load()
        second = loader.load()
        assert first is second


class TestConfigLoaderGet:
    def test_get_top_level(self, simple_config):
        loader = ConfigLoader(simple_config)
        app = loader.get("app")
        assert isinstance(app, dict)
        assert app["port"] == 8080

    def test_get_nested_dot_notation(self, simple_config):
        loader = ConfigLoader(simple_config)
        assert loader.get("app.port") == 8080
        assert loader.get("llm.base_url") == "http://localhost:30000/v1"

    def test_get_missing_key_returns_default(self, simple_config):
        loader = ConfigLoader(simple_config)
        assert loader.get("nonexistent") is None
        assert loader.get("nonexistent", default=42) == 42

    def test_get_missing_nested_key_returns_default(self, simple_config):
        loader = ConfigLoader(simple_config)
        assert loader.get("app.nonexistent") is None
        assert loader.get("app.nonexistent", default="fallback") == "fallback"

    def test_get_deeply_nested(self, nested_config):
        loader = ConfigLoader(nested_config)
        assert loader.get("level1.level2.level3.value") == "deep"

    def test_get_partially_missing_path(self, nested_config):
        loader = ConfigLoader(nested_config)
        assert loader.get("level1.level2.missing.value") is None


class TestConfigLoaderEnvSubstitution:
    def test_env_var_substitution(self, env_var_config, monkeypatch):
        monkeypatch.setenv("DB_URL", "postgresql://localhost/test")
        monkeypatch.setenv("DB_PASS", "secret123")
        loader = ConfigLoader(env_var_config)
        config = loader.load()
        assert config["database"]["url"] == "postgresql://localhost/test"
        assert config["database"]["password"] == "secret123"

    def test_missing_env_var_raises(self, env_var_config, monkeypatch):
        monkeypatch.delenv("DB_URL", raising=False)
        monkeypatch.delenv("DB_PASS", raising=False)
        loader = ConfigLoader(env_var_config)
        with pytest.raises(EnvironmentError, match="Environment variable"):
            loader.load()

    def test_non_env_strings_unchanged(self, simple_config):
        loader = ConfigLoader(simple_config)
        config = loader.load()
        assert config["app"]["host"] == "localhost"

    def test_env_var_in_list(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ITEM_VAL", "resolved")
        config_data = {"items": ["${ITEM_VAL}", "static"]}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))
        loader = ConfigLoader(str(config_file))
        config = loader.load()
        assert config["items"] == ["resolved", "static"]


class TestConfigLoaderReload:
    def test_reload_clears_cache(self, tmp_path):
        config_data = {"version": 1}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))
        loader = ConfigLoader(str(config_file))

        assert loader.load()["version"] == 1

        config_file.write_text(yaml.dump({"version": 2}))
        reloaded = loader.reload()
        assert reloaded["version"] == 2

    def test_reload_returns_new_dict(self, simple_config):
        loader = ConfigLoader(simple_config)
        first = loader.load()
        reloaded = loader.reload()
        assert first is not reloaded


class TestCoherence:
    """Settings that are each valid and wrong together, caught at startup."""

    def _loader(self, tmp_path, **overrides) -> ConfigLoader:
        data = {
            "leases": {"wait_timeout_s": 90},
            "tools": {"call_timeout_s": 120},
            "sandbox": {"max_concurrent_per_user": 5},
            "quotas": {"max_unattended_sessions": 5},
        }
        for dotted, value in overrides.items():
            section, key = dotted.split("__")
            data[section][key] = value
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(data))
        return ConfigLoader(str(config_file))

    def test_coherent_settings_pass(self, tmp_path):
        self._loader(tmp_path).assert_coherent()

    def test_a_wait_that_outlives_its_call_is_refused(self, tmp_path):
        with pytest.raises(RuntimeError, match="never ran"):
            self._loader(tmp_path, leases__wait_timeout_s=120).assert_coherent()

    def test_a_wait_with_too_little_margin_is_refused(self, tmp_path):
        with pytest.raises(RuntimeError, match="wait_timeout_s"):
            self._loader(tmp_path, leases__wait_timeout_s=115).assert_coherent()

    def test_a_cap_below_the_session_quota_is_refused(self, tmp_path):
        with pytest.raises(RuntimeError, match="could never get a computer"):
            self._loader(tmp_path, sandbox__max_concurrent_per_user=3).assert_coherent()

    def test_the_shipped_config_is_coherent(self):
        """The defaults are the ones that ship, so they are the ones that must agree."""
        ConfigLoader().assert_coherent()
