"""Tests for the config loader module."""

import os
import tempfile

from config_module.loader import ConfigLoader


class TestConfigLoader:
    """Test cases for ConfigLoader."""

    def test_yaml_loading(self, sample_config_yaml):
        """Test that YAML config file loads successfully."""
        loader = ConfigLoader(sample_config_yaml)
        loaded_config = loader.load()

        assert loaded_config is not None
        assert isinstance(loaded_config, dict)

    def test_environment_variable_substitution(self):
        """Test that ${VAR} patterns are replaced with environment variables."""
        # Set up test environment variables
        os.environ["TEST_DB_URL"] = "postgresql://testuser:testpass@localhost:5432/testdb"
        os.environ["TEST_API_KEY"] = "test-api-key-12345"
        os.environ["TEST_PORT"] = "8080"

        # Create a temporary config file with env vars
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
test:
  db_url: ${TEST_DB_URL}
  api_key: ${TEST_API_KEY}
  port: ${TEST_PORT}
  static_value: "no_substitution"
""")
            temp_config_path = f.name

        try:
            test_loader = ConfigLoader(temp_config_path)

            assert (
                test_loader.get("test.db_url")
                == "postgresql://testuser:testpass@localhost:5432/testdb"
            )
            assert test_loader.get("test.api_key") == "test-api-key-12345"
            assert test_loader.get("test.port") == "8080"
            assert test_loader.get("test.static_value") == "no_substitution"
        finally:
            os.unlink(temp_config_path)

    def test_nested_config_access(self, sample_config_yaml):
        """Test accessing nested config values with dot notation."""
        loader = ConfigLoader(sample_config_yaml)

        assert loader.get("app.host") == "0.0.0.0"
        assert loader.get("app.port") == 1112
        assert loader.get("llm.base_url") == "http://localhost:30000/v1"
        assert loader.get("memory.user_id") == "test_user"

    def test_default_values(self, sample_config_yaml):
        """Test that get() returns None for missing keys."""
        loader = ConfigLoader(sample_config_yaml)

        result = loader.get("nonexistent.key")
        assert result is None

        result = loader.get("nonexistent.key", default="custom_default")
        assert result == "custom_default"

    def test_missing_env_var_handling(self):
        """Test handling of missing environment variables."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
test:
  missing_var: ${DEFINITELY_MISSING_VAR}
""")
            temp_config_path = f.name

        try:
            test_loader = ConfigLoader(temp_config_path)
            # Should return the original ${VAR} syntax if not found
            result = test_loader.get("test.missing_var")
            assert result is not None
        finally:
            os.unlink(temp_config_path)
