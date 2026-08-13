#!/usr/bin/env python3
"""Standalone checks for the config loader: YAML load, ${VAR} substitution, get()."""

import os
import tempfile

os.environ["TEST_DB_URL"] = "postgresql://testuser:testpass@localhost:5432/testdb"
os.environ["TEST_API_KEY"] = "test-api-key-12345"
os.environ["TEST_PORT"] = "8080"

from loader import ConfigLoader, config


def test_yaml_loading():
    """Check that the YAML config file loads."""
    print("[OK] Testing YAML loading...")
    loaded_config = config.load()
    assert loaded_config is not None, "Config should be loaded"
    assert isinstance(loaded_config, dict), "Config should be a dictionary"
    print("  [OK] Config loaded successfully")


def test_environment_variable_substitution():
    """Check that ${VAR} patterns are replaced with environment variables."""
    print("\n[OK] Testing environment variable substitution...")

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

        assert test_loader.get("test.db_url") == "postgresql://testuser:testpass@localhost:5432/testdb", (
            "DB URL should be substituted"
        )
        print("  [OK] Database URL substituted correctly")

        assert test_loader.get("test.api_key") == "test-api-key-12345", "API key should be substituted"
        print("  [OK] API key substituted correctly")

        assert test_loader.get("test.port") == "8080", "Port should be substituted"
        print("  [OK] Port substituted correctly")

        assert test_loader.get("test.static_value") == "no_substitution", "Static values should remain unchanged"
        print("  [OK] Static values preserved correctly")

    finally:
        os.unlink(temp_config_path)


def test_nested_config_access():
    """Check nested config access via dot notation."""
    print("\n[OK] Testing nested config access...")

    host = config.get("app.host")
    assert host is not None, "app.host should exist"
    print(f"  [OK] app.host = {host}")

    port = config.get("app.port")
    assert port is not None, "app.port should exist"
    print(f"  [OK] app.port = {port}")

    llm_url = config.get("llm.base_url")
    assert llm_url is not None, "llm.base_url should exist"
    print(f"  [OK] llm.base_url = {llm_url}")

    user_id = config.get("memory.user_id")
    assert user_id is not None, "memory.user_id should exist"
    print(f"  [OK] memory.user_id = {user_id}")


def test_default_values():
    """Check that missing keys return None or the supplied default."""
    print("\n[OK] Testing default values for missing keys...")

    result = config.get("nonexistent.key")
    assert result is None, "Non-existent keys should return None"
    print("  [OK] Non-existent keys return None")

    result = config.get("nonexistent.key", default="custom_default")
    assert result == "custom_default", "Custom default should be returned"
    print("  [OK] Custom defaults work correctly")


def test_existing_config_values():
    """Check that values in the real config.yaml are readable."""
    print("\n[OK] Testing actual config.yaml values...")

    graph_path = config.get("state.graph_path")
    if graph_path:
        print(f"  [OK] state.graph_path = {graph_path}")

    system_prompt = config.get("app.system_prompt")
    if system_prompt:
        print(f"  [OK] app.system_prompt exists (length: {len(system_prompt)})")

    db_url = config.get("database.url")
    if db_url:
        print("  [OK] database.url configured")


def main():
    """Run every check and report the result."""
    print("=" * 60)
    print("Testing Config Loader (config_module/loader.py)")
    print("=" * 60)

    try:
        test_yaml_loading()
        test_environment_variable_substitution()
        test_nested_config_access()
        test_default_values()
        test_existing_config_values()

        print("\n" + "=" * 60)
        print("[PASS] All tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
