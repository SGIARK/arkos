"""Shared fixtures for arkos tests."""

import os
import sys

# Ensure project root is on sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Real config first, placeholders second. The DB-touching tests need the actual
# DB_URL; everything else only needs the var to exist, because
# config_module.loader hard-fails on any unset ${VAR} in config.yaml.
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=False)
except Exception:
    pass

os.environ.setdefault("DB_URL", "postgresql://test:test@localhost:5432/test")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-key")
os.environ.setdefault("SMITHERY_API_KEY", "sk-test-smithery-key")
os.environ.setdefault("SMITHERY_NAMESPACE", "arkos-test")
