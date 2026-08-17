"""Shared fixtures for arkos tests."""

import os
import sys

# Ensure project root is on sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Real .env first: the loader hard-fails on any unset ${VAR} in config.yaml.
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=False)
except Exception:
    pass

# Every ${VAR} config.yaml interpolates must be set, or the loader raises during
# collection and the whole suite fails before a single test runs. Adding a server
# under mcp_servers: with a ${...} header means adding it here too.
os.environ.setdefault("DB_URL", "postgresql://test:test@localhost:5432/test")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-key")
os.environ.setdefault("SMITHERY_API_KEY", "sk-test-smithery-key")
os.environ.setdefault("SMITHERY_NAMESPACE", "arkos-test")
os.environ.setdefault("SLACK_BOT_TOKEN", "xoxb-test-token")

# jwt_utils reads ARK_JWT_SECRET at IMPORT time via os.environ.get(K, DEFAULT),
# so an empty value is used as the signing key instead of falling back and PyJWT
# rejects it. Force a usable one when .env left it blank; `or` not setdefault,
# because setdefault treats "" as already set.
if not os.environ.get("ARK_JWT_SECRET"):
    os.environ["ARK_JWT_SECRET"] = "test-secret-at-least-32-characters-long"
