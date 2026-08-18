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

# Two secrets, two trust domains: SUPABASE_JWT_SECRET verifies a token somebody
# else issued, ARK_SESSION_SECRET signs the cookie we issue. `or` not setdefault,
# because setdefault treats an empty value from .env as already set.
if not os.environ.get("SUPABASE_JWT_SECRET"):
    os.environ["SUPABASE_JWT_SECRET"] = "test-supabase-secret-at-least-32-chars"
if not os.environ.get("ARK_SESSION_SECRET"):
    os.environ["ARK_SESSION_SECRET"] = "test-session-secret-at-least-32-chars"

