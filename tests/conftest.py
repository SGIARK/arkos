"""Shared fixtures for arkos tests."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Fallback DSN only; tests truncate tables, so this must never be a real database.
TEST_DB_URL = "postgresql://test:test@localhost:5432/test"

# Captured BEFORE load_dotenv: a DB_URL already in the environment (CI's service
# container, or `DB_URL=... pytest`) wins; one that only .env supplies is discarded.
# `override=False` alone is not enough: .env points at PRODUCTION on a dev machine and would fill an unset DB_URL.
_explicit_db_url = os.environ.get("DB_URL")

# Real .env first: the loader hard-fails on any unset ${VAR} in config.yaml.
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=False)
except Exception:
    pass

# Hard-forced, not setdefault: whatever .env just set is overwritten.
os.environ["DB_URL"] = _explicit_db_url or TEST_DB_URL

# Every ${VAR} config.yaml interpolates must be set, or the loader raises during
# collection and the whole suite fails before a single test runs.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-key")
os.environ.setdefault("COMPOSIO_API_KEY", "comp-test-dummy-key")
os.environ.setdefault("SERPAPI_API_KEY", "serp-test-dummy-key")

# `or`, not setdefault: setdefault treats an empty value from .env as already set.
if not os.environ.get("SUPABASE_JWT_SECRET"):
    os.environ["SUPABASE_JWT_SECRET"] = "test-supabase-secret-at-least-32-chars"
if not os.environ.get("ARKOS_SESSION_SECRET"):
    os.environ["ARKOS_SESSION_SECRET"] = "test-session-secret-at-least-32-chars"


@pytest.fixture(autouse=True)
def _fresh_auth_window():
    """Give every test its own sliding window for `POST /auth/session`.

    The limiter (12.2.5) is per source address, and every test shares one — so
    without this the suite signs in past the ceiling and starts 429ing partway
    through, which looks like a bug in whatever test happened to be running.
    The limit stays ON; `test_the_sign_in_endpoint_is_rate_limited` exercises it.
    """
    from harness_module import api

    api._auth_hits.clear()
    yield
    api._auth_hits.clear()
