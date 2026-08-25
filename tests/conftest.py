"""Shared fixtures for buddy tests."""

import os
import sys

# Ensure project root is on sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# The DSN the suite falls back to. Nothing here may ever touch a real database:
# tests truncate tables and write junk rows.
TEST_DB_URL = "postgresql://test:test@localhost:5432/test"

# Captured BEFORE .env is read, which is the whole point. A DB_URL already in the
# real process environment is a deliberate act — CI's service container, or
# `DB_URL=... pytest` — and it wins. A DB_URL that only appears after load_dotenv
# came from .env, which on a developer machine points at PRODUCTION, and it is
# discarded. `override=False` alone is not enough: it would let .env fill an
# unset DB_URL and silently aim the suite at the live database.
_explicit_db_url = os.environ.get("DB_URL")

# Real .env first: the loader hard-fails on any unset ${VAR} in config.yaml.
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=False)
except Exception:
    pass

# Hard-forced, not setdefault. Whatever .env just set is overwritten.
os.environ["DB_URL"] = _explicit_db_url or TEST_DB_URL

# Every ${VAR} config.yaml interpolates must be set, or the loader raises during
# collection and the whole suite fails before a single test runs.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-key")
os.environ.setdefault("ARCADE_API_KEY", "arc-test-dummy-key")

# Two secrets, two trust domains: SUPABASE_JWT_SECRET verifies a token somebody
# else issued, ARK_SESSION_SECRET signs the cookie we issue. `or` not setdefault,
# because setdefault treats an empty value from .env as already set.
if not os.environ.get("SUPABASE_JWT_SECRET"):
    os.environ["SUPABASE_JWT_SECRET"] = "test-supabase-secret-at-least-32-chars"
if not os.environ.get("ARK_SESSION_SECRET"):
    os.environ["ARK_SESSION_SECRET"] = "test-session-secret-at-least-32-chars"

