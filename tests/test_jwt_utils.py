"""Verifying somebody else's token, and minting our own cookie."""

import time

import jwt
import pytest

from harness_module import jwt_utils
from harness_module.jwt_utils import assert_secure_secrets, mint_session, read_session, verify_supabase

SUPABASE_SECRET = "test-supabase-secret-at-least-32-chars"


def _supabase(secret: str | None = None, **claims) -> str:
    """Mint a token shaped like the ones Supabase issues."""
    payload = {
        "sub": "8f1d4a02-0000-4000-8000-000000000001",
        "email": "a@example.com",
        "aud": "authenticated",
        "exp": int(time.time()) + 300,
        **claims,
    }
    return jwt.encode(payload, secret or SUPABASE_SECRET, algorithm="HS256")


class TestVerifySupabase:
    def test_round_trips_a_valid_token(self):
        claims = verify_supabase(_supabase())

        assert claims["sub"] == "8f1d4a02-0000-4000-8000-000000000001"
        assert claims["email"] == "a@example.com"

    def test_rejects_a_token_signed_with_another_secret(self):
        with pytest.raises(jwt.PyJWTError):
            verify_supabase(_supabase(secret="a-different-secret-entirely"))

    def test_rejects_an_expired_token(self):
        with pytest.raises(jwt.ExpiredSignatureError):
            verify_supabase(_supabase(exp=int(time.time()) - 1))

    def test_rejects_a_token_minted_for_another_audience(self):
        with pytest.raises(jwt.InvalidAudienceError):
            verify_supabase(_supabase(aud="someone-elses-app"))


class TestSessionCookie:
    def test_round_trips_what_we_signed(self):
        claims = read_session(mint_session("u-1", "a@example.com"))

        assert claims["sub"] == "u-1"
        assert claims["email"] == "a@example.com"

    def test_a_cookie_we_did_not_sign_is_refused(self):
        forged = jwt.encode({"sub": "u-1", "iss": "arkos"}, "not-our-secret", algorithm="HS256")

        with pytest.raises(jwt.PyJWTError):
            read_session(forged)

    def test_a_supabase_token_is_not_a_session_cookie(self):
        """A Supabase token is refused where a session cookie is expected."""
        with pytest.raises(jwt.PyJWTError):
            read_session(_supabase())

    def test_an_expired_cookie_is_refused(self, monkeypatch):
        monkeypatch.setattr(jwt_utils.config, "get", lambda key, default=None: -1)

        with pytest.raises(jwt.ExpiredSignatureError):
            read_session(mint_session("u-1"))


class TestAssertSecureSecrets:
    def test_raises_when_either_secret_is_unset(self, monkeypatch):
        for name in ("SUPABASE_JWT_SECRET", "ARK_SESSION_SECRET"):
            monkeypatch.delenv(name, raising=False)
            with pytest.raises(RuntimeError, match=name):
                assert_secure_secrets()
            monkeypatch.undo()

    def test_passes_when_both_are_set(self):
        assert assert_secure_secrets() is None

    def test_no_demo_bypass_exists(self, monkeypatch):
        """ARK_DEMO_MODE does not excuse a missing session secret."""
        monkeypatch.delenv("ARK_SESSION_SECRET", raising=False)
        monkeypatch.setenv("ARK_DEMO_MODE", "1")

        with pytest.raises(RuntimeError):
            assert_secure_secrets()


def test_extract_bearer_takes_only_a_bearer_scheme():
    assert jwt_utils.extract_bearer("Bearer abc") == "abc"
    assert jwt_utils.extract_bearer("Basic dXNlcjpwYXNz") is None
    assert jwt_utils.extract_bearer(None) is None
    assert jwt_utils.extract_bearer("Bearer ") is None
