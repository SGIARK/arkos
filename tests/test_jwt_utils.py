"""JWT verification and the auth dependency in harness_module/jwt_utils.py.

Deliberately absent: tests for token issuance, an X-User-ID fallback, or a demo
mode. All three were deleted — see auth.md's "Deleted" list. A test asserting
that a forged identity is ACCEPTED is a test that defends the hole, so the only
thing pinned here is that it is refused.
"""

import time

import jwt
import pytest
from fastapi import HTTPException

from harness_module import jwt_utils
from harness_module.jwt_utils import assert_secure_secret, decode_token, get_current_user


def _token(secret: str = None, **claims) -> str:
    """Mint a token the way Supabase would, so these tests do not need issue_token."""
    payload = {"sub": "u-1", "username": "alice", "exp": int(time.time()) + 300, **claims}
    return jwt.encode(payload, secret or jwt_utils._SECRET, algorithm="HS256")


class TestDecodeToken:
    def test_round_trips_a_valid_token(self):
        payload = decode_token(_token())
        assert payload["sub"] == "u-1"
        assert payload["username"] == "alice"

    def test_rejects_token_signed_with_wrong_secret(self):
        with pytest.raises(jwt.PyJWTError):
            decode_token(_token(secret="a-different-secret-entirely"))

    def test_rejects_expired_token(self):
        with pytest.raises(jwt.ExpiredSignatureError):
            decode_token(_token(exp=int(time.time()) - 1))


class TestAssertSecureSecret:
    def test_raises_when_secret_unset(self, monkeypatch):
        monkeypatch.setattr(jwt_utils, "_SECRET", None)
        with pytest.raises(RuntimeError, match="ARK_JWT_SECRET"):
            assert_secure_secret()

    def test_passes_when_secret_set(self, monkeypatch):
        monkeypatch.setattr(jwt_utils, "_SECRET", "a-real-secret-thirty-two-chars-ok")
        assert_secure_secret() is None

    def test_no_demo_bypass_exists(self, monkeypatch):
        # The old code let ARK_DEMO_MODE excuse a missing secret. It must not.
        monkeypatch.setattr(jwt_utils, "_SECRET", None)
        monkeypatch.setenv("ARK_DEMO_MODE", "1")
        with pytest.raises(RuntimeError):
            assert_secure_secret()


class TestGetCurrentUser:
    @pytest.mark.asyncio
    async def test_accepts_valid_bearer_token(self):
        assert await get_current_user(authorization=f"Bearer {_token()}") == {
            "user_id": "u-1",
            "username": "alice",
        }

    @pytest.mark.asyncio
    async def test_username_defaults_to_anon_when_missing(self):
        result = await get_current_user(authorization=f"Bearer {_token(username=None)}")
        assert result["username"] == "anon"

    @pytest.mark.asyncio
    async def test_rejects_missing_header(self):
        with pytest.raises(HTTPException) as exc:
            await get_current_user(authorization=None)
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_rejects_garbage_token(self):
        with pytest.raises(HTTPException) as exc:
            await get_current_user(authorization="Bearer garbage")
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_rejects_non_bearer_scheme(self):
        with pytest.raises(HTTPException) as exc:
            await get_current_user(authorization="Basic dXNlcjpwYXNz")
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_x_user_id_header_is_not_a_credential(self, monkeypatch):
        """The deleted bypass, pinned shut: no header and no env var revives it."""
        monkeypatch.setenv("ARK_DEMO_MODE", "1")
        with pytest.raises(HTTPException) as exc:
            await get_current_user(authorization=None)
        assert exc.value.status_code == 401
