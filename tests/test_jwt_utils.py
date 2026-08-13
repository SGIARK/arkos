"""Tests for harness_module/jwt_utils.py — JWT issue/decode and the auth dependency."""

from __future__ import annotations

import time
import uuid
from unittest.mock import patch

import jwt
import pytest
from fastapi import HTTPException

from harness_module.jwt_utils import (
    _extract_bearer,
    decode_token,
    get_current_user,
    issue_token,
)


class TestIssueToken:
    def test_issues_string_token(self):
        token = issue_token("user-1", "alice")
        assert isinstance(token, str)
        # JWT is three base64-ish chunks separated by dots
        assert token.count(".") == 2

    def test_payload_round_trips_through_decode(self):
        token = issue_token("user-1", "alice")
        payload = decode_token(token)
        assert payload["sub"] == "user-1"
        assert payload["username"] == "alice"
        assert "iat" in payload
        assert "exp" in payload

    def test_accepts_uuid_for_user_id(self):
        u = uuid.uuid4()
        token = issue_token(u, "bob")
        payload = decode_token(token)
        assert payload["sub"] == str(u)

    def test_exp_in_future(self):
        token = issue_token("u", "n")
        payload = decode_token(token)
        assert payload["exp"] > int(time.time())


class TestDecodeToken:
    def test_rejects_token_signed_with_wrong_secret(self):
        bad = jwt.encode({"sub": "x"}, "wrong-secret", algorithm="HS256")
        with pytest.raises(jwt.PyJWTError):
            decode_token(bad)

    def test_rejects_expired_token(self):
        # 1970 + 1 second iat; exp also in the past
        with patch("harness_module.jwt_utils._SECRET", "ark-dev-secret-change-me"):
            past = jwt.encode(
                {"sub": "u", "username": "n", "iat": 1, "exp": 100},
                "ark-dev-secret-change-me",
                algorithm="HS256",
            )
            with pytest.raises(jwt.ExpiredSignatureError):
                decode_token(past)

    def test_rejects_garbage(self):
        with pytest.raises(jwt.PyJWTError):
            decode_token("not.a.token")


class TestExtractBearer:
    def test_returns_token_from_valid_header(self):
        assert _extract_bearer("Bearer abc.def.ghi") == "abc.def.ghi"

    def test_case_insensitive_scheme(self):
        assert _extract_bearer("bearer abc") == "abc"
        assert _extract_bearer("BEARER abc") == "abc"

    def test_returns_none_for_missing_header(self):
        assert _extract_bearer(None) is None
        assert _extract_bearer("") is None

    def test_returns_none_for_wrong_scheme(self):
        assert _extract_bearer("Basic abc") is None
        assert _extract_bearer("Token abc") is None

    def test_returns_none_for_malformed_header(self):
        # Single token, no scheme
        assert _extract_bearer("just-a-string") is None
        # Bearer with empty token
        assert _extract_bearer("Bearer ") is None


class TestGetCurrentUser:
    @pytest.mark.asyncio
    async def test_accepts_valid_bearer_token(self):
        token = issue_token("u-1", "alice")
        result = await get_current_user(authorization=f"Bearer {token}", x_user_id=None)
        assert result == {"user_id": "u-1", "username": "alice"}

    @pytest.mark.asyncio
    async def test_falls_back_to_x_user_id_header(self):
        # Backwards-compat path: no Bearer, but X-User-ID is set.
        result = await get_current_user(authorization=None, x_user_id="legacy-id")
        assert result == {"user_id": "legacy-id", "username": "legacy-id"}

    @pytest.mark.asyncio
    async def test_bearer_takes_precedence_over_x_user_id(self):
        token = issue_token("real-user", "alice")
        result = await get_current_user(
            authorization=f"Bearer {token}",
            x_user_id="should-be-ignored",
        )
        assert result["user_id"] == "real-user"

    @pytest.mark.asyncio
    async def test_rejects_missing_credentials(self):
        with pytest.raises(HTTPException) as excinfo:
            await get_current_user(authorization=None, x_user_id=None)
        assert excinfo.value.status_code == 401

    @pytest.mark.asyncio
    async def test_rejects_invalid_bearer_token(self):
        with pytest.raises(HTTPException) as excinfo:
            await get_current_user(authorization="Bearer garbage", x_user_id=None)
        assert excinfo.value.status_code == 401
        assert "invalid token" in excinfo.value.detail

    @pytest.mark.asyncio
    async def test_username_defaults_to_anon_when_missing(self):
        # Forge a token with no username field
        from harness_module.jwt_utils import _SECRET

        token = jwt.encode(
            {"sub": "u", "iat": int(time.time()), "exp": int(time.time()) + 60},
            _SECRET,
            algorithm="HS256",
        )
        result = await get_current_user(authorization=f"Bearer {token}", x_user_id=None)
        assert result["username"] == "anon"
