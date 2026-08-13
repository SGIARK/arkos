"""Minimal JWT helpers for demo-style auth.

Payload carries sub, username, iat and exp; the secret comes from ARK_JWT_SECRET.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import jwt  # PyJWT
from fastapi import Depends, Header, HTTPException, status

_DEFAULT_SECRET = "ark-dev-secret-change-me"
_SECRET = os.environ.get("ARK_JWT_SECRET", _DEFAULT_SECRET)
_ALG = "HS256"
_TTL_SECONDS = 60 * 60 * 24 * 30


def _demo_mode() -> bool:
    """Return True if ARK_DEMO_MODE is set, which gates the X-User-ID fallback."""
    return os.environ.get("ARK_DEMO_MODE", "").strip().lower() in ("1", "true", "yes", "on")


def assert_secure_secret() -> None:
    """Raise unless a real ARK_JWT_SECRET is set, since the default is forgeable."""
    if _SECRET == _DEFAULT_SECRET and not _demo_mode():
        raise RuntimeError(
            "ARK_JWT_SECRET is the built-in default. Set a real secret, or set "
            "ARK_DEMO_MODE=1 for local dev. Refusing to start with forgeable tokens."
        )


def issue_token(user_id: str | uuid.UUID, username: str) -> str:
    """Issue a signed JWT for the given user."""
    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "username": username,
        "iat": now,
        "exp": now + _TTL_SECONDS,
    }
    return jwt.encode(payload, _SECRET, algorithm=_ALG)


def decode_token(token: str) -> dict[str, Any]:
    """Decode and verify a JWT, raising jwt.PyJWTError if invalid or expired."""
    return jwt.decode(token, _SECRET, algorithms=[_ALG])


def _extract_bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


async def get_current_user(
    authorization: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
) -> dict[str, Any]:
    """Resolve the caller to {"user_id", "username"} from the Bearer token, or raise 401."""
    token = _extract_bearer(authorization)
    if token:
        try:
            payload = decode_token(token)
        except jwt.PyJWTError as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"invalid token: {e}") from e
        return {"user_id": payload["sub"], "username": payload.get("username") or "anon"}

    if x_user_id and _demo_mode():
        # Forgeable pass-through, gated on demo mode; never enable in prod.
        return {"user_id": x_user_id, "username": x_user_id}

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="missing Authorization: Bearer <token>",
    )


CurrentUser = Depends(get_current_user)
