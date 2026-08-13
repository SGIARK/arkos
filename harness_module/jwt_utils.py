"""
Minimal JWT helpers for demo-style auth.

Token payload:
    {
        "sub": "<user uuid>",
        "username": "<username>",
        "iat": <unix>,
        "exp": <unix>,
    }

Secret is read from env var ARK_JWT_SECRET, with a dev fallback.
Swap `DEMO_MODE_NO_PASSWORD` semantics later without changing this file.
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
_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days, plenty for demos (see UNSAFE_DECISIONS U7)


def _demo_mode() -> bool:
    """True if ARK_DEMO_MODE is set. Gates the X-User-ID fallback (UNSAFE_DECISIONS U2)."""
    return os.environ.get("ARK_DEMO_MODE", "").strip().lower() in ("1", "true", "yes", "on")


def assert_secure_secret() -> None:
    """
    Fail-fast at startup: refuse to boot with the built-in default JWT secret
    unless explicitly in demo mode. A default secret means forgeable tokens.
    """
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
    """Decode and verify a JWT. Raises jwt.PyJWTError on invalid/expired."""
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
    """
    FastAPI dependency. Returns {"user_id": str, "username": str} or raises 401.

    A valid Bearer token always wins. The X-User-ID header is honored ONLY in
    demo mode (ARK_DEMO_MODE) -- otherwise a missing/invalid token is a 401.
    See UNSAFE_DECISIONS U2.
    """
    token = _extract_bearer(authorization)
    if token:
        try:
            payload = decode_token(token)
        except jwt.PyJWTError as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"invalid token: {e}") from e
        return {"user_id": payload["sub"], "username": payload.get("username") or "anon"}

    if x_user_id and _demo_mode():
        # Demo-only legacy pass-through. Forgeable -- never enable in prod.
        return {"user_id": x_user_id, "username": x_user_id}

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="missing Authorization: Bearer <token>",
    )


CurrentUser = Depends(get_current_user)
