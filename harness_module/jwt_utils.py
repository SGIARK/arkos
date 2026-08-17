"""JWT verification for the API.

INTERIM. `docs/auth.md` replaces this wholesale in Task 4: the frontend logs in
with supabase-js, we verify that JWT ONCE and set our own httpOnly + Secure +
SameSite=Lax session cookie, and every later request is authenticated by the
cookie the browser attaches automatically. Until that lands there is no HTTP
server, so nothing imports this module.

What this file no longer does, per auth.md's "Deleted" list: mint its own tokens,
honour an `X-User-ID` header, or recognise a demo mode. Token ISSUANCE is
Supabase's job; we only ever verify.
"""

from __future__ import annotations

import os
from typing import Any

import jwt  # PyJWT
from fastapi import Depends, Header, HTTPException, status

_SECRET = os.environ.get("ARK_JWT_SECRET") or None
_ALG = "HS256"


def assert_secure_secret() -> None:
    """Raise unless a real signing secret is configured. Called at startup."""
    if not _SECRET:
        raise RuntimeError(
            "ARK_JWT_SECRET is unset. Refusing to start: without it every token "
            "is unverifiable. There is no demo bypass."
        )


def decode_token(token: str) -> dict[str, Any]:
    """Decode and verify a JWT, raising jwt.PyJWTError if invalid or expired."""
    if not _SECRET:
        raise jwt.InvalidKeyError("ARK_JWT_SECRET is unset")
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
) -> dict[str, Any]:
    """Resolve the caller to {"user_id", "username"} from the Bearer token, or raise 401."""
    token = _extract_bearer(authorization)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing Authorization: Bearer <token>",
        )
    try:
        payload = decode_token(token)
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"invalid token: {e}") from e
    return {"user_id": payload["sub"], "username": payload.get("username") or "anon"}


CurrentUser = Depends(get_current_user)
