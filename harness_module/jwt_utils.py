"""
Identity: verify Supabase's JWT once, then carry our own session cookie.

Two secrets for two trust domains. `SUPABASE_JWT_SECRET` only verifies tokens
issued elsewhere; `ARK_SESSION_SECRET` only signs the cookie we issue, and the
mint is reachable from `POST /auth/session` alone.

The cookie is httpOnly, so XSS cannot read it, and the browser attaches it to
`EventSource` requests, so SSE needs no stream token.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt  # PyJWT

from config_module.loader import config

_ALG = "HS256"

# Marks a cookie as ours, so a Supabase token in the cookie jar is not read as
# a session we issued.
_ISSUER = "arkos"


def _secret(name: str) -> str | None:
    value = os.environ.get(name)
    return value or None


def assert_secure_secrets() -> None:
    """Refuse to start without both signing secrets. There is no demo bypass."""
    missing = [name for name in ("SUPABASE_JWT_SECRET", "ARK_SESSION_SECRET") if not _secret(name)]
    if missing:
        raise RuntimeError(
            f"{', '.join(missing)} unset. Refusing to start: without them tokens are "
            "unverifiable and sessions unsignable. See .env.example."
        )


# --- verifying somebody else's token ------------------------------------------


def verify_supabase(token: str) -> dict[str, Any]:
    """
    Verify a Supabase access token and return its claims.

    Raises:
        jwt.PyJWTError: invalid signature, expired, or wrong audience.
    """
    secret = _secret("SUPABASE_JWT_SECRET")
    if not secret:
        raise jwt.InvalidKeyError("SUPABASE_JWT_SECRET is unset")
    # Supabase stamps aud=authenticated, and PyJWT rejects a token whose
    # audience the caller did not name.
    return jwt.decode(
        token,
        secret,
        algorithms=[_ALG],
        audience=config.get("auth.jwt_audience") or "authenticated",
    )


def extract_bearer(authorization: str | None) -> str | None:
    """Pull the token out of an `Authorization: Bearer <token>` header."""
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


# --- minting and reading our own cookie ---------------------------------------


def mint_session(user_id: str, email: str | None = None) -> str:
    """
    Sign a session cookie for an already-verified user.

    The only legitimate caller is `POST /auth/session`, after `verify_supabase`
    has returned.
    """
    secret = _secret("ARK_SESSION_SECRET")
    if not secret:
        raise RuntimeError("ARK_SESSION_SECRET is unset")
    now = datetime.now(UTC)
    return jwt.encode(
        {
            "sub": user_id,
            "email": email,
            "iss": _ISSUER,
            "iat": now,
            "exp": now + timedelta(seconds=int(config.get("auth.session_ttl_s") or 604800)),
        },
        secret,
        algorithm=_ALG,
    )


def read_session(cookie: str) -> dict[str, Any]:
    """
    Verify a session cookie and return its claims.

    Raises:
        jwt.PyJWTError: invalid signature, expired, or not one of ours.
    """
    secret = _secret("ARK_SESSION_SECRET")
    if not secret:
        raise jwt.InvalidKeyError("ARK_SESSION_SECRET is unset")
    return jwt.decode(cookie, secret, algorithms=[_ALG], issuer=_ISSUER, options={"require": ["sub", "exp"]})
