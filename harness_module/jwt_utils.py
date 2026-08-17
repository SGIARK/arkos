"""Identity: verifies a Supabase JWT once, then carries a session cookie of our own.

`SUPABASE_JWT_SECRET` verifies tokens issued by Supabase; `ARK_SESSION_SECRET` signs the
session cookie, minted from `POST /auth/session`.

The cookie is httpOnly, and the browser attaches it to `EventSource` requests, so SSE
carries no stream token.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt  # PyJWT

from config_module.loader import config

_ALG = "HS256"

# Issuer claim on the cookies minted here; `read_session` requires it.
_ISSUER = "arkos"


def _secret(name: str) -> str | None:
    value = os.environ.get(name)
    return value or None


def assert_secure_secrets() -> None:
    """Raises unless both signing secrets are present in the environment."""
    missing = [name for name in ("SUPABASE_JWT_SECRET", "ARK_SESSION_SECRET") if not _secret(name)]
    if missing:
        raise RuntimeError(
            f"{', '.join(missing)} unset. Refusing to start: without them tokens are "
            "unverifiable and sessions unsignable. See .env.example."
        )


# --- verifying somebody else's token ------------------------------------------


def verify_supabase(token: str) -> dict[str, Any]:
    """Verifies a Supabase access token and returns its claims.

    Raises:
        jwt.PyJWTError: invalid signature, expired, or wrong audience.
    """
    secret = _secret("SUPABASE_JWT_SECRET")
    if not secret:
        raise jwt.InvalidKeyError("SUPABASE_JWT_SECRET is unset")
    # Supabase stamps aud=authenticated, and PyJWT rejects a token whose audience the
    # caller did not name.
    return jwt.decode(
        token,
        secret,
        algorithms=[_ALG],
        audience=config.get("auth.jwt_audience") or "authenticated",
    )


def extract_bearer(authorization: str | None) -> str | None:
    """Pulls the token out of an `Authorization: Bearer <token>` header."""
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


# --- minting and reading our own cookie ---------------------------------------


def mint_session(user_id: str, email: str | None = None) -> str:
    """Signs a session cookie for an already-verified user.

    Called from `POST /auth/session`, once `verify_supabase` has returned.
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
    """Verifies a session cookie and returns its claims.

    Raises:
        jwt.PyJWTError: invalid signature, expired, or issued by someone else.
    """
    secret = _secret("ARK_SESSION_SECRET")
    if not secret:
        raise jwt.InvalidKeyError("ARK_SESSION_SECRET is unset")
    return jwt.decode(cookie, secret, algorithms=[_ALG], issuer=_ISSUER, options={"require": ["sub", "exp"]})
