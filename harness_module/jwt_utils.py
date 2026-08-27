"""Identity: verifies a Supabase JWT once, then carries a session cookie of our own.

The cookie is httpOnly, so the browser attaches it to `EventSource` and SSE needs
no stream token of its own.
"""

from __future__ import annotations

import asyncio
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
    """Refuses to start without a way to sign sessions and a way to verify tokens."""
    if not _secret("ARK_SESSION_SECRET"):
        raise RuntimeError(
            "ARK_SESSION_SECRET is unset. Refusing to start: sessions would be unsignable. See .env.example."
        )
    if not jwks_url() and not _secret("SUPABASE_JWT_SECRET"):
        raise RuntimeError(
            "No way to verify a Supabase token. Set SUPABASE_URL, or a Supabase database.url to derive "
            "it from, so signing keys can be fetched; or SUPABASE_JWT_SECRET for a project still signing "
            "with a shared secret. See .env.example."
        )


# --- verifying somebody else's token ------------------------------------------


# Supabase signs asymmetrically via the project JWKS; HS256 is the older
# shared-secret scheme some projects still use.
_ASYMMETRIC = ("ES256", "RS256", "EdDSA")

_jwks_client: Any = None


def jwks_url() -> str | None:
    """Where the project publishes the keys its tokens are signed with."""
    from harness_module import blobs

    project = blobs.project_url()
    return f"{project}/auth/v1/.well-known/jwks.json" if project else None


def _jwks() -> Any:
    """The JWKS client; caches keys and refetches on an unknown `kid`."""
    global _jwks_client
    if _jwks_client is None:
        url = jwks_url()
        if url is None:
            return None
        _jwks_client = jwt.PyJWKClient(url, cache_keys=True)
    return _jwks_client


def reset_jwks() -> None:
    """Drop the cached JWKS client, for tests and key rotation."""
    global _jwks_client
    _jwks_client = None


def verify_supabase(token: str) -> dict[str, Any]:
    """Verifies a Supabase access token and returns its claims.

    The header's `alg` picks the key, and only from what this deployment has;
    anything else is refused. `audience` must be passed or PyJWT rejects the token.
    """
    audience = config.get("auth.jwt_audience") or "authenticated"
    algorithm = jwt.get_unverified_header(token).get("alg", "")

    if algorithm in _ASYMMETRIC:
        client = _jwks()
        if client is None:
            raise jwt.InvalidKeyError(
                f"the token is signed with {algorithm}, and no project URL is configured to fetch keys from"
            )
        return jwt.decode(token, client.get_signing_key_from_jwt(token).key, algorithms=[algorithm], audience=audience)

    if algorithm == _ALG:
        secret = _secret("SUPABASE_JWT_SECRET")
        if not secret:
            raise jwt.InvalidKeyError("the token is signed with HS256, and SUPABASE_JWT_SECRET is unset")
        return jwt.decode(token, secret, algorithms=[_ALG], audience=audience)

    raise jwt.InvalidAlgorithmError(f"tokens signed with {algorithm!r} are not accepted")


async def verify_supabase_off_loop(token: str) -> dict[str, Any]:
    """`verify_supabase`, run off the event loop.

    The asymmetric path fetches JWKS with blocking urllib, and a warm cache does
    not retire it: PyJWT refetches on any unknown `kid`.
    """
    return await asyncio.to_thread(verify_supabase, token)


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
    """Signs a session cookie for a user `verify_supabase` has already cleared."""
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
    """Verifies a session cookie and returns its claims."""
    secret = _secret("ARK_SESSION_SECRET")
    if not secret:
        raise jwt.InvalidKeyError("ARK_SESSION_SECRET is unset")
    return jwt.decode(cookie, secret, algorithms=[_ALG], issuer=_ISSUER, options={"require": ["sub", "exp"]})
