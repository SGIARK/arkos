"""Identity: verifies a Supabase JWT once, then carries a session cookie of our own.

The cookie is httpOnly, so the browser attaches it to `EventSource` and SSE needs
no stream token of its own.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
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
# When the JWK set was last pulled. Refresh is on our clock, not a token's.
_jwks_at: float = 0.0
_JWKS_LIFESPAN_S = 600.0


def jwks_url() -> str | None:
    """Where the project publishes the keys its tokens are signed with."""
    from harness_module import blobs

    project = blobs.project_url()
    return f"{project}/auth/v1/.well-known/jwks.json" if project else None


def _jwks() -> Any:
    """The JWKS client. Fetching is ours to schedule, never a token's to trigger."""
    global _jwks_client
    if _jwks_client is None:
        url = jwks_url()
        if url is None:
            return None
        _jwks_client = jwt.PyJWKClient(url, cache_keys=True)
    return _jwks_client


def reset_jwks_clock() -> None:
    """Force the next lookup to refresh. For tests and an urgent rotation."""
    global _jwks_at
    _jwks_at = 0.0


def reset_jwks() -> None:
    """Drop the cached JWKS client, for tests and key rotation."""
    global _jwks_client, _jwks_at
    _jwks_client = None
    _jwks_at = 0.0


def _signing_key(token: str) -> Any:
    """The key for this token's `kid`, from a periodically refreshed cache.

    NEVER fetches because a `kid` is unknown (12.2.5). PyJWKClient's own
    behaviour is to refetch on a miss, which hands an unauthenticated caller a
    lever: `POST /auth/session` is public, the header is attacker-chosen, and
    each miss cost one blocking 30s-timeout fetch on the default thread pool —
    the pool shared with blob IO and every sandbox call. Refresh is on OUR
    clock, so a flood of unknown kids costs one lookup each and no network.

    Raises InvalidKeyError for a kid the cache does not hold, which is a 401.
    """
    client = _jwks()
    if client is None:
        return None
    global _jwks_at
    now = time.monotonic()
    if now - _jwks_at > _JWKS_LIFESPAN_S:
        # One fetch per lifespan, whatever arrives in between. A real rotation
        # is visible within that window; an attacker's misses never are.
        client.get_jwk_set(refresh=True)
        _jwks_at = now
    kid = jwt.get_unverified_header(token).get("kid")
    for key in client.get_jwk_set().keys:
        if key.key_id == kid:
            return key.key
    raise jwt.InvalidKeyError(f"no signing key for kid {kid!r} in the cached JWKS")


def verify_supabase(token: str) -> dict[str, Any]:
    """Verifies a Supabase access token and returns its claims.

    The header's `alg` picks the key, and only from what this deployment has;
    anything else is refused. `audience` must be passed or PyJWT rejects the token.
    """
    audience = config.get("auth.jwt_audience") or "authenticated"
    algorithm = jwt.get_unverified_header(token).get("alg", "")

    if algorithm in _ASYMMETRIC:
        if _jwks() is None:
            raise jwt.InvalidKeyError(
                f"the token is signed with {algorithm}, and no project URL is configured to fetch keys from"
            )
        return jwt.decode(token, _signing_key(token), algorithms=[algorithm], audience=audience)

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


def mint_session(user_id: str, email: str | None = None) -> tuple[str, str, datetime]:
    """Signs a session cookie for a user `verify_supabase` has already cleared."""
    secret = _secret("ARK_SESSION_SECRET")
    if not secret:
        raise RuntimeError("ARK_SESSION_SECRET is unset")
    now = datetime.now(UTC)
    jti = str(uuid.uuid4())
    expires = now + timedelta(seconds=int(config.get("auth.session_ttl_s") or 604800))
    cookie = jwt.encode(
        {
            "sub": user_id,
            "email": email,
            "iss": _ISSUER,
            # The handle the server revokes by. A cookie is valid only while a
            # row for this jti exists, which is what makes signing out and a
            # password change able to reach a session in another browser.
            "jti": jti,
            "iat": now,
            "exp": expires,
        },
        secret,
        algorithm=_ALG,
    )
    return cookie, jti, expires


def read_session(cookie: str) -> dict[str, Any]:
    """Verifies a session cookie and returns its claims."""
    secret = _secret("ARK_SESSION_SECRET")
    if not secret:
        raise jwt.InvalidKeyError("ARK_SESSION_SECRET is unset")
    # `jti` is required: a cookie minted before 12.2.5 cannot be revoked, and a
    # session the server cannot take back is the thing this replaced.
    return jwt.decode(cookie, secret, algorithms=[_ALG], issuer=_ISSUER, options={"require": ["sub", "exp", "jti"]})
