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
    """Refuses to start without a way to sign sessions and a way to verify tokens.

    Verification needs one of two things: a project URL, which is where the
    signing keys are fetched from, or the shared HS256 secret for a project
    still signing with one.
    """
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


# Supabase signs with an asymmetric key published at the project's JWKS
# endpoint. A shared HS256 secret is the older scheme, still used by projects
# that have not moved.
_ASYMMETRIC = ("ES256", "RS256", "EdDSA")

_jwks_client: Any = None


def jwks_url() -> str | None:
    """Where the project publishes the keys its tokens are signed with."""
    from harness_module import store

    project = store.project_url()
    return f"{project}/auth/v1/.well-known/jwks.json" if project else None


def _jwks() -> Any:
    """The JWKS client. It caches keys and refetches when it meets an unknown kid."""
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

    The algorithm named in the token's header selects which key to check it
    against, but only from the mechanisms this deployment has: an asymmetric
    token against the project's published key, HS256 against the shared secret.
    A token naming anything else is refused. Supabase stamps aud=authenticated,
    and PyJWT rejects a token whose audience the caller did not name.

    Raises:
        jwt.PyJWTError: invalid signature, expired, wrong audience, or an
            algorithm this deployment cannot verify.
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
