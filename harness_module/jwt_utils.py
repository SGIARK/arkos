"""Identity: verifies a Supabase JWT once, then carries a session cookie of our own.

The cookie is httpOnly, so the browser attaches it to `EventSource` and SSE needs
no stream token of its own.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt  # PyJWT

from config_module.loader import config

logger = logging.getLogger(__name__)

_ALG = "HS256"

# Issuer claim on the cookies minted here; `read_session` requires it.
_ISSUER = "arkos"


def _secret(name: str) -> str | None:
    value = os.environ.get(name)
    return value or None


def assert_secure_secrets() -> None:
    """Refuses to start without a way to sign sessions and a way to verify tokens."""
    if not _secret("ARKOS_SESSION_SECRET"):
        raise RuntimeError(
            "ARKOS_SESSION_SECRET is unset. Refusing to start: sessions would be unsignable. See .env.example."
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
        # A SHORT timeout on purpose: this runs on the default thread pool,
        # shared with blob IO and every sandbox call, and 30s of holding one
        # thread because an endpoint is unreachable is the outage, not the fix.
        _jwks_client = jwt.PyJWKClient(
            url,
            cache_keys=True,
            timeout=float(config.get("auth.jwks_timeout_s") or 5),
            # FAR longer than the refresh interval, because the tick owns the
            # schedule and this is only the fallback. At PyJWT's 300s default a
            # tick failing for five minutes empties the cache and every sign-in
            # is refused — a flaky endpoint would become an outage. Keys are
            # stable; a genuinely rotated one fails verification anyway.
            lifespan=float(config.get("auth.jwks_cache_ttl_s") or 86400),
        )
    return _jwks_client


def reset_jwks() -> None:
    """Drop the cached JWKS client, for tests and key rotation."""
    global _jwks_client
    _jwks_client = None


def _jwks_file() -> pathlib.Path:
    return pathlib.Path(str(config.get("auth.jwks_cache_file") or "/tmp/arkos-jwks.json"))


def prime_jwks_from_disk() -> bool:
    """Seed the cache from the last good fetch, so a restart is never cold.

    Without this, a process that restarts into the endpoint's flaky window
    refuses every sign-in until a fetch lands. The keys are public material —
    they are served unauthenticated to anyone who asks — so a file is fine, and
    it is only ever a starting point: the tick refreshes it on the usual clock.

    A corrupt or unreadable file is not an error worth failing over; the tick
    will overwrite it.
    """
    client = _jwks()
    if client is None or client.jwk_set_cache is None:
        return False
    try:
        data = json.loads(_jwks_file().read_text())
        # The RAW dict: PyJWT's cache holds what `fetch_data` returns, and
        # `get_jwk_set` re-parses it. Putting a PyJWKSet here type-errors on the
        # next read, which looks exactly like an empty cache.
        jwt.PyJWKSet.from_dict(data)  # parsed only to reject a malformed file
        client.jwk_set_cache.put(data)
        logger.info("primed the key cache from %s", _jwks_file())
        return True
    except FileNotFoundError:
        return False
    except Exception as e:  # noqa: BLE001 - a bad cache file is not fatal
        logger.warning("could not prime keys from %s: %s", _jwks_file(), e)
        return False


class NoKeysPublished(Exception):
    """The JWKS endpoint answered with an empty set: the project signs HS256 only."""


def refresh_jwks() -> bool:
    """Pull the JWK set into the cache and onto disk. BLOCKING — run off-loop.

    The only place this process fetches keys. It is called on a timer, never by
    a request, which is what makes `_signing_key` a pure cache read: the JWKS
    host is normally 200ms and occasionally 30s, and a request must never be the
    thing that discovers which.

    Raises `NoKeysPublished` when the endpoint serves an empty set, which is a
    project signing HS256 only and no reason to keep asking.
    """
    client = _jwks()
    if client is None:
        return False
    try:
        data = client.fetch_data()
        # `fetch_data` already caches it; put again so the path is explicit and
        # does not depend on that staying true.
        client.jwk_set_cache.put(data)
        if not data.get("keys"):
            raise jwt.PyJWKSetError("The JWK Set did not contain any keys")
    except jwt.PyJWKSetError as e:
        # Not an outage: the endpoint answered, with nothing in it. Newer PyJWT
        # raises this from `fetch_data`, older takes the empty set silently; both
        # land here. A cache file from before says the same, so it goes too.
        _jwks_file().unlink(missing_ok=True)
        raise NoKeysPublished(str(e)) from e
    except Exception as e:  # noqa: BLE001 - a stale cache beats a blocked request
        logger.warning("JWKS refresh failed; serving whatever is cached: %s", e)
        return False
    try:
        # Written atomically: a half-written file read at the next boot is a
        # cold start with extra steps.
        path = _jwks_file()
        scratch = path.with_suffix(".tmp")
        scratch.write_text(json.dumps(data))
        scratch.replace(path)
    except Exception as e:  # noqa: BLE001 - the cache is in memory either way
        logger.warning("could not persist keys to %s: %s", _jwks_file(), e)
    return True


def _signing_key(token: str) -> Any:
    """The key for this token's `kid`, from the cache. NEVER fetches.

    An unknown kid is a 401 and no network at all (12.2.5). `POST /auth/session`
    is public and the header is the caller's, so a miss must cost a dictionary
    lookup — PyJWKClient's own behaviour is to refetch, which would hand an
    unauthenticated caller a lever on the pool shared with blob IO and every
    sandbox call.
    """
    client = _jwks()
    if client is None:
        return None
    kid = jwt.get_unverified_header(token).get("kid")
    # Ask the CACHE, not the client: `get_jwk_set()` fetches when it is empty.
    cache = getattr(client, "jwk_set_cache", None)
    if cache is not None and cache.get() is None:
        raise jwt.InvalidKeyError(f"no JWKS cached to verify kid {kid!r}; the refresh tick has not landed one yet")
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
    secret = _secret("ARKOS_SESSION_SECRET")
    if not secret:
        raise RuntimeError("ARKOS_SESSION_SECRET is unset")
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
    secret = _secret("ARKOS_SESSION_SECRET")
    if not secret:
        raise jwt.InvalidKeyError("ARKOS_SESSION_SECRET is unset")
    # `jti` is required: a cookie minted before 12.2.5 cannot be revoked, and a
    # session the server cannot take back is the thing this replaced.
    return jwt.decode(cookie, secret, algorithms=[_ALG], issuer=_ISSUER, options={"require": ["sub", "exp", "jti"]})
