"""
Claims on the stateful hands: the sandbox and the browser.

A resource is leased when it is both shared across a user's concurrent sessions
and carries state between calls. The lease is held for the whole session, so one
session's half-finished write cannot be interleaved with another's.

Leases carry an expiry, so a process that dies holding one does not lock the
resource until someone intervenes.
"""

from __future__ import annotations

import logging
import uuid

from db import pool

logger = logging.getLogger(__name__)


def key(resource: str, user_id: str) -> str:
    """Build the key a resource is leased under. One sandbox and one browser per user."""
    return f"{resource}:{user_id}"


async def acquire(resource_key: str, session_id: str, ttl_s: float) -> bool:
    """
    Take or renew a lease.

    Returns:
        True if the session now holds it. False if another session holds an
        unexpired lease.
    """
    held = await pool.fetchval(
        """
        INSERT INTO resource_leases (resource_key, session_id, expires_at)
        VALUES ($1, $2, now() + make_interval(secs => $3))
        ON CONFLICT (resource_key) DO UPDATE
           SET session_id  = EXCLUDED.session_id,
               acquired_at = now(),
               expires_at  = EXCLUDED.expires_at
         WHERE resource_leases.session_id = EXCLUDED.session_id
            OR resource_leases.expires_at < now()
        RETURNING session_id
        """,
        resource_key,
        _uuid(session_id),
        float(ttl_s),
    )
    return held is not None


async def release(resource_key: str, session_id: str) -> bool:
    """Give up a lease this session holds. Another session's lease is left alone."""
    result = await pool.execute(
        "DELETE FROM resource_leases WHERE resource_key = $1 AND session_id = $2",
        resource_key,
        _uuid(session_id),
    )
    return result.endswith(" 1")


async def release_all(session_id: str) -> int:
    """Give up every lease a session holds, on terminal or on park."""
    result = await pool.execute("DELETE FROM resource_leases WHERE session_id = $1", _uuid(session_id))
    freed = int(result.rsplit(" ", 1)[-1] or 0)
    if freed:
        logger.info("released %d lease(s) held by session %s", freed, session_id)
    return freed


async def holder(resource_key: str) -> str | None:
    """Return the session holding an unexpired lease, if any."""
    held = await pool.fetchval(
        "SELECT session_id FROM resource_leases WHERE resource_key = $1 AND expires_at > now()",
        resource_key,
    )
    return str(held) if held else None


def _uuid(value: str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))
