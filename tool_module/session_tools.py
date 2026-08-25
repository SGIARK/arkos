"""
Which MCP servers one session may reach, stored per session and keyed by `server`.

The default is nothing: a server the user has connected is not a server this
session can reach until it is toggled on. An absent row reads as off, so a fresh
session gets exactly our own tools and cannot be the one that puts 164 schemas
in a request.

`server` is the Arcade app prefix — `Gmail`, `Linear` — the same identity
`user_connections` is keyed by, and for the same reason: a `mcp_servers:` config
key is an in-process label rebuilt at every startup, so nothing durable may
reference it. It was the `mcp_url` until 11.10, when every app moved behind one
gateway url and a url stopped telling two servers apart.

This module records and reports the toggles and nothing else. `registry.manifest`
is where they become reach.
"""

from __future__ import annotations

import uuid

from db import pool


def _sid(session_id: str) -> uuid.UUID:
    """Parse a session id as the UUID it must be, raising ValueError otherwise."""
    try:
        return uuid.UUID(str(session_id))
    except (ValueError, TypeError, AttributeError) as e:
        raise ValueError(f"session_id must be a UUID, got {session_id!r}") from e


async def enabled_servers(session_id: str) -> list[str]:
    """Return the servers this session has been given, LONGEST-ENABLED FIRST.

    The order is load-bearing, not cosmetic. When the cap forces a server to be
    left out, the most recently enabled one goes first — the session keeps the
    reach it has been working with, and loses the thing that was just added. The
    caller reads this list front to back and stops when the next server will not
    fit, so the order here IS the drop rule.
    """
    rows = await pool.fetch(
        """
        SELECT server FROM session_tools
         WHERE session_id = $1 AND enabled
         ORDER BY updated_at, server
        """,
        _sid(session_id),
    )
    return [r["server"] for r in rows]


async def set_enabled(session_id: str, server: str, enabled: bool) -> None:
    """Record one server as reachable, or not, for this session.

    `updated_at` moves on every write, including one that re-asserts a toggle
    already on, which is what makes it the recency the drop rule reads.
    """
    await pool.execute(
        """
        INSERT INTO session_tools (session_id, server, enabled)
        VALUES ($1, $2, $3)
        ON CONFLICT (session_id, server)
        DO UPDATE SET enabled = EXCLUDED.enabled, updated_at = now()
        """,
        _sid(session_id),
        server,
        enabled,
    )
