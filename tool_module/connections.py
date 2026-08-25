"""
Per-user Arcade connections, keyed by `(user_id, server)`.

`server` is the Arcade app prefix — `Gmail`, `Linear`, `GoogleCalendar` — which
is also what every one of that app's tool names is prefixed with. It is the
vendor's name for the app, not a config label, which is what makes it safe to
store: renaming the key under `mcp_servers:` in config.yaml still changes
nothing here.

A row is a CACHE of a fact that lives at Arcade, not the fact itself. The grant
belongs to Arcade, keyed by the user id we send in `Arcade-User-Id`, and it
survives anything we do to this table — including recreating the gateway, which
is why the gateway url is not part of the key. What the row buys is a settings
panel that can render without a round trip, and a `PUT /sessions/{id}/tools`
that can refuse an unconnected server without one either.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from db import pool
from db.ids import as_uuid as _uid

PENDING = "pending"
CONNECTED = "connected"
DISCONNECTED = "disconnected"


@dataclass(slots=True)
class Connection:
    """One user's standing with one Arcade app."""

    server: str
    status: str = PENDING
    refreshed_at: datetime | None = None

    @property
    def connected(self) -> bool:
        return self.status == CONNECTED


def _row(record: Any) -> Connection:
    return Connection(
        server=record["server"],
        status=record["status"],
        refreshed_at=record["refreshed_at"],
    )


async def load(user_id: str) -> dict[str, Connection]:
    """Return every stored connection for one user, keyed by `server`."""
    rows = await pool.fetch(
        "SELECT server, status, refreshed_at FROM user_connections WHERE user_id = $1",
        _uid(user_id),
    )
    return {r["server"]: _row(r) for r in rows}


async def mark(user_id: str, server: str, status: str) -> None:
    """Record what Arcade says about one app, inserting the row if it is the first word."""
    await pool.execute(
        """
        INSERT INTO user_connections (user_id, server, status)
        VALUES ($1, $2, $3)
        ON CONFLICT (user_id, server)
        DO UPDATE SET status = EXCLUDED.status, refreshed_at = now()
        """,
        _uid(user_id),
        server,
        status,
    )


async def sync(user_id: str, statuses: dict[str, str]) -> None:
    """Write a whole reading of Arcade's per-user state in one transaction.

    `Arcade_ListApps` answers for every app at once, so the rows move together
    or not at all: a partial write would leave the panel showing one app's
    reading beside another's from minutes ago, with nothing to say which.
    """
    if not statuses:
        return
    async with (await pool.pool()).acquire() as conn, conn.transaction():
        for server, status in statuses.items():
            await conn.execute(
                """
                INSERT INTO user_connections (user_id, server, status)
                VALUES ($1, $2, $3)
                ON CONFLICT (user_id, server)
                DO UPDATE SET status = EXCLUDED.status, refreshed_at = now()
                """,
                _uid(user_id),
                server,
                status,
            )


async def forget(user_id: str, server: str) -> None:
    """Drop the row, so the app reads as never-connected until Arcade says otherwise."""
    await pool.execute(
        "DELETE FROM user_connections WHERE user_id = $1 AND server = $2",
        _uid(user_id),
        server,
    )
