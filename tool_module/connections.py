"""
Per-user Composio connections, keyed by `(user_id, server)`.

`server` is Composio's toolkit prefix in upper snake (`GMAIL`, `LINEAR`) — the
vendor's durable key, never the `mcp_servers:` config label. A row is a cache of
a grant that lives at Composio; `connected_account_id` is Composio's id for that
grant, and a disconnect needs it to revoke rather than merely forget.
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
# Composio ERRORED: authorized once, since expired or revoked provider-side.
RECONNECT = "reconnect"


@dataclass(slots=True)
class Connection:
    """One user's standing with one Composio toolkit."""

    server: str
    status: str = PENDING
    refreshed_at: datetime | None = None
    connected_account_id: str | None = None

    @property
    def connected(self) -> bool:
        return self.status == CONNECTED


def _row(record: Any) -> Connection:
    return Connection(
        server=record["server"],
        status=record["status"],
        refreshed_at=record["refreshed_at"],
        connected_account_id=record["connected_account_id"],
    )


async def load(user_id: str) -> dict[str, Connection]:
    """Return every stored connection for one user, keyed by `server`."""
    rows = await pool.fetch(
        "SELECT server, status, refreshed_at, connected_account_id FROM user_connections WHERE user_id = $1",
        _uid(user_id),
    )
    return {r["server"]: _row(r) for r in rows}


async def mark(user_id: str, server: str, status: str, account_id: str | None = None) -> None:
    """Record what Composio says about one toolkit, inserting the row if it is the first word.

    A null `account_id` never clears a stored one: consent starts before the id exists.
    """
    await pool.execute(
        """
        INSERT INTO user_connections (user_id, server, status, connected_account_id)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (user_id, server)
        DO UPDATE SET status = EXCLUDED.status,
                      connected_account_id = COALESCE(EXCLUDED.connected_account_id,
                                                      user_connections.connected_account_id),
                      refreshed_at = now()
        """,
        _uid(user_id),
        server,
        status,
        account_id,
    )


async def sync(user_id: str, statuses: dict[str, str], account_ids: dict[str, str | None] | None = None) -> None:
    """Write a whole reading of Composio's per-user state in one transaction."""
    if not statuses:
        return
    ids = account_ids or {}
    async with (await pool.pool()).acquire() as conn, conn.transaction():
        for server, status in statuses.items():
            await conn.execute(
                """
                INSERT INTO user_connections (user_id, server, status, connected_account_id)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (user_id, server)
                DO UPDATE SET status = EXCLUDED.status,
                              connected_account_id = COALESCE(EXCLUDED.connected_account_id,
                                                              user_connections.connected_account_id),
                              refreshed_at = now()
                """,
                _uid(user_id),
                server,
                status,
                ids.get(server),
            )


async def forget(user_id: str, server: str) -> None:
    """Drop the row, so the toolkit reads as never-connected until Composio says otherwise."""
    await pool.execute(
        "DELETE FROM user_connections WHERE user_id = $1 AND server = $2",
        _uid(user_id),
        server,
    )
