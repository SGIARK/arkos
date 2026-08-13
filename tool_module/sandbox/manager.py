"""
Per-user persistent e2b sandbox lifecycle -- the user's "computer".

One sandbox per user: created on first use, resumed on later use (filesystem
persists across pause/resume), paused when idle. This is the ONLY module that
touches the e2b SDK, so the persistence flavor (pause/resume now, volumes later)
can change without touching the agent, runner, or endpoints.

Does NOT run the agent loop or make routing decisions -- it is plumbing.

MOVED HERE FROM computer_module, NOT YET FINISHED (Task 8). It still reads
`user_sandboxes` with psycopg2, and migration 0 rebuilt that table with a UUID
`user_id` where this expects TEXT, so the DB half is broken until it moves to
`db.pool`. Nothing calls it yet.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import psycopg2
import psycopg2.extras
from e2b_code_interpreter import Sandbox

from config_module.loader import config

logger = logging.getLogger(__name__)

# e2b 2.25.1 facts (verified against the live API by a spike since deleted):
#   Sandbox.create(timeout=...) -> sandbox with .sandbox_id
#   sbx.pause() returns None; resume via Sandbox.connect(sandbox_id)
#   sbx.commands.run(cmd) -> .stdout/.stderr/.exit_code (raises on non-zero exit)
#   sbx.files.write/read/list ; EntryInfo has .name/.path/.size/.type
_DEFAULT_TIMEOUT = 300


def _sbx_timeout() -> int:
    return int(config.get("sandbox.timeout_seconds") or _DEFAULT_TIMEOUT)


def _sbx_template() -> str | None:
    t = config.get("sandbox.template")
    # "base" is the SDK default; pass nothing so we don't risk an invalid-template error.
    return t if t and t != "base" else None


# ---------------------------------------------------------------------------
# DB helpers (sync; called via asyncio.to_thread). user_sandboxes: see 0005.
# ---------------------------------------------------------------------------
def _connect():
    return psycopg2.connect(config.get("database.url"))


def _db_get_row(user_id: str) -> dict[str, Any] | None:
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT user_id, e2b_sandbox_id, status FROM user_sandboxes WHERE user_id = %s",
                (user_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def _db_upsert(user_id: str, sandbox_id: str) -> None:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_sandboxes (user_id, e2b_sandbox_id, status, last_active_at)
                VALUES (%s, %s, 'active', now())
                ON CONFLICT (user_id)
                DO UPDATE SET e2b_sandbox_id = EXCLUDED.e2b_sandbox_id,
                              status = 'active',
                              last_active_at = now()
                """,
                (user_id, sandbox_id),
            )
            conn.commit()
    finally:
        conn.close()


def _db_set_status(user_id: str, status: str) -> None:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE user_sandboxes SET status = %s, last_active_at = now() WHERE user_id = %s",
                (status, user_id),
            )
            conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
class SandboxManager:
    """
    Owns the per-user sandbox handle cache + lifecycle. Async wrapper over the
    synchronous e2b SDK (calls run in a thread). One sandbox per user_id.
    """

    def __init__(self) -> None:
        self._live: dict[str, Sandbox] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock(self, user_id: str) -> asyncio.Lock:
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    async def get_or_create(self, user_id: str) -> Sandbox:
        """
        Return the user's live sandbox, creating or resuming as needed.

        Resolves in priority order: in-memory handle -> resume stored sandbox_id
        -> create fresh. A per-user lock prevents two concurrent calls from
        creating duplicate sandboxes.
        """
        async with self._lock(user_id):
            cached = self._live.get(user_id)
            if cached is not None:
                # Validate + keep alive in one call: set_timeout pings the sandbox
                # and resets its idle timer. If e2b already reaped it (idle past
                # the timeout), this raises -- evict the dead handle and recreate.
                try:
                    await asyncio.to_thread(cached.set_timeout, _sbx_timeout())
                    return cached
                except Exception as e:
                    logger.warning("cached sandbox for user %s is gone (%s); recreating", user_id, e)
                    self._live.pop(user_id, None)

            row = await asyncio.to_thread(_db_get_row, user_id)
            if row:
                try:
                    sbx = await asyncio.to_thread(Sandbox.connect, row["e2b_sandbox_id"])
                    await asyncio.to_thread(sbx.set_timeout, _sbx_timeout())
                    self._live[user_id] = sbx
                    await asyncio.to_thread(_db_set_status, user_id, "active")
                    logger.info("resumed sandbox %s for user %s", row["e2b_sandbox_id"], user_id)
                    return sbx
                except Exception as e:
                    # Stored sandbox is gone/expired; fall through to create a fresh one.
                    logger.warning("resume failed for user %s (%s); creating new sandbox", user_id, e)

            sbx = await asyncio.to_thread(self._create)
            self._live[user_id] = sbx
            await asyncio.to_thread(_db_upsert, user_id, sbx.sandbox_id)
            logger.info("created sandbox %s for user %s", sbx.sandbox_id, user_id)
            return sbx

    def _create(self) -> Sandbox:
        template = _sbx_template()
        if template:
            return Sandbox.create(template=template, timeout=_sbx_timeout())
        return Sandbox.create(timeout=_sbx_timeout())

    async def exec(self, user_id: str, command: str, timeout: int = 120) -> dict[str, Any]:
        """Run a shell command. Returns {stdout, stderr, exit_code} even on non-zero exit."""
        sbx = await self.get_or_create(user_id)

        def _run() -> dict[str, Any]:
            try:
                res = sbx.commands.run(command, timeout=timeout)
                return {"stdout": res.stdout, "stderr": res.stderr, "exit_code": res.exit_code}
            except Exception as e:
                # e2b raises on non-zero exit; the exception carries the streams.
                return {
                    "stdout": getattr(e, "stdout", ""),
                    "stderr": getattr(e, "stderr", str(e)),
                    "exit_code": getattr(e, "exit_code", 1),
                }

        return await asyncio.to_thread(_run)

    async def read_file(self, user_id: str, path: str) -> str:
        sbx = await self.get_or_create(user_id)
        return await asyncio.to_thread(sbx.files.read, path)

    async def write_file(self, user_id: str, path: str, content: str) -> None:
        sbx = await self.get_or_create(user_id)
        await asyncio.to_thread(sbx.files.write, path, content)

    async def list_dir(self, user_id: str, path: str = "/home/user") -> list[dict[str, Any]]:
        """List a directory. Returns [{name, path, is_dir, size}]."""
        sbx = await self.get_or_create(user_id)
        entries = await asyncio.to_thread(sbx.files.list, path)

        def _is_dir(entry: Any) -> bool:
            t = getattr(entry, "type", None)
            return getattr(t, "value", str(t)).lower() == "dir"

        return [{"name": e.name, "path": e.path, "is_dir": _is_dir(e), "size": getattr(e, "size", 0)} for e in entries]

    async def pause(self, user_id: str) -> None:
        """Hibernate the sandbox (persist state, stop compute cost)."""
        sbx = self._live.pop(user_id, None)
        if sbx is None:
            return
        await asyncio.to_thread(sbx.pause)
        await asyncio.to_thread(_db_set_status, user_id, "paused")
        logger.info("paused sandbox for user %s", user_id)


# Module singleton (mirrors config / tool_manager).
sandbox_manager = SandboxManager()
