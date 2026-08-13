"""
Per-user persistent e2b sandbox lifecycle: created on first use, resumed later,
paused when idle. The only module that touches the e2b SDK.

UNINTEGRATED: nothing calls it yet, and it still reads `user_sandboxes` with
psycopg2 expecting a TEXT `user_id` where the table now has UUID.
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

_DEFAULT_TIMEOUT = 300


def _sbx_timeout() -> int:
    return int(config.get("sandbox.timeout_seconds") or _DEFAULT_TIMEOUT)


def _sbx_template() -> str | None:
    t = config.get("sandbox.template")
    # "base" is the SDK default; pass nothing rather than risk an invalid-template error.
    return t if t and t != "base" else None


# DB helpers, sync; called via asyncio.to_thread.
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


class SandboxManager:
    """One sandbox per user_id: an async wrapper over the synchronous e2b SDK."""

    def __init__(self) -> None:
        self._live: dict[str, Sandbox] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock(self, user_id: str) -> asyncio.Lock:
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    async def get_or_create(self, user_id: str) -> Sandbox:
        """Return the user's live sandbox: cached handle, else resume the stored id, else create."""
        async with self._lock(user_id):
            cached = self._live.get(user_id)
            if cached is not None:
                # set_timeout both validates the handle and resets the idle timer;
                # it raises if e2b already reaped the sandbox.
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
                    # Stored sandbox is gone; fall through and create a fresh one.
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
        """Run a shell command, returning {stdout, stderr, exit_code} even on non-zero exit."""
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
        """List a directory as [{name, path, is_dir, size}]."""
        sbx = await self.get_or_create(user_id)
        entries = await asyncio.to_thread(sbx.files.list, path)

        def _is_dir(entry: Any) -> bool:
            t = getattr(entry, "type", None)
            return getattr(t, "value", str(t)).lower() == "dir"

        return [{"name": e.name, "path": e.path, "is_dir": _is_dir(e), "size": getattr(e, "size", 0)} for e in entries]

    async def pause(self, user_id: str) -> None:
        """Hibernate the sandbox so its state persists without compute cost."""
        sbx = self._live.pop(user_id, None)
        if sbx is None:
            return
        await asyncio.to_thread(sbx.pause)
        await asyncio.to_thread(_db_set_status, user_id, "paused")
        logger.info("paused sandbox for user %s", user_id)


sandbox_manager = SandboxManager()
