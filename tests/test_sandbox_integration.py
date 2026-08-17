"""The sandbox manager against a real e2b sandbox.

Marked `integration`: it boots a real VM, costs money and takes seconds. Run it
with `pytest -m integration`. Skipped without `E2B_API_KEY` or the SDK.

The unit tests in test_sandbox_tools.py cover the tool logic against a fake.
What can only be checked here is that the SDK is driven correctly and that the
sandbox's environment holds none of our credentials.
"""

from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio

from db import pool
from tool_module.sandbox import manager as sandbox_manager

pytest.importorskip("e2b_code_interpreter", reason="the e2b SDK is not installed")

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("E2B_API_KEY"), reason="E2B_API_KEY is not set"),
]

_seeded: list[uuid.UUID] = []


@pytest_asyncio.fixture(autouse=True)
async def _db():
    try:
        await pool.fetchval("SELECT 1")
    except Exception as e:  # noqa: BLE001 - any connection failure means skip
        await pool.close()
        pytest.skip(f"needs the arkos database (migration 0 applied): {e}")
    yield
    await pool.execute("DELETE FROM user_sandboxes WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM users WHERE id = ANY($1::uuid[])", _seeded)
    _seeded.clear()
    sandbox_manager.reset()
    await pool.close()


async def _user() -> str:
    user_id = uuid.uuid4()
    await pool.execute("INSERT INTO users (id) VALUES ($1)", user_id)
    _seeded.append(user_id)
    return str(user_id)


async def _kill(manager: sandbox_manager.SandboxManager, user_id: str) -> None:
    """Destroy the sandbox. An orphan keeps billing until its own idle timeout."""
    live = manager._live.pop(user_id, None)
    if live is not None:
        import asyncio

        await asyncio.to_thread(live.kill)


async def test_a_real_sandbox_round_trips_files_and_commands():
    user_id = await _user()
    manager = sandbox_manager.manager()
    try:
        await manager.write_file(user_id, "/home/user/hello.txt", "alpha\nbeta\n")
        content = await manager.read_file(user_id, "/home/user/hello.txt")
        listing = await manager.list_dir(user_id, "/home/user")
        ran = await manager.exec(user_id, "wc -l < /home/user/hello.txt")
        failed = await manager.exec(user_id, "cat /home/user/nope.txt")

        assert content == "alpha\nbeta\n"
        assert any(entry["name"] == "hello.txt" and not entry["is_dir"] for entry in listing)
        assert ran["exit_code"] == 0
        assert ran["stdout"].strip() == "2"
        # A failing command returns its streams rather than raising.
        assert failed["exit_code"] != 0
        assert failed["stderr"]

        stored = await pool.fetchval("SELECT sandbox_id FROM user_sandboxes WHERE user_id = $1", uuid.UUID(user_id))
        assert stored, "the sandbox id was not recorded, so it cannot be resumed"
    finally:
        await _kill(manager, user_id)


async def test_no_credentials_reach_a_real_sandbox():
    """The card requires it and only the sandbox itself can confirm it."""
    user_id = await _user()
    manager = sandbox_manager.manager()
    try:
        env = (await manager.exec(user_id, "env"))["stdout"]

        for name in ("OPENAI_API_KEY", "DB_URL", "SMITHERY_API_KEY", "E2B_API_KEY", "ARK_SESSION_SECRET"):
            assert name not in env, f"{name} is set inside the sandbox"
            value = os.environ.get(name)
            if value:
                assert value not in env, f"the value of {name} is in the sandbox environment"
    finally:
        await _kill(manager, user_id)


async def test_a_second_call_reuses_the_same_sandbox():
    """One sandbox per user: the filesystem is the point, so a second call must not get a fresh one."""
    user_id = await _user()
    manager = sandbox_manager.manager()
    try:
        first = await manager.get_or_create(user_id)
        second = await manager.get_or_create(user_id)

        assert first.sandbox_id == second.sandbox_id
    finally:
        await _kill(manager, user_id)


async def test_a_sandbox_is_resumed_from_its_stored_id():
    """State persists, the running instance is cattle: a new process reconnects rather than rebuilding."""
    user_id = await _user()
    manager = sandbox_manager.manager()
    try:
        original = await manager.get_or_create(user_id)
        await manager.write_file(user_id, "/home/user/marker.txt", "still here")

        # Forget the handle, as a restarted process would.
        manager._live.pop(user_id, None)
        resumed = await manager.get_or_create(user_id)

        assert resumed.sandbox_id == original.sandbox_id
        assert await manager.read_file(user_id, "/home/user/marker.txt") == "still here"
    finally:
        await _kill(manager, user_id)
