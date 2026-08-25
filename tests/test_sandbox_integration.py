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
from tests.dbgate import require_db
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
    await require_db()
    yield
    await pool.execute("DELETE FROM sessions WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM users WHERE id = ANY($1::uuid[])", _seeded)
    _seeded.clear()
    sandbox_manager.reset()
    await pool.close()


async def _session() -> str:
    user_id = uuid.uuid4()
    await pool.execute("INSERT INTO users (id) VALUES ($1)", user_id)
    _seeded.append(user_id)
    session_id = str(
        await pool.fetchval(
            "INSERT INTO sessions (user_id, mode, status) VALUES ($1, 'attended', 'running') RETURNING id",
            user_id,
        )
    )
    # A box is only booted against a slot.
    assert await sandbox_manager.claim_slot(session_id)
    return session_id


async def _kill(manager: sandbox_manager.SandboxManager, session_id: str) -> None:
    """Destroy the sandbox. An orphan keeps billing until its own idle timeout."""
    await manager.reap(session_id)


async def test_a_real_sandbox_round_trips_files_and_commands():
    session_id = await _session()
    manager = sandbox_manager.manager()
    try:
        await manager.write_file(session_id, "/home/user/hello.txt", "alpha\nbeta\n")
        content = await manager.read_file(session_id, "/home/user/hello.txt")
        listing = await manager.list_dir(session_id, "/home/user")
        ran = await manager.exec(session_id, "wc -l < /home/user/hello.txt")
        failed = await manager.exec(session_id, "cat /home/user/nope.txt")

        assert content == "alpha\nbeta\n"
        assert any(entry["name"] == "hello.txt" and not entry["is_dir"] for entry in listing)
        assert ran["exit_code"] == 0
        assert ran["stdout"].strip() == "2"
        # A failing command returns its streams rather than raising.
        assert failed["exit_code"] != 0
        assert failed["stderr"]

        stored = await pool.fetchval(
            "SELECT sandbox_id FROM session_sandboxes WHERE session_id = $1", uuid.UUID(session_id)
        )
        assert stored, "the sandbox id was not recorded, so it cannot be resumed"
    finally:
        await _kill(manager, session_id)


async def test_no_credentials_reach_a_real_sandbox():
    """The card requires it and only the sandbox itself can confirm it."""
    session_id = await _session()
    manager = sandbox_manager.manager()
    try:
        env = (await manager.exec(session_id, "env"))["stdout"]

        for name in ("OPENAI_API_KEY", "DB_URL", "ARCADE_API_KEY", "E2B_API_KEY", "ARK_SESSION_SECRET"):
            assert name not in env, f"{name} is set inside the sandbox"
            value = os.environ.get(name)
            if value:
                assert value not in env, f"the value of {name} is in the sandbox environment"
    finally:
        await _kill(manager, session_id)


async def test_a_second_call_reuses_the_same_sandbox():
    """The box follows the session: a second call within the run must not get a fresh one."""
    session_id = await _session()
    manager = sandbox_manager.manager()
    try:
        first = await manager.get_or_create(session_id)
        second = await manager.get_or_create(session_id)

        assert first.sandbox_id == second.sandbox_id
    finally:
        await _kill(manager, session_id)


async def test_a_sandbox_is_resumed_from_its_stored_id():
    """The running instance is cattle: a restarted process reconnects rather than rebuilding."""
    session_id = await _session()
    manager = sandbox_manager.manager()
    try:
        original = await manager.get_or_create(session_id)
        await manager.write_file(session_id, "/home/user/marker.txt", "still here")

        # Forget the handle, as a restarted process would.
        manager._live.pop(session_id, None)
        resumed = await manager.get_or_create(session_id)

        assert resumed.sandbox_id == original.sandbox_id
        assert await manager.read_file(session_id, "/home/user/marker.txt") == "still here"
    finally:
        await _kill(manager, session_id)


async def test_two_sessions_of_one_user_get_two_boxes():
    """Per-session boxes: the second session is not handed the first's computer."""
    user_id = uuid.uuid4()
    await pool.execute("INSERT INTO users (id) VALUES ($1)", user_id)
    _seeded.append(user_id)
    sessions = []
    for _ in range(2):
        session_id = str(
            await pool.fetchval(
                "INSERT INTO sessions (user_id, mode, status) VALUES ($1, 'attended', 'running') RETURNING id",
                user_id,
            )
        )
        assert await sandbox_manager.claim_slot(session_id)
        sessions.append(session_id)
    manager = sandbox_manager.manager()
    try:
        first = await manager.get_or_create(sessions[0])
        second = await manager.get_or_create(sessions[1])

        assert first.sandbox_id != second.sandbox_id
    finally:
        for session_id in sessions:
            await _kill(manager, session_id)


async def test_a_reaped_sandbox_is_gone_and_its_slot_is_free():
    session_id = await _session()
    manager = sandbox_manager.manager()
    sandbox = await manager.get_or_create(session_id)
    sandbox_id = sandbox.sandbox_id

    await manager.reap(session_id)

    slots = await pool.fetchval(
        "SELECT count(*) FROM session_sandboxes WHERE session_id = $1", uuid.UUID(session_id)
    )
    assert slots == 0
    assert _is_dead(sandbox_id), "the slot was freed but the box is still running"


async def test_a_box_with_no_handle_in_this_process_is_still_killed():
    """The restarted-process path: the id in the row is all there is to kill by."""
    session_id = await _session()
    manager = sandbox_manager.manager()
    sandbox = await manager.get_or_create(session_id)
    sandbox_id = sandbox.sandbox_id
    # Forget the handle, as a restarted process would.
    manager._live.pop(session_id, None)

    await manager.reap(session_id)

    assert _is_dead(sandbox_id), "kill by id did not destroy the box"


async def test_a_paused_box_resumes_with_its_files():
    """Park hibernates rather than kills, so the next turn starts warm."""
    session_id = await _session()
    manager = sandbox_manager.manager()
    try:
        original = await manager.get_or_create(session_id)
        await manager.write_file(session_id, "/home/user/scratch.txt", "written before the park")

        await manager.pause(session_id)
        resumed = await manager.get_or_create(session_id)

        assert resumed.sandbox_id == original.sandbox_id
        assert await manager.read_file(session_id, "/home/user/scratch.txt") == "written before the park"
    finally:
        await _kill(manager, session_id)


async def test_a_paused_box_can_still_be_reaped():
    """Terminal kills whatever state the box is in, so a park cannot strand one."""
    session_id = await _session()
    manager = sandbox_manager.manager()
    sandbox = await manager.get_or_create(session_id)
    sandbox_id = sandbox.sandbox_id
    await manager.pause(session_id)

    await manager.reap(session_id)

    slots = await pool.fetchval(
        "SELECT count(*) FROM session_sandboxes WHERE session_id = $1", uuid.UUID(session_id)
    )
    assert slots == 0
    assert _is_dead(sandbox_id), "a paused box survived its reap"


def _is_dead(sandbox_id: str) -> bool:
    """True if nothing answers on that box any more."""
    from e2b_code_interpreter import Sandbox

    try:
        Sandbox.connect(sandbox_id).commands.run("true", timeout=10)
    except Exception:  # noqa: BLE001 - any failure to reach it is the answer
        return True
    return False
