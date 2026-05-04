"""
Background subagent runner.

Every approved plan becomes one `run_task(task_id)` coroutine spawned via
asyncio.create_task. It builds a subagent with:
  - its own Memory row (new session_id, same user_id as the chat agent, so
    long-term mem0 is shared)
  - the executor state graph (state_module/graphs/executor.yaml)
  - a ScopedToolManager limited to the plan's required_tools
and then walks plan_steps one at a time until the executor graph terminates.

The runner keeps a process-local registry of active asyncio.Tasks so we can
spawn, cancel, and sweep for orphans on startup.
"""

from __future__ import annotations

import asyncio
import os
import sys
import traceback
import uuid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_module.agent import Agent  # noqa: E402
from base_module.task_store import (  # noqa: E402
    get_task,
    log_event,
    mark_task_completed,
    mark_task_failed,
    set_task_status,
)
from config_module.loader import config  # noqa: E402
from memory_module.memory import Memory  # noqa: E402
from model_module.ArkModelNew import SystemMessage, UserMessage  # noqa: E402
from state_module.core.state_handler import StateHandler  # noqa: E402
from tool_module.scoped import ScopedToolManager  # noqa: E402

# Keep strong refs to in-flight tasks so GC doesn't cancel them.
_RUNNING: dict[str, asyncio.Task] = {}

EXECUTOR_GRAPH_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "state_module", "agent_executor", "executor.yaml")
)

EXECUTOR_SYSTEM_PROMPT = """\
You are an ARK subagent running a plan approved by the user.
You execute plan steps literally. You do NOT re-plan or skip steps.
If you cannot complete a step with the tools you have, ask the human.
Be concise in your messages.
"""


def _shared_deps():
    """Pull the process-wide llm + tool_manager off base_module.app at call time."""
    # Local import to avoid a circular import at module load.
    from base_module import app as app_mod  # noqa: WPS433

    return app_mod.llm, app_mod.tool_manager


async def run_task(task_id: str) -> None:
    """Main coroutine for a single subagent task. Logs events, updates status."""
    try:
        await _run_task_inner(task_id)
    except asyncio.CancelledError:
        log_event(task_id, "cancelled", "subagent task cancelled")
        set_task_status(task_id, "cancelled")
        raise
    except Exception as e:
        tb = traceback.format_exc()
        log_event(task_id, "error", str(e), payload={"traceback": tb})
        mark_task_failed(task_id, str(e))
    finally:
        _RUNNING.pop(task_id, None)


async def _run_task_inner(task_id: str) -> None:
    task = get_task(task_id)
    if not task:
        log_event(task_id, "error", "task row not found at start")
        return

    user_id = str(task["user_id"])
    payload = task["context_payload"] or {}
    if isinstance(payload, str):
        import json as _json

        payload = _json.loads(payload)

    plan_text = payload.get("plan") or ""
    plan_steps = payload.get("plan_steps") or []
    if not plan_steps and plan_text:
        # fall back: split a "1. foo\n2. bar" blob into lines
        plan_steps = [line.split(". ", 1)[-1].strip() for line in plan_text.splitlines() if line.strip()]

    if not plan_steps:
        log_event(task_id, "error", "task has no plan_steps; nothing to execute")
        mark_task_failed(task_id, "no plan_steps")
        return

    title = payload.get("title", "(untitled task)")
    required_tools = list(task["required_tools"] or [])

    # session_id: either the one stored at creation time, or a fresh one.
    session_id = task.get("session_id") or str(uuid.uuid4())
    if not task.get("session_id"):
        # persist so restarts can resume the same memory row
        import psycopg2

        conn = psycopg2.connect(config.get("database.url"))
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE tasks SET session_id = %s WHERE task_id = %s",
                    (session_id, task_id),
                )
                conn.commit()
        finally:
            conn.close()

    # Long-term memory stays scoped to the user (shared with chat agent).
    memory = Memory(
        user_id=user_id,
        session_id=session_id,
        db_url=config.get("database.url"),
        use_long_term=bool(config.get("memory.use_long_term", False)),
    )

    llm, shared_tm = _shared_deps()
    tool_manager = ScopedToolManager(shared_tm, allowed=required_tools) if shared_tm else None

    flow = StateHandler(yaml_path=EXECUTOR_GRAPH_PATH)
    subagent = Agent(
        agent_id=f"task-{task_id}",
        flow=flow,
        memory=memory,
        llm=llm,
        tool_manager=tool_manager,
    )
    subagent.system_prompt = EXECUTOR_SYSTEM_PROMPT
    subagent.task_id = task_id
    subagent.plan_steps = plan_steps
    subagent.step_idx = 0
    subagent.current_user_id = user_id
    # Generous iteration cap so multi-step plans don't trip MAX_ITER.
    subagent.max_iter = max(10, len(plan_steps) * 4 + 8)

    set_task_status(task_id, "running")
    log_event(
        task_id,
        "started",
        title,
        payload={"plan_steps": plan_steps, "required_tools": required_tools},
    )

    # Seed the subagent's memory with the plan so the executor state's LLM
    # call has context beyond just the system prompt.
    kickoff = [
        SystemMessage(content=EXECUTOR_SYSTEM_PROMPT),
        UserMessage(
            content=(
                f"Execute this approved plan titled '{title}'. Steps:\n"
                + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(plan_steps))
            )
        ),
    ]

    # Run the FSM. The executor graph loops pick_step -> use_tool -> pick_step
    # until all steps are handled, then transitions to executor_done (terminal).
    final_output = await subagent.step(kickoff, user_id=user_id)

    summary = ""
    if final_output:
        sd = getattr(final_output, "structured_data", {}) or {}
        summary = sd.get("summary") or final_output.content or ""

    mark_task_completed(task_id, summary=summary)
    log_event(task_id, "completed", summary)


def spawn(task_id: str) -> asyncio.Task:
    """Schedule run_task on the current event loop. Returns the asyncio.Task."""
    existing = _RUNNING.get(task_id)
    if existing and not existing.done():
        return existing
    t = asyncio.create_task(run_task(task_id), name=f"task-{task_id}")
    _RUNNING[task_id] = t
    return t


def cancel(task_id: str) -> bool:
    """Cancel a running subagent. Returns True if there was something to cancel."""
    t = _RUNNING.get(task_id)
    if t and not t.done():
        t.cancel()
        return True
    return False


def is_running(task_id: str) -> bool:
    t = _RUNNING.get(task_id)
    return bool(t and not t.done())


async def sweep_orphans() -> int:
    """
    On startup, find tasks that were left in 'running' or 'awaiting_approval'
    and have no live asyncio.Task. Respawn them.
    Returns the number of tasks resumed.
    """
    import psycopg2
    import psycopg2.extras

    conn = psycopg2.connect(config.get("database.url"))
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT task_id
                FROM tasks
                WHERE status IN ('running', 'awaiting_approval')
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    resumed = 0
    for row in rows:
        tid = str(row["task_id"])
        if is_running(tid):
            continue
        log_event(tid, "resumed", "spawning runner after restart")
        spawn(tid)
        resumed += 1
    return resumed
