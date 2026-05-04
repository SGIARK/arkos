"""
Approval/ask-human state. Writes a task_approvals row, flips the parent task
to awaiting_approval, then DB-polls until the user responds.

Supports two kinds:
  binary  -> the UI shows approve/decline buttons
  text    -> the UI shows a textarea; whatever the user types becomes the answer

DB polling is intentional: it survives web-process restarts because the state
can be rehydrated from the DB row.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from base_module.task_store import (  # noqa: E402
    create_approval,
    get_approval,
    get_task,
    log_event,
    set_task_status,
)
from config_module.loader import config  # noqa: E402
from model_module.ArkModelNew import UserMessage  # noqa: E402
from state_module.core.base_state import StateOutput  # noqa: E402
from state_module.core.state import State  # noqa: E402
from state_module.core.state_registry import register_state  # noqa: E402


@register_state
class StateApproval(State):
    """Persists an approval request, then polls the DB until it is resolved."""

    type = "approval"

    def __init__(self, name: str, config_dict: dict):
        super().__init__(name, config_dict)
        self.is_terminal = False

    def check_transition_ready(self, context):
        return True

    @property
    def _poll_interval(self) -> float:
        try:
            return float(config.get("approval.poll_interval_seconds") or 2.0)
        except Exception:
            return 2.0

    @property
    def _poll_timeout(self) -> float:
        try:
            return float(config.get("approval.poll_timeout_seconds") or 60 * 60 * 24)
        except Exception:
            return 60 * 60 * 24

    async def run(self, context, agent=None):
        task_id = getattr(agent, "task_id", None)
        pending = getattr(agent, "pending_ask", None) or {}
        prompt = pending.get("prompt") or "The subagent needs your input."
        kind = pending.get("kind") or "text"
        if kind not in ("binary", "text"):
            kind = "text"

        if not task_id:
            return StateOutput(
                content="(approval state reached without task_id; aborting)",
                completion_signal="error",
                error_detail="missing task_id on subagent",
                structured_data={"next_state": "executor_done"},
            )

        task_row = get_task(task_id)
        if not task_row:
            return StateOutput(
                content="(task row missing)",
                completion_signal="error",
                error_detail="task row disappeared",
                structured_data={"next_state": "executor_done"},
            )

        user_id = str(task_row["user_id"])

        approval_id = create_approval(
            task_id=task_id,
            user_id=user_id,
            kind=kind,
            prompt=prompt,
            context={"plan_step_idx": getattr(agent, "step_idx", 0)},
        )

        set_task_status(task_id, "awaiting_approval")
        log_event(
            task_id,
            "approval_requested",
            prompt,
            payload={"approval_id": approval_id, "kind": kind},
        )

        # Poll the DB until resolved (or timeout)
        interval = self._poll_interval
        deadline_left = self._poll_timeout
        resolved = None
        while deadline_left > 0:
            await asyncio.sleep(interval)
            deadline_left -= interval
            row = get_approval(approval_id)
            if row and row["status"] != "pending":
                resolved = row
                break
            # also bail if the whole task got cancelled from the UI
            task_now = get_task(task_id)
            if task_now and task_now["status"] in ("cancelled", "failed"):
                log_event(task_id, "approval_aborted", "task was cancelled while awaiting approval")
                return StateOutput(
                    content="(task was cancelled)",
                    completion_signal="complete",
                    structured_data={"next_state": "executor_done"},
                )

        if not resolved:
            log_event(task_id, "approval_timeout", prompt, payload={"approval_id": approval_id})
            return StateOutput(
                content="(approval timed out)",
                completion_signal="error",
                structured_data={"next_state": "executor_done"},
            )

        # Clear the pending ask so the next executor iteration re-evaluates
        agent.pending_ask = None

        if resolved["kind"] == "binary":
            answer_text = "user approved" if resolved["response_bool"] else "user declined"
        else:
            answer_text = resolved["response_text"] or ""

        # Record the human answer as a UserMessage so the next step's LLM sees it
        try:
            await agent.memory.add_memory(UserMessage(content=f"[human answer for '{prompt}']: {answer_text}"))
        except Exception as e:
            log_event(task_id, "error", f"could not write answer to memory: {e}")

        log_event(
            task_id,
            "approval_resolved",
            answer_text,
            payload={
                "approval_id": approval_id,
                "kind": resolved["kind"],
                "status": resolved["status"],
            },
        )

        # Declined binary approval => end the task
        if resolved["kind"] == "binary" and not resolved["response_bool"]:
            return StateOutput(
                content=f"User declined: {prompt}",
                completion_signal="complete",
                structured_data={"next_state": "executor_done", "declined": True},
            )

        # Approved or answered: advance past this plan step and loop back to executor
        agent.step_idx = getattr(agent, "step_idx", 0) + 1
        set_task_status(task_id, "running")

        return StateOutput(
            content=f"Got answer: {answer_text}",
            completion_signal="complete",
            structured_data={"next_state": "executor"},
        )
