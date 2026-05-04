"""
Terminal state for the executor graph. Writes a short summary event and
returns. TaskRunner is the one that flips tasks.status to 'completed'
after step() returns, so this state's job is just to produce the final
user-visible message.
"""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from base_module.task_store import log_event  # noqa: E402
from state_module.core.base_state import StateOutput  # noqa: E402
from state_module.core.state import State  # noqa: E402
from state_module.core.state_registry import register_state  # noqa: E402


@register_state
class StateExecutorDone(State):
    type = "executor_done"

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.is_terminal = True

    def check_transition_ready(self, context):
        return True

    async def run(self, context, agent=None):
        from model_module.ArkModelNew import SystemMessage

        task_id = getattr(agent, "task_id", None)
        plan_steps = getattr(agent, "plan_steps", []) or []
        step_idx = getattr(agent, "step_idx", 0)

        # Ask the LLM to produce a user-facing summary of what was actually done.
        try:
            system = SystemMessage(
                content=(
                    "You are summarising the results of a completed task for the user.\n"
                    "Based on the conversation above (which contains tool call results), "
                    "write a clear, concise summary of what was found or accomplished. "
                    "Include the actual data returned (e.g. event names, times, titles). "
                    "Do not say 'the task is complete' — just present the results naturally. "
                    "Keep it under 200 words."
                )
            )
            output = await agent.call_llm(context=list(context) + [system], json_schema=None)
            summary = (output.content or "").strip() if output else ""
        except Exception as e:
            summary = ""
            if task_id:
                log_event(task_id, "error", f"summary LLM call failed: {e}")

        if not summary:
            summary = (
                f"Finished all {len(plan_steps)} plan steps."
                if step_idx >= len(plan_steps)
                else f"Stopped at step {step_idx + 1} of {len(plan_steps)}."
            )

        if task_id:
            log_event(task_id, "done", summary, payload={"step_idx": step_idx, "total": len(plan_steps)})

        return StateOutput(
            content=summary,
            completion_signal="complete",
            structured_data={"summary": summary},
        )
