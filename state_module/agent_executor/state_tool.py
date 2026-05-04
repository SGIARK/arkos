import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from base_module.task_store import log_event  # noqa: E402
from state_module.core.base_state import StateOutput  # noqa: E402
from state_module.core.state import State  # noqa: E402
from state_module.core.state_registry import register_state  # noqa: E402
from tool_module.tool_call import AuthRequiredError  # noqa: E402


@register_state
class StateExecutorTool(State):
    """Executor-path tool state. Executes a tool that was pre-selected by StateExecutor.
    Never uses the LLM to choose a tool — that decision was already made."""

    type = "executor_tool"

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.is_terminal = False

    def check_transition_ready(self, context):
        return True

    async def run(self, context, agent=None):
        task_id = getattr(agent, "task_id", None)
        pending = getattr(agent, "pending_tool", None)

        if not pending or not pending.get("tool_name"):
            return StateOutput(
                content="(executor_tool reached without a pending tool; skipping step)",
                completion_signal="error",
                error_detail="missing pending_tool on subagent",
                structured_data={"next_state": "executor"},
            )

        tool_name = pending["tool_name"]
        tool_args = pending.get("tool_args") or {}
        agent.pending_tool = None

        try:
            if task_id:
                log_event(
                    task_id,
                    "tool_call",
                    tool_name,
                    payload={"args": tool_args},
                )

            tool_result = await agent.tool_manager.call_tool(
                tool_name=tool_name,
                arguments=tool_args,
                user_id=agent.current_user_id,
            )

            if task_id:
                log_event(
                    task_id,
                    "tool_result",
                    str(tool_result),
                    payload={"tool_name": tool_name},
                )

            agent.step_idx = getattr(agent, "step_idx", 0) + 1

            return StateOutput(
                content=f"tool `{tool_name}` -> {tool_result}",
                completion_signal="complete",
                structured_data={"tool_result": tool_result, "next_state": "executor"},
            )

        except AuthRequiredError as e:
            service_label = e.service_info.get("name", e.service) if getattr(e, "service_info", None) else e.service
            if task_id:
                log_event(task_id, "error", f"auth required for {service_label}")
            return StateOutput(
                content=f"Auth required for {service_label}.",
                completion_signal="error",
                structured_data={"next_state": "executor_done"},
            )
