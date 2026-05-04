"""
Executor state. Runs inside a subagent (TaskRunner) and walks plan_steps one
at a time. For each step it decides whether the next action is a tool call
or a human question, records the decision, and forces the FSM transition.

Never re-plans. If a step can't be handled by the available tools, it routes
to ask_human instead of silently deviating.
"""

from __future__ import annotations

import os
import sys
from enum import StrEnum

from pydantic import BaseModel, Field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from base_module.task_store import log_event  # noqa: E402
from model_module.ArkModelNew import SystemMessage  # noqa: E402
from state_module.core.base_state import StateOutput  # noqa: E402
from state_module.core.state import State  # noqa: E402
from state_module.core.state_registry import register_state  # noqa: E402


class _ActionKind(StrEnum):
    tool = "tool"
    ask = "ask"


class _AskKind(StrEnum):
    binary = "binary"
    text = "text"


class ExecutorDecision(BaseModel):
    """Structured choice for the current plan step."""

    action: _ActionKind = Field(
        ...,
        description="tool if the step can be performed by calling a tool, ask if human input is required",
    )
    reason: str = Field(..., description="One-sentence justification")

    # populated when action == tool
    tool_name: str | None = Field(None, description="Exact tool name from the tool list")
    tool_args: dict | None = Field(None, description="Arguments dict for the tool")

    # populated when action == ask
    ask_kind: _AskKind | None = Field(None, description="binary (approve/deny) or text (free-form answer)")
    ask_prompt: str | None = Field(None, description="Exact question to present to the user")


@register_state
class StateExecutor(State):
    """Subagent decision state. Picks the next step's action deterministically."""

    type = "executor"

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.is_terminal = False

    def check_transition_ready(self, context):
        return True

    async def run(self, context, agent=None):
        plan_steps: list[str] = getattr(agent, "plan_steps", []) or []
        step_idx: int = getattr(agent, "step_idx", 0)
        task_id: str | None = getattr(agent, "task_id", None)

        # --- no more steps -> done ------------------------------------------
        if step_idx >= len(plan_steps):
            return StateOutput(
                content="",
                completion_signal="complete",
                structured_data={"next_state": "executor_done"},
            )

        current_step = plan_steps[step_idx]

        # Describe available tools to the LLM. Keep the description compact.
        tool_lines: list[str] = []
        tool_names: list[str] = []
        if agent.tool_manager is not None:
            try:
                servers = await agent.tool_manager.list_all_tools()
                for _server, tools in servers.items():
                    for tname, tspec in tools.items():
                        tool_names.append(tname)
                        desc = ""
                        if isinstance(tspec, dict):
                            desc = tspec.get("description", "") or ""
                        else:
                            desc = getattr(tspec, "description", "") or ""
                        tool_lines.append(f"- {tname}: {desc[:160]}")
            except Exception as e:
                log_event(task_id, "error", f"could not list tools: {e}") if task_id else None

        tools_block = "\n".join(tool_lines) if tool_lines else "(no tools available)"

        system = SystemMessage(
            content=(
                "You are a subagent executing a plan one step at a time.\n"
                "You are NOT allowed to re-plan or skip steps.\n"
                "For the CURRENT step, decide if it can be performed by calling a tool, "
                "or if you need to ask the human a question.\n"
                "Prefer ask=binary when the question is a simple yes/no (confirmation).\n"
                "Prefer ask=text when the question is open-ended or needs data from the user.\n"
                "If a step can only be partially handled by tools, ask the human.\n"
                "Never invent tool names. Pick from the list below or set action=ask.\n\n"
                f"Available tools:\n{tools_block}\n\n"
                f"Current plan step ({step_idx + 1}/{len(plan_steps)}): {current_step}\n"
            )
        )

        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "executor_decision",
                "schema": ExecutorDecision.model_json_schema(),
            },
        }

        output = await agent.call_llm(context=[system], json_schema=schema)
        try:
            decision = ExecutorDecision.model_validate_json(output.content if output else "")
        except Exception as e:
            if task_id:
                log_event(task_id, "error", f"executor decision parse failed: {e}")
            # Default to asking the human so the subagent doesn't crash silently
            decision = ExecutorDecision(
                action=_ActionKind.ask,
                reason=f"could not parse decision: {e}",
                ask_kind=_AskKind.text,
                ask_prompt=f"I'm stuck on step {step_idx + 1}: {current_step}. How should I proceed?",
            )

        if task_id:
            log_event(
                task_id,
                "step_started",
                current_step,
                payload={"step_idx": step_idx, "decision": decision.model_dump(mode="json")},
            )

        if decision.action == _ActionKind.tool:
            # Validate tool name against the live list so we don't send garbage
            if not decision.tool_name or (tool_names and decision.tool_name not in tool_names):
                # fall back to asking the human if the LLM picked a fake tool
                agent.pending_ask = {
                    "kind": "text",
                    "prompt": (
                        f"I need to do: {current_step}\nBut I don't have a tool that matches. How should I handle this?"
                    ),
                }
                if task_id:
                    log_event(
                        task_id,
                        "fallback_ask",
                        "invalid tool name from LLM",
                        payload={"llm_choice": decision.tool_name},
                    )
                return StateOutput(
                    content="",
                    completion_signal="incomplete",
                    structured_data={"next_state": "ask_human"},
                )

            agent.pending_tool = {
                "tool_name": decision.tool_name,
                "tool_args": decision.tool_args or {},
            }
            return StateOutput(
                content=f"calling tool `{decision.tool_name}` for step {step_idx + 1}",
                completion_signal="incomplete",
                structured_data={"next_state": "use_tool"},
            )

        # action == ask
        agent.pending_ask = {
            "kind": (decision.ask_kind or _AskKind.text).value,
            "prompt": decision.ask_prompt or current_step,
        }
        return StateOutput(
            content="",
            completion_signal="incomplete",
            structured_data={"next_state": "ask_human"},
        )
