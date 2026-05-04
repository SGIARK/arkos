"""
Chat/reasoning state for buddy.

This is the default state on the main chat graph. It owns three responsibilities:

1. Respond to the user in natural language when the question is conversational
   (no external action needed).
2. Workshop the user's request in chat when it's vague, until a concrete plan
   is in view. This is where buddy asks clarifying questions.
3. Decide to hand off to `workshop_plan` ONLY when the user has clearly asked
   for a concrete multi-step action on an external system.

Key fixes landed here:
- The state now sees the full tool listing (via agent.system_prompt) so buddy
  doesn't falsely claim "I don't have access to Linear" when the Linear MCP
  is wired up.
- The state's own reasoning "approach" is NOT shown to the user. It stays
  internal. Only `final` (the user-facing message) is returned as content.
- The state picks its own next_state via structured_data.next_state instead of
  leaving the decision to the LLM-based choose_transition pass. That stops
  buddy from jumping into workshop_plan every time "external systems" is
  mentioned in passing.
"""

from __future__ import annotations

import os
import sys
from enum import StrEnum

from pydantic import BaseModel, Field

from model_module.ArkModelNew import SystemMessage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from state_module.core.base_state import StateOutput  # noqa: E402
from state_module.core.state import State  # noqa: E402
from state_module.core.state_registry import register_state  # noqa: E402


class _Route(StrEnum):
    reply = "reply"  # stay in chat; just say `final`
    ask = "ask"  # stay in chat; ask a clarifying question
    plan = "plan"  # hand off to workshop_plan


class ReasonedOutput(BaseModel):
    """Routing + user-facing reply. Chain-of-thought stays internal."""

    intent: str = Field(..., description="What buddy thinks the user is trying to do (internal)")
    approach: list[str] = Field(default_factory=list, description="Internal reasoning. NEVER shown to user.")
    route: _Route = Field(
        ...,
        description=(
            "reply = simple chat answer. "
            "ask = need a clarification from the user, include it in final. "
            "plan = the user has asked for a concrete multi-step action buddy should execute. "
            "Only choose plan when the user has explicitly asked buddy to DO something "
            "concrete involving external systems or tools and the goal is clear enough to "
            "write a plan for."
        ),
    )
    final: str = Field(..., description="The ONLY text shown to the user")


@register_state
class StateAI(State):
    type = "agent"

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.is_terminal = False

    def check_transition_ready(self, context):
        return True

    async def run(self, context, agent):
        messages = context if isinstance(context, list) else context.get("messages", [])

        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "reasoned_output",
                "schema": ReasonedOutput.model_json_schema(),
            },
        }

        # Start from the agent's root system prompt (which carries the live
        # tool listing so buddy knows Linear/Gmail/Calendar/etc exist). Fall
        # back to an empty string if the startup hook hasn't populated it.
        root_prompt = (getattr(agent, "system_prompt", None) or "").strip()

        chat_guidance = (
            "You are buddy, the ark chat agent.\n"
            "Read the user's latest message and pick a route:\n"
            "  - reply: answer them in chat. This is the default.\n"
            "  - ask:   ask ONE clarifying question if you need more info.\n"
            "  - plan:  ONLY when the user has explicitly asked you to DO a concrete, "
            "multi-step action (create, send, schedule, update, etc) on an external "
            "system where the target of the action is already clear.\n"
            "\n"
            "Workshop ideas in chat FIRST. Never jump to plan just because the user "
            "mentioned an external system. If they say 'check my linear tickets', answer "
            "in chat using the tools you have; do NOT write a plan for a single tool call. "
            "If you truly do not have a tool that would help, say so plainly and ask the "
            "user what they'd like to do instead (route=ask).\n"
            "\n"
            "Put your reasoning in `approach`. The user will NEVER see it. Only `final` "
            "is shown. Do not paraphrase your reasoning into the final message."
        )

        system_parts: list[str] = []
        if root_prompt:
            system_parts.append(root_prompt)
        system_parts.append(chat_guidance)

        system = SystemMessage(content="\n\n".join(system_parts))

        llm_context = [system] + messages
        output = await agent.call_llm(context=llm_context, json_schema=json_schema)

        if not output or not output.content:
            return StateOutput(
                content="I had trouble forming a response. Could you rephrase?",
                completion_signal="error",
                error_detail="LLM returned empty content",
                structured_data={"next_state": "ask_user"},
            )

        try:
            data = ReasonedOutput.model_validate_json(output.content)
        except Exception as e:
            # Soft fallback: surface the raw text and stay in chat.
            print(f"[state_ai] schema parse failed: {e}")
            return StateOutput(
                content=output.content,
                completion_signal="complete",
                structured_data={"next_state": "ask_user"},
            )

        user_text = (data.final or "").strip()
        if not user_text:
            user_text = "(no content)"

        # Deterministic routing. The chat graph has [ask_user, use_tool, workshop_plan]
        # as transitions; we map reply/ask to ask_user (hand back to the user) and
        # plan to workshop_plan. use_tool is NEVER chosen from this state directly,
        # that path is reserved for post-approval execution.
        if data.route == _Route.plan:
            next_state = "workshop_plan"
            signal = "complete"
        elif data.route == _Route.ask:
            next_state = "ask_user"
            signal = "needs_input"
        else:
            next_state = "ask_user"
            signal = "complete"

        return StateOutput(
            content=user_text,
            completion_signal=signal,
            structured_data={"next_state": next_state, "route": data.route.value},
        )
