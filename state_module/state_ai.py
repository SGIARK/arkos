import sys
import os
import json
import logging

from typing import Optional, List
from pydantic import BaseModel, Field

from model_module.ArkModelNew import UserMessage, AIMessage, SystemMessage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from state_module.state import State
from state_module.state_registry import register_state

logger = logging.getLogger(__name__)


class ReasonedOutput(BaseModel):
    """Enforced reasoning contract for the agent state."""
    intent: str = Field(..., description="What the agent is trying to accomplish")
    approach: List[str] = Field(..., description="High-level reasoning steps")
    needs_clarification: bool = Field(..., description="Whether more user input is required")
    clarifying_question: Optional[str] = Field(
        None, description="Single clarifying question if needed"
    )
    has_tool_result: bool = Field(
        False,
        description="True ONLY if presenting data from a TOOL_RESULT SystemMessage in this conversation."
    )
    final: str = Field(..., description="User-facing response")


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

        # Detect a fresh tool result as the last message
        has_tool_result = (
            bool(messages)
            and isinstance(messages[-1], SystemMessage)
            and messages[-1].content
            and messages[-1].content.startswith("TOOL_RESULT from")
        )

        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "reasoned_output",
                "schema": ReasonedOutput.model_json_schema(),
            },
        }

        tool_result_guidance = ""
        if has_tool_result:
            tool_result_guidance = (
                "\n\nCRITICAL: A TOOL_RESULT SystemMessage is present. You MUST:\n"
                "1. Set has_tool_result=true\n"
                "2. Extract and present the data from the TOOL_RESULT in 'final'\n"
                "3. Do NOT fabricate data — only use what the tool returned\n"
            )

        system = SystemMessage(
            content=(
                "You are the agent reasoning state. Respond with a JSON object matching the schema.\n"
                "Set has_tool_result=false unless a TOOL_RESULT SystemMessage exists in the conversation.\n"
                "NEVER fabricate tool results or external data (calendar events, search results, etc.).\n"
                "If a TOOL_RESULT is present, extract and present that data in 'final'."
                f"{tool_result_guidance}"
            )
        )

        llm_context = [system] + messages
        output = await agent.call_llm(context=llm_context, json_schema=json_schema)

        if not output or not output.content:
            return AIMessage(content="I encountered an issue processing your request. Please try again.")

        try:
            data = ReasonedOutput.model_validate_json(output.content)
            if has_tool_result and not data.has_tool_result:
                logger.warning("LLM did not set has_tool_result=true despite tool result present — overriding")
                data.has_tool_result = True
        except Exception as e:
            logger.error("Failed to parse structured output: %s | Raw: %s", e, output.content[:200])
            return AIMessage(content="I encountered an issue processing that request. Could you rephrase it?")

        response_parts = []

        if not data.has_tool_result and data.approach:
            for step in data.approach:
                response_parts.append(f"• {step}")

        if data.final:
            if response_parts:
                response_parts.append("")

            response_parts.append(data.final)

        if data.needs_clarification and data.clarifying_question:
            response_parts.append("")
            response_parts.append(data.clarifying_question)

        response = "\n".join(response_parts) if response_parts else data.final
        return AIMessage(content=response)
