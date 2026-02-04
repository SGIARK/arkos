import sys
import os

from typing import Optional, List
from pydantic import BaseModel, Field


from model_module.ArkModelNew import ArkModelLink, UserMessage, AIMessage, SystemMessage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from state_module.state import State


from state_module.state_registry import register_state



class ReasonedOutput(BaseModel):
    """
    Enforced reasoning contract for the agent state.
    No tools, no chain-of-thought.
    """
    intent: str = Field(..., description="What the agent is trying to accomplish")
    approach: List[str] = Field(..., description="High-level reasoning steps")
    needs_clarification: bool = Field(..., description="Whether more user input is required")
    clarifying_question: Optional[str] = Field(
        None, description="Single clarifying question if needed"
    )
    has_tool_result: bool = Field(
        False, description="Whether responding with a tool result (if true, final should present the tool output)"
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
        """
        Pure reasoning state.
        - One LLM call
        - Structured reasoning enforced via schema
        - No tools
        - No recovery heuristics
        """

        messages = context if isinstance(context, list) else context.get("messages", [])

        # Check if there's a tool result in the context (SystemMessage from tool execution)
        has_tool_result = any(
            isinstance(msg, SystemMessage) and
            msg.content and
            not msg.content.startswith("You are")  # Skip system prompts
            for msg in messages
        )
        if has_tool_result:
            print(f"[StateAI] TOOL RESULT DETECTED in context - instructing LLM to use it")

        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "reasoned_output",
                "schema": ReasonedOutput.model_json_schema(),
            },
        }

        # Build system prompt with tool result detection
        tool_result_guidance = ""
        if has_tool_result:
            tool_result_guidance = (
                "\n\n🚨 CRITICAL: A SystemMessage with TOOL_RESULT is present in the conversation. "
                "You MUST:\n"
                "1. Set has_tool_result=true\n"
                "2. Extract data from the TOOL_RESULT SystemMessage\n"
                "3. Present that data in the 'final' field\n"
                "4. Do NOT make up data, ONLY use what's in the tool result\n"
                "5. Do NOT include reasoning steps (approach will be hidden)\n"
            )

        system = SystemMessage(
            content=(
                "You are the agent reasoning state.\n"
                "No tools are available in THIS state.\n"
                "Never repeat yourself.\n"
                "Produce a JSON object matching the provided schema.\n"
                "Do not reveal chain-of-thought.\n"
                "Use concise, high-level reasoning steps only."
                f"{tool_result_guidance}"
            )
        )

        llm_context = [system] + messages
        output = await agent.call_llm(context=llm_context, json_schema=json_schema)
        print("Reasoned Output: \n\n", output)

        # Handle None or empty content
        if not output or not output.content:
            return AIMessage(content="I encountered an issue processing your request. Please try again.")

        try:
            data = ReasonedOutput.model_validate_json(output.content)
            print(f"[StateAI] Parsed output - has_tool_result={data.has_tool_result}")
        except Exception as e:
            # If JSON parsing fails, return the raw content as fallback
            print(f"Failed to parse structured output: {e}")
            return AIMessage(content=output.content)






        # Build response including the approach/reasoning
        response_parts = []

        # If presenting tool results, skip showing reasoning steps
        if not data.has_tool_result:
            # Include approach if it has substantive content
            if data.approach:
                for step in data.approach:
                    response_parts.append(f"• {step}")

        # Add final answer
        if data.final:
            if response_parts:
                response_parts.append("")  # blank line
            response_parts.append(data.final)

        # Add clarifying question if needed
        if data.needs_clarification and data.clarifying_question:
            response_parts.append("")
            response_parts.append(data.clarifying_question)

        response = "\n".join(response_parts) if response_parts else data.final
        return AIMessage(content=response)

