import sys
import os

from typing import Optional, List
from pydantic import BaseModel, Field


from model_module.ArkModelNew import ArkModelLink, UserMessage, AIMessage, SystemMessage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from state_module.state import State
from state_module.result_formatters import format_tool_result


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
        False, description="True ONLY if presenting data from a TOOL_RESULT SystemMessage in this conversation. False if reasoning about what to do or answering without tool data."
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

        # Check if there's a FRESH tool result (last message in context)
        # Only flag if the most recent message is a tool result (not old ones from memory)
        has_tool_result = False
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, SystemMessage):
                print(f"[StateAI] Last message is SystemMessage, content preview: {last_msg.content[:100]}")
            if (isinstance(last_msg, SystemMessage) and
                last_msg.content and
                last_msg.content.startswith("TOOL_RESULT from")):
                has_tool_result = True
                print(f"[StateAI] FRESH TOOL RESULT detected (last message) - instructing LLM to use it")
            else:
                print(f"[StateAI] No fresh tool result (last msg type: {type(last_msg).__name__})")

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
                "You are the agent reasoning state. Respond with a JSON object matching the schema.\n"
                "Set has_tool_result=false unless a TOOL_RESULT SystemMessage exists in the conversation.\n"
                "NEVER fabricate tool results or external data (calendar events, search results, etc.).\n"
                "If a TOOL_RESULT is present, extract and present that data in 'final'."
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
            print(f"[StateAI] LLM set has_tool_result={data.has_tool_result}")

            # OVERRIDE: If we detected tool result but LLM didn't flag it, force it
            if has_tool_result and not data.has_tool_result:
                print(f"[StateAI] ⚠️  Overriding has_tool_result to True (LLM ignored instructions)")
                data.has_tool_result = True

        except Exception as e:
            # If JSON parsing fails, return user-friendly error instead of raw JSON
            print(f"[StateAI] Failed to parse structured output: {e}")
            print(f"[StateAI] Raw output (first 200 chars): {output.content[:200]}")
            return AIMessage(content="I encountered an issue processing that request. Could you rephrase it?")






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

            # Check if final contains raw JSON and format it
            final_content = data.final
            if final_content.strip().startswith('{"') or final_content.strip().startswith("{\\"):
                try:
                    # Try to parse and format the JSON
                    import json
                    parsed = json.loads(final_content)
                    print(f"[StateAI] Detected raw JSON in final field, formatting...")

                    # Use formatter registry for generic formatting
                    final_content = format_tool_result(parsed)
                except Exception as e:
                    print(f"[StateAI] Failed to parse/format JSON: {e}")
                    pass  # Not valid JSON, use as-is

            response_parts.append(final_content)

        # Add clarifying question if needed
        if data.needs_clarification and data.clarifying_question:
            response_parts.append("")
            response_parts.append(data.clarifying_question)

        response = "\n".join(response_parts) if response_parts else data.final
        return AIMessage(content=response)

