import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from model_module.ArkModelNew import SystemMessage
from tool_module.tool_call import AuthRequiredError

from state_module.state import State
from state_module.state_registry import register_state


@register_state
class StateTool(State):
    type = "tool"

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.is_terminal = False

    def check_transition_ready(self, context):
        return True

    async def choose_tool(self, context, agent):
        """
        Chooses tool to use based on the context and server
        """
        print(f"[StateTool] Phase 1: Choosing tool...")

        # Get all tools with descriptions
        all_tools = await agent.tool_manager.list_all_tools()
        tool_descriptions = []
        for server_name, tools in all_tools.items():
            for tool_name, tool_spec in tools.items():
                desc = tool_spec.get('description', 'No description')
                tool_descriptions.append(f"- {tool_name}: {desc}")

        tools_list = "\n".join(tool_descriptions)
        prompt = f"""Based on the user request, choose the tool that best satisfies it.

Available tools:
{tools_list}

Choose the most appropriate tool."""

        instructions = context + [SystemMessage(content=prompt)]

        # Get Pydantic class and convert to JSON schema format
        tool_option_class = await agent.create_tool_option_class()
        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "tool_choice",
                "schema": tool_option_class.model_json_schema(),
            },
        }

        # Call LLM and parse response
        try:
            output = await agent.call_llm(instructions, json_schema)
            print(f"[StateTool] Tool choice output: {output.content[:100]}")
            structured_output = json.loads(output.content)
            tool_name = structured_output["tool_name"]
            print(f"[StateTool] Selected tool: {tool_name}")
        except json.JSONDecodeError as e:
            print(f"[StateTool] ERROR in tool selection: {e}")
            print(f"[StateTool] Full output:")
            print(f"---")
            print(output.content if output else "None")
            print(f"---")
            raise

        server_name = agent.tool_manager._tool_registry[tool_name]

        all_tools = await agent.tool_manager.list_all_tools()
        tool_spec = all_tools[server_name][tool_name]

        # Build schema for tool arguments
        tool_args_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "tool_args",
                "schema": tool_spec.get("inputSchema", {}),
            },
        }

        # Build context-aware args prompt
        args_guidance = """Fill in the arguments for the tool '{tool_name}' based on the user's request.

IMPORTANT DATE/TIME FORMATTING:
- For any date/time fields (timeMin, timeMax, start, end, etc.), use ISO 8601 format with timezone
- Format: YYYY-MM-DDTHH:MM:SS (e.g., "2026-02-04T00:00:00")
- For "today", use the current date with 00:00:00 for start and 23:59:59 for end
- For date ranges, ensure both start and end times are properly formatted"""

        # Add calendar-specific guidance
        if "calendar" in tool_name.lower() or tool_name in ["manage-accounts", "list-events", "create-event", "list-calendars"]:
            args_guidance += """

🚨 CRITICAL CALENDAR ACCOUNT GUIDANCE:
- The ONLY valid account ID is: "normal" (do NOT make up account IDs)
- For manage-accounts with action="list": OMIT the account_id parameter entirely (set to null or don't include it)
- For list-events, list-calendars, create-event: Use account="normal" or account_id="normal"
- NEVER use placeholder values like "user-account-12345" or "user_google_calendar..."
- If unsure, use "normal" as the account identifier"""

        args_prompt = args_guidance.format(tool_name=tool_name)

        # Log if calendar guidance is applied
        if "CRITICAL CALENDAR" in args_prompt:
            print(f"[StateTool] Applied calendar account guidance for '{tool_name}'")

        # Use full context for arg generation
        args_context = context + [SystemMessage(content=args_prompt)]

        print(f"[StateTool] Context length for args: {len(args_context)} messages")

        args_output = await agent.call_llm(args_context, tool_args_schema)

        # Handle None or empty response with retry
        if not args_output or not args_output.content:
            print(f"[StateTool] WARN: LLM returned None/empty for tool args, retrying with simpler prompt")
            # Retry with minimal context (just the user's last message)
            simple_context = [context[-1], SystemMessage(content=args_prompt)]
            args_output = await agent.call_llm(simple_context, tool_args_schema)

            if not args_output or not args_output.content:
                print(f"[StateTool] ERROR: LLM returned None/empty even after retry")
                raise ValueError("Failed to generate tool arguments after retry")

        try:
            tool_args = json.loads(args_output.content)
        except json.JSONDecodeError as e:
            print(f"[StateTool] ERROR: Invalid JSON from LLM: {e}")
            print(f"[StateTool] Full LLM output:")
            print(f"---")
            print(args_output.content)
            print(f"---")

            # Try to extract partial JSON or provide defaults
            print(f"[StateTool] Attempting to use default values for missing fields")
            # For calendar tools, provide sensible defaults
            if tool_name == "list-events":
                from datetime import datetime, timedelta
                today = datetime.now().strftime("%Y-%m-%dT00:00:00")
                tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00")
                tool_args = {
                    "account": "normal",
                    "timeMin": today,
                    "timeMax": tomorrow
                }
                print(f"[StateTool] Using fallback args: {tool_args}")
            else:
                raise

        print(f"[StateTool] Calling '{tool_name}' with args: {json.dumps(tool_args, indent=2)}")

        return {"tool_name": tool_name, "tool_args": tool_args}

    async def execute_tool(self, tool_call, agent):
        """
        Parses and fills args for chosen tool for tool call execution
        """
        tool_name = tool_call["tool_name"]
        tool_args = tool_call["tool_args"]

        tool_result = await agent.tool_manager.call_tool(
            tool_name=tool_name,
            arguments=tool_args,
            user_id=agent.current_user_id,
        )

        return tool_result

    async def run(self, context, agent=None):
        try:
            tool_arg_dict = await self.choose_tool(context=context, agent=agent)
            tool_result = await self.execute_tool(tool_call=tool_arg_dict, agent=agent)

            # Format tool result with clear marker
            tool_name = tool_arg_dict.get("tool_name", "unknown")

            # Extract actual content from MCP format if present
            if isinstance(tool_result, dict) and "content" in tool_result:
                # MCP format: {"content": [{"type": "text", "text": "..."}]}
                content_items = tool_result.get("content", [])
                if content_items and isinstance(content_items, list):
                    # Extract text from first content item
                    first_item = content_items[0]
                    if isinstance(first_item, dict) and "text" in first_item:
                        actual_data = first_item["text"]
                        # Try to parse as JSON for pretty printing
                        try:
                            parsed = json.loads(actual_data)
                            formatted_result = f"TOOL_RESULT from '{tool_name}':\n{json.dumps(parsed, indent=2)}"
                        except:
                            formatted_result = f"TOOL_RESULT from '{tool_name}':\n{actual_data}"
                    else:
                        formatted_result = f"TOOL_RESULT from '{tool_name}':\n{json.dumps(tool_result, indent=2)}"
                else:
                    formatted_result = f"TOOL_RESULT from '{tool_name}':\n{json.dumps(tool_result, indent=2)}"
            else:
                formatted_result = f"TOOL_RESULT from '{tool_name}':\n{json.dumps(tool_result, indent=2)}"

            print(f"[StateTool] Returning tool result ({len(formatted_result)} chars):")
            print(formatted_result[:500])  # Print first 500 chars

            return SystemMessage(content=formatted_result)

        except AuthRequiredError as e:
            # Return friendly message with connect link
            return SystemMessage(
                content=f"To complete this request, please connect your {e.service_info.get('name', e.service)}:\n\n"
                        f"👉 {e.connect_url}\n\n"
                        f"After connecting, try your request again."
            )
