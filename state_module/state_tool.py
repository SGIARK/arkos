import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from model_module.ArkModelNew import SystemMessage
from tool_module.tool_call import AuthRequiredError, PER_USER_SERVICES

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

    async def _get_available_accounts(self, server_name: str, agent) -> list:
        """
        Dynamically discover available accounts for per-user services.
        Returns list of account IDs that can be used.
        """
        try:
            # Check if this is a per-user service
            if server_name not in PER_USER_SERVICES:
                return []

            # Check if user is authenticated
            if not agent.current_user_id:
                return []

            # Try to get user's client to see if they're authenticated
            client = await agent.tool_manager._get_user_client(agent.current_user_id, server_name)
            if not client:
                return []

            # For MCP services, "normal" is the standard account ID
            # In the future, this could call a manage-accounts tool to list actual accounts
            return ["normal"]

        except Exception as e:
            print(f"[StateTool] Failed to get accounts for {server_name}: {e}")
            return []

    async def _get_schema_defaults(self, input_schema: dict, agent) -> dict:
        """
        Generate default values based on JSON schema.
        Uses schema properties, types, and defaults to create minimal valid args.
        """
        if not input_schema or "properties" not in input_schema:
            return {}

        defaults = {}
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        for field_name, field_spec in properties.items():
            # Only include required fields in defaults
            if field_name not in required:
                continue

            field_type = field_spec.get("type", "string")

            # Generate type-appropriate defaults
            if field_name in ["account", "account_id"]:
                # For account fields, use discovered accounts
                defaults[field_name] = "normal"  # Standard MCP account
            elif "time" in field_name.lower() or "date" in field_name.lower():
                # For date/time fields, use current time
                from datetime import datetime
                defaults[field_name] = datetime.now().strftime("%Y-%m-%dT00:00:00")
            elif field_type == "string":
                defaults[field_name] = field_spec.get("default", "")
            elif field_type == "boolean":
                defaults[field_name] = field_spec.get("default", False)
            elif field_type == "integer" or field_type == "number":
                defaults[field_name] = field_spec.get("default", 0)

        return defaults if defaults else None

    async def choose_tool(self, context, agent):
        """
        Chooses tool to use based on the context and server
        """
        import time
        t0 = time.time()
        print(f"[StateTool] Phase 1: Choosing tool...")

        # Get all tools with descriptions
        all_tools = await agent.tool_manager.list_all_tools()
        print(f"[StateTool] list_all_tools took {time.time() - t0:.2f}s")
        tool_descriptions = []
        for server_name, tools in all_tools.items():
            for tool_name, tool_spec in tools.items():
                desc = tool_spec.get('description', 'No description')
                tool_descriptions.append(f"- {tool_name}: {desc}")

        tools_list = "\n".join(tool_descriptions)
        print(f"[StateTool] Built tool list with {len(tool_descriptions)} tools, {len(tools_list)} chars")

        prompt = f"""Based on the user request, choose the tool that best satisfies it.

Available tools:
{tools_list}

Choose the most appropriate tool."""

        instructions = context + [SystemMessage(content=prompt)]
        print(f"[StateTool] Total context: {len(context)} messages + prompt = {len(instructions)} messages")

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
            t1 = time.time()
            print(f"[StateTool] Calling LLM for tool choice with context of {len(instructions)} messages...")
            output = await agent.call_llm(instructions, json_schema)
            print(f"[StateTool] LLM call took {time.time() - t1:.2f}s")
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

        # Build schema-driven args prompt
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        args_guidance = f"""Fill in the arguments for the tool '{{tool_name}}' based on the user's request.

IMPORTANT FORMATTING GUIDELINES:
- For date/time fields: Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS)
- Current date is {current_date}
- For "today": Use {current_date} with appropriate time bounds (00:00:00 to 23:59:59)
- Follow the schema requirements carefully"""

        # Add per-user service guidance dynamically
        server_name = agent.tool_manager._tool_registry.get(tool_name)
        if server_name and server_name in PER_USER_SERVICES:
            # Get available accounts for this user
            available_accounts = await self._get_available_accounts(server_name, agent)
            if available_accounts:
                account_list = ", ".join(f'"{acc}"' for acc in available_accounts)
                args_guidance += f"""

ACCOUNT PARAMETER:
- Available accounts for this service: {account_list}
- Use one of these account IDs when required
- Do NOT make up account IDs"""
                print(f"[StateTool] Available accounts for {server_name}: {available_accounts}")

        args_prompt = args_guidance.format(tool_name=tool_name)
        args_context = context + [SystemMessage(content=args_prompt)]

        print(f"[StateTool] Context length for args: {len(args_context)} messages")

        t2 = time.time()
        print(f"[StateTool] Calling LLM for tool args...")
        args_output = await agent.call_llm(args_context, tool_args_schema)
        print(f"[StateTool] Args LLM call took {time.time() - t2:.2f}s")

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

            # Try to use schema defaults
            print(f"[StateTool] Attempting to use schema defaults")
            tool_args = await self._get_schema_defaults(tool_spec.get("inputSchema", {}), agent)
            if tool_args:
                print(f"[StateTool] Using schema-based fallback args: {tool_args}")
            else:
                raise ValueError(f"Failed to generate valid arguments for {tool_name}")

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
