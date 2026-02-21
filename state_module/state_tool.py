import os
import sys
import json
import logging
from datetime import datetime, timezone, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model_module.ArkModelNew import SystemMessage, UserMessage
from tool_module.tool_call import AuthRequiredError, PER_USER_SERVICES

from state_module.state import State
from state_module.state_registry import register_state

logger = logging.getLogger(__name__)


def _get_eastern_date() -> str:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now(timezone(timedelta(hours=-5))).strftime("%Y-%m-%d")


@register_state
class StateTool(State):
    type = "tool"

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.is_terminal = False

    def check_transition_ready(self, context):
        return True

    async def _get_available_accounts(self, server_name: str, agent) -> list:
        """Return list of authenticated account IDs for a per-user service."""
        try:
            if server_name not in PER_USER_SERVICES:
                return []
            if not agent.current_user_id:
                return []
            client = await agent.tool_manager._get_user_client(agent.current_user_id, server_name)
            if not client:
                return []
            return ["normal"]
        except Exception as e:
            logger.error("Failed to get accounts for %s: %s", server_name, e)
            return []

    async def _get_schema_defaults(self, input_schema: dict, agent) -> dict:
        """Generate minimal valid args from JSON schema required fields."""
        if not input_schema or "properties" not in input_schema:
            return {}

        defaults = {}
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        for field_name, field_spec in properties.items():
            if field_name not in required:
                continue

            field_type = field_spec.get("type", "string")

            if field_name in ["account", "account_id"]:
                defaults[field_name] = "normal"
            elif field_name in ["calendarId", "calendar_id"]:
                defaults[field_name] = "primary"
            elif "time" in field_name.lower() or "date" in field_name.lower():
                defaults[field_name] = datetime.now().strftime("%Y-%m-%dT00:00:00")
            elif field_type == "string":
                defaults[field_name] = field_spec.get("default", "")
            elif field_type == "boolean":
                defaults[field_name] = field_spec.get("default", False)
            elif field_type in ("integer", "number"):
                defaults[field_name] = field_spec.get("default", 0)

        return defaults if defaults else None

    async def choose_tool(self, context, agent):
        """Select the best tool and generate its arguments."""
        all_tools = await agent.tool_manager.list_all_tools()
        tool_descriptions = []
        for server_name, tools in all_tools.items():
            for tool_name, tool_spec in tools.items():
                desc = tool_spec.get("description", "No description")
                if len(desc) > 80:
                    desc = desc[:80] + "..."
                tool_descriptions.append(f"- {tool_name}: {desc}")

        prompt = f"""Based on the user request, choose the tool that best satisfies it.

Available tools:
{chr(10).join(tool_descriptions)}

Choose the most appropriate tool."""

        instructions = context + [SystemMessage(content=prompt)]

        tool_option_class = await agent.create_tool_option_class()
        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "tool_choice",
                "schema": tool_option_class.model_json_schema(),
            },
        }

        try:
            output = await agent.call_llm(instructions, json_schema)
            structured_output = json.loads(output.content)
            tool_name = structured_output["tool_name"]
            logger.info("Selected tool: %s", tool_name)
        except json.JSONDecodeError as e:
            logger.error("Tool selection JSON error: %s\nOutput: %s", e, output.content if output else "None")
            raise

        server_name = agent.tool_manager._tool_registry[tool_name]
        all_tools = await agent.tool_manager.list_all_tools()
        tool_spec = all_tools[server_name][tool_name]

        tool_args_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "tool_args",
                "schema": tool_spec.get("inputSchema", {}),
            },
        }

        current_date = _get_eastern_date()
        args_guidance = f"""Fill in the arguments for the tool '{{tool_name}}' based on the user's request.

IMPORTANT FORMATTING GUIDELINES:
- For date/time fields: Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS)
- Current date is {current_date}
- For "today": Use {current_date} with appropriate time bounds (00:00:00 to 23:59:59)
- Follow the schema requirements carefully"""

        server_name = agent.tool_manager._tool_registry.get(tool_name)
        if server_name and server_name in PER_USER_SERVICES:
            available_accounts = await self._get_available_accounts(server_name, agent)
            if available_accounts:
                account_list = ", ".join(f'"{acc}"' for acc in available_accounts)
                args_guidance += f"""

ACCOUNT PARAMETER:
- Available accounts for this service: {account_list}
- Use one of these account IDs when required
- Do NOT make up account IDs"""

        args_prompt = args_guidance.format(tool_name=tool_name)
        recent_user_msgs = [m for m in context if isinstance(m, UserMessage)]
        json_only = SystemMessage(content="Output ONLY a valid JSON object. No explanation, no prose, no markdown. Just the JSON.")
        args_context = (
            recent_user_msgs[-1:] + [json_only, SystemMessage(content=args_prompt)]
            if recent_user_msgs
            else context[-1:] + [json_only, SystemMessage(content=args_prompt)]
        )

        args_output = await agent.call_llm(args_context, tool_args_schema)

        if not args_output or not args_output.content:
            logger.warning("LLM returned empty tool args, retrying")
            simple_context = [context[-1], SystemMessage(content=args_prompt)]
            args_output = await agent.call_llm(simple_context, tool_args_schema)
            if not args_output or not args_output.content:
                raise ValueError("Failed to generate tool arguments after retry")

        try:
            tool_args = json.loads(args_output.content)
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON for tool args (%s), falling back to schema defaults", e)
            tool_args = await self._get_schema_defaults(tool_spec.get("inputSchema", {}), agent)
            if not tool_args:
                raise ValueError(f"Failed to generate valid arguments for {tool_name}")

        logger.info("Calling '%s' with args: %s", tool_name, json.dumps(tool_args))
        return {"tool_name": tool_name, "tool_args": tool_args}

    async def execute_tool(self, tool_call, agent):
        """Execute the chosen tool call."""
        tool_result = await agent.tool_manager.call_tool(
            tool_name=tool_call["tool_name"],
            arguments=tool_call["tool_args"],
            user_id=agent.current_user_id,
        )
        return tool_result

    async def run(self, context, agent=None):
        try:
            tool_arg_dict = await self.choose_tool(context=context, agent=agent)
            tool_result = await self.execute_tool(tool_call=tool_arg_dict, agent=agent)

            tool_name = tool_arg_dict.get("tool_name", "unknown")

            # Unwrap MCP content format: {"content": [{"type": "text", "text": "..."}]}
            if isinstance(tool_result, dict) and "content" in tool_result:
                content_items = tool_result.get("content", [])
                if content_items and isinstance(content_items, list):
                    first_item = content_items[0]
                    if isinstance(first_item, dict) and "text" in first_item:
                        actual_data = first_item["text"]
                        try:
                            parsed = json.loads(actual_data)
                            formatted_result = f"TOOL_RESULT from '{tool_name}':\n{json.dumps(parsed, indent=2)}"
                        except Exception:
                            formatted_result = f"TOOL_RESULT from '{tool_name}':\n{actual_data}"
                    else:
                        formatted_result = f"TOOL_RESULT from '{tool_name}':\n{json.dumps(tool_result, indent=2)}"
                else:
                    formatted_result = f"TOOL_RESULT from '{tool_name}':\n{json.dumps(tool_result, indent=2)}"
            else:
                formatted_result = f"TOOL_RESULT from '{tool_name}':\n{json.dumps(tool_result, indent=2)}"

            return SystemMessage(content=formatted_result)

        except AuthRequiredError as e:
            return SystemMessage(
                content=f"To complete this request, please connect your {e.service_info.get('name', e.service)}:\n\n"
                        f"👉 {e.connect_url}\n\n"
                        f"After connecting, try your request again."
            )
