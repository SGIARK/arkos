import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from model_module.ArkModelNew import SystemMessage  # noqa: E402
from state_module.core.base_state import StateOutput  # noqa: E402
from state_module.core.state import State  # noqa: E402
from state_module.core.state_registry import register_state  # noqa: E402
from tool_module.tool_call import AuthRequiredError  # noqa: E402


@register_state
class StateTool(State):
    """Chat-path tool state. Uses the LLM to choose and execute a tool based on conversation context."""

    type = "tool"

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.is_terminal = False

    def check_transition_ready(self, context):
        return True

    async def choose_tool(self, context, agent):
        """Choose a tool based on context using the LLM."""
        prompt = "based on the above user request, choose the tool which best satisfies the users request"
        instructions = context + [SystemMessage(content=prompt)]

        tool_option_class = await agent.create_tool_option_class()
        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "tool_choice",
                "schema": tool_option_class.model_json_schema(),
            },
        }

        output = await agent.call_llm(instructions, json_schema)
        structured_output = json.loads(output.content)
        tool_name = structured_output["tool_name"]

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

        args_prompt = f"Fill in the arguments for the tool '{tool_name}' based on the user's request."
        args_context = context + [SystemMessage(content=args_prompt)]

        args_output = await agent.call_llm(args_context, tool_args_schema)
        tool_args = json.loads(args_output.content)

        return {"tool_name": tool_name, "tool_args": tool_args}

    async def execute_tool(self, tool_call, agent):
        """Execute a tool call."""
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

            return StateOutput(
                content=str(tool_result),
                completion_signal="complete",
                structured_data={"tool_result": tool_result},
            )

        except AuthRequiredError as e:
            service_label = e.service_info.get("name", e.service) if getattr(e, "service_info", None) else e.service
            link = e.setup_url or e.connect_url or ""
            if link:
                body = (
                    f"To do that I need access to **{service_label}**. "
                    f"Open this link to connect it via Smithery, then ask me again.\n\n"
                    f"[connect {service_label.lower()}]({link})"
                )
            else:
                body = (
                    f"To do that I need access to **{service_label}**, "
                    f"but Smithery didn't return a setup URL. "
                    f"Check the server's config (it may need an API key)."
                )
            return StateOutput(
                content=body,
                completion_signal="needs_input",
                structured_data={
                    "auth_required": True,
                    "service": e.service,
                    "setup_url": link or None,
                    "state": getattr(e, "state", "auth_required"),
                },
            )
