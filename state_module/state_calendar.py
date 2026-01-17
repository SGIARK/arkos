import os
import sys

from model_module.ArkModelNew import (
    ToolMessage,
)
from state_module.state import State
from state_module.state_registry import register_state
from tool_module.tool_call import MCPToolManager

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@register_state
class StateCal(State):
    type = "calendar"

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.is_terminal = False

    def check_transition_ready(self, _context):
        return True

    async def calendar_retrieval(self):
        env = os.environ.copy()
        env["GOOGLE_OAUTH_CREDENTIALS"] = "path-to-oauth.json"
        env["GOOGLE_CALENDAR_MCP_TOKEN_PATH"] = "path-to-google-generated-tokens.json"
        config = {
            "google-calendar": {
                "command": "npx",
                "args": ["@cocal/google-calendar-mcp"],
                "env": env,
            }
        }

        manager = MCPToolManager(config)
        await manager.initialize_servers()

        result = await manager.call_tool(
            "list-events",
            {
                "calendarId": "primary",
                "timeMin": "2025-11-27T00:00:00",
                "timeMax": "2025-12-02T00:00:00",
            },
        )

        assert result is not None

        return result

    async def run(self, _context, _agent):
        # calendar_contents = await self.calendar_retrieval()
        calendar_contents = """
            calendar_placeholder = (
    "Calendar (placeholder)\n\n"
    "• Tomorrow, 10:00–11:00\n"
    "  Team sync\n"
    "  Location: TBD\n\n"
    "• Wednesday, 2:00–4:00\n"
    "  Deep work block\n"
    "  Location: —\n\n"
    "Calendar integration is not enabled yet. This is a placeholder result."
            )
            """
        return ToolMessage(
            content=f"Calendar Retreival Results \n\n {calendar_contents} \n\n Ensure you return control back to the user now",
            tool_calls={"CalendarTool": True},
        )
