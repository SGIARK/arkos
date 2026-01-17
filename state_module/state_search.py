import asyncio
import json
import os
import sys

from model_module.ArkModelNew import (
    ToolMessage,
)
from state_module.state import State
from state_module.state_registry import register_state
from tool_module.tool_call import MCPToolManager

# from .state import State
# from .state_registry import register_state
#
# from ..model_module.ArkModelNew import ArkModelLink, UserMessage, AIMessage, SystemMessage, ToolMessage
# from ..tool_module.tool_call import MCPClient, MCPToolManager, MCPServerConfig


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@register_state
class StateSearch(State):
    type = "search"

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.is_terminal = False  # Stop after this state

    def check_transition_ready(self, _context):
        # ALWAYS allow transition after user provides input
        return True

    async def brave_search(self, query):
        """Test actual tool execution."""
        env = os.environ.copy()
        # env["BRAVE_API_KEY"] = "BRAVE_API_KEY"
        config = {
            "brave-search-mcp-server": {
                "command": "npx",
                "args": ["-y", "@brave/brave-search-mcp-server", "--transport", "stdio"],
                "env": env,
            }
        }

        manager = MCPToolManager(config)
        await manager.initialize_servers()

        tools = await manager.list_all_tools()
        assert len(tools) > 0

        # Test tool call for brave search mcp(assuming brave_web_search exists)
        result = await manager.call_tool(
            "brave_web_search",
            {
                "query": query,
            },
        )

        assert result is not None

        return result

    def extract_top_k(self, response, k=2):
        assert not response["isError"]

        parsed = []
        for item in response["content"]:
            if item.get("type") != "text":
                continue

            try:
                parsed.append(json.loads(item["text"]))
            except json.JSONDecodeError:
                continue

        return parsed[:k]

    def parse_query(self, context, _agent):
        """
        Extracts the most recent user query from context.
        """

        # context = [long_term_mem] + short_term_mem
        # we care about short_term_mem only
        messages = context[1:] if len(context) > 1 else context

        # walk backwards to find latest user message
        for msg in reversed(messages):
            if hasattr(msg, "role") and msg.role == "user":
                return msg.content

        # fallback: last message content
        if messages:
            return messages[-1].content

        raise ValueError("No valid query found in context")

    async def run(self, context, agent):
        query = self.parse_query(context, agent)
        search_results = await self.brave_search(query)
        top_k_results = self.extract_top_k(search_results)

        return ToolMessage(
            content=f"This is your search result. It is GROUND TRUTH, ignore your previous knowledge \n\n {top_k_results} \n\n Summarize and return control back to the user now",
            tool_calls={"SearchTool": True},
        )


if __name__ == "__main__":
    obj = StateSearch(name="name", config={"empty": "dict"})
    print(obj.brave_search(query="Tell me about cats"))
    result = asyncio.run(obj.brave_search(query="Tell me about cats"))

    top_k = obj.extract_top_k(result)
    print(top_k)
