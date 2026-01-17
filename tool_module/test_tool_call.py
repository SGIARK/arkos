# tests/test_tool_call.py
import os

import pytest

from tool_module.tool_call import MCPClient, MCPServerConfig, MCPToolManager


@pytest.mark.asyncio
async def test_mcp_client_initialization():
    """Test MCP client starts and initializes."""
    config = MCPServerConfig(
        name="test_server",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    )

    client = MCPClient(config)
    await client.start()

    assert client._initialized is True

    tools = await client.list_tools()
    assert len(tools) > 0

    await client.stop()


@pytest.mark.asyncio
async def test_tool_manager_multiple_servers():
    """Test tool manager with multiple servers."""
    config = {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        }
    }

    manager = MCPToolManager(config)
    await manager.initialize_servers()

    tools = await manager.list_all_tools()
    assert len(tools) > 0

    await manager.shutdown()


@pytest.mark.asyncio
async def test_tool_execution():
    """Test actual tool execution."""
    config = {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        }
    }

    manager = MCPToolManager(config)
    await manager.initialize_servers()

    # Test tool call (assuming filesystem server has list_directory)
    result = await manager.call_tool("list_directory", {"path": "/tmp"})
    assert result is not None

    await manager.shutdown()


@pytest.mark.asyncio
async def test_tool_execution_google_calendar():
    """Test actual tool execution."""
    env = os.environ.copy()
    env["GOOGLE_OAUTH_CREDENTIALS"] = "path-to-oauth.json"
    env["GOOGLE_CALENDAR_MCP_TOKEN_PATH"] = "path-to-google-generated-tokens.json"
    config = {
        "google-calendar": {"command": "npx", "args": ["@cocal/google-calendar-mcp"], "env": env}
    }

    manager = MCPToolManager(config)
    await manager.initialize_servers()

    # Test tool calls from google-calendar mcp(assuming list-events and create-event exists)
    result = await manager.call_tool(
        "list-events",
        {
            "calendarId": "primary",
            "timeMin": "2025-11-27T00:00:00",
            "timeMax": "2025-12-02T00:00:00",
        },
    )

    assert result is not None

    result = await manager.call_tool(
        "create-event",
        {
            "calendarId": "primary",
            "summary": "test event created using the google calendar mcp with ac ver.2",
            "start": "2025-12-14T17:00:00",
            "end": "2025-12-14T18:00:00",
            "timeZone": "America/New_York",
        },
    )

    assert result is not None

    await manager.shutdown()


@pytest.mark.asyncio
async def test_tool_execution_brave_search():
    """Test actual tool execution."""
    env = os.environ.copy()
    env["BRAVE_API_KEY"] = "BRAVE_API_KEY"
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
            "query": "talk about cats",
        },
    )

    assert result is not None

    await manager.shutdown()
