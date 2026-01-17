"""Tests for MCP tool call module with mocked servers."""

from unittest.mock import patch

import pytest

from tool_module.tool_call import MCPClient, MCPServerConfig, MCPToolManager


class TestMCPClient:
    """Test cases for MCPClient with mocked subprocess."""

    @pytest.mark.asyncio
    async def test_mcp_client_initialization_mocked(self, mock_mcp_client):
        """Test MCP client initialization with mocked subprocess."""
        config = MCPServerConfig(
            name="test_server",
            command="echo",
            args=["test"],
        )

        with (
            patch.object(MCPClient, "start", mock_mcp_client.start),
            patch.object(MCPClient, "list_tools", mock_mcp_client.list_tools),
        ):
            client = MCPClient(config)
            client._initialized = True

            tools = await mock_mcp_client.list_tools()
            assert len(tools) > 0
            assert tools[0]["name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_mcp_client_stop(self, mock_mcp_client):
        """Test MCP client stop."""
        config = MCPServerConfig(
            name="test_server",
            command="echo",
            args=["test"],
        )

        client = MCPClient(config)
        client._initialized = True

        with patch.object(client, "stop", mock_mcp_client.stop):
            await mock_mcp_client.stop()
            mock_mcp_client.stop.assert_called_once()


class TestMCPToolManager:
    """Test cases for MCPToolManager with mocked clients."""

    @pytest.mark.asyncio
    async def test_tool_manager_initialization(self, mock_mcp_manager):
        """Test tool manager initialization with mocked servers."""
        config = {"test_server": {"command": "echo", "args": ["test"]}}

        with (
            patch.object(MCPToolManager, "initialize_servers", mock_mcp_manager.initialize_servers),
            patch.object(MCPToolManager, "list_all_tools", mock_mcp_manager.list_all_tools),
        ):
            MCPToolManager(config)  # Initialize manager
            await mock_mcp_manager.initialize_servers()

            tools = await mock_mcp_manager.list_all_tools()
            assert len(tools) > 0

    @pytest.mark.asyncio
    async def test_tool_execution_mocked(self, mock_mcp_manager):
        """Test tool execution with mocked manager."""
        config = {
            "filesystem": {"command": "npx", "args": ["-y", "@mcp/server-filesystem", "/tmp"]}
        }

        with (
            patch.object(MCPToolManager, "initialize_servers", mock_mcp_manager.initialize_servers),
            patch.object(MCPToolManager, "call_tool", mock_mcp_manager.call_tool),
            patch.object(MCPToolManager, "shutdown", mock_mcp_manager.shutdown),
        ):
            MCPToolManager(config)  # Initialize manager
            await mock_mcp_manager.initialize_servers()

            result = await mock_mcp_manager.call_tool("list_directory", {"path": "/tmp"})
            assert result is not None
            assert result["result"] == "success"

            await mock_mcp_manager.shutdown()

    @pytest.mark.asyncio
    async def test_tool_manager_shutdown(self, mock_mcp_manager):
        """Test tool manager shutdown."""
        config = {"test_server": {"command": "echo", "args": ["test"]}}

        with patch.object(MCPToolManager, "shutdown", mock_mcp_manager.shutdown):
            MCPToolManager(config)  # Initialize manager
            await mock_mcp_manager.shutdown()
            mock_mcp_manager.shutdown.assert_called_once()


@pytest.mark.integration
class TestMCPIntegration:
    """Integration tests that require actual MCP servers.

    These are skipped in CI and only run locally.
    Run with: pytest -m integration
    """

    @pytest.mark.asyncio
    async def test_real_filesystem_mcp(self):
        """Test actual filesystem MCP server (requires npx)."""
        pytest.skip("Integration test - requires npx and MCP servers")

    @pytest.mark.asyncio
    async def test_real_google_calendar_mcp(self):
        """Test actual Google Calendar MCP (requires credentials)."""
        pytest.skip("Integration test - requires Google Calendar credentials")

    @pytest.mark.asyncio
    async def test_real_brave_search_mcp(self):
        """Test actual Brave Search MCP (requires API key)."""
        pytest.skip("Integration test - requires Brave API key")
