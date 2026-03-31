"""Tests for tool_module/tool_call.py: AuthRequiredError, MCPServerConfig, MCPClient, MCPToolManager."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from tool_module.tool_call import (
    AuthRequiredError,
    MCPServerConfig,
    MCPClient,
    MCPToolManager,
    PER_USER_SERVICES,
)


# --- AuthRequiredError ---


class TestAuthRequiredError:
    def test_known_service(self):
        err = AuthRequiredError(service="google-calendar", user_id="user1")
        assert err.service == "google-calendar"
        assert err.user_id == "user1"
        assert "Google Calendar" in err.message
        assert "user1" in err.connect_url

    def test_unknown_service(self):
        err = AuthRequiredError(service="unknown-svc", user_id="u2")
        assert err.service == "unknown-svc"
        assert "unknown-svc" in err.message

    def test_custom_message(self):
        err = AuthRequiredError(
            service="google-calendar", user_id="u1", message="Custom error"
        )
        assert err.message == "Custom error"

    def test_to_dict(self):
        err = AuthRequiredError(service="google-calendar", user_id="user1")
        d = err.to_dict()
        assert d["error"] == "auth_required"
        assert d["service"] == "google-calendar"
        assert d["service_name"] == "Google Calendar"
        assert "user1" in d["connect_url"]
        assert isinstance(d["message"], str)

    def test_is_exception(self):
        err = AuthRequiredError(service="x", user_id="u")
        assert isinstance(err, Exception)


# --- MCPServerConfig ---


class TestMCPServerConfig:
    def test_defaults(self):
        cfg = MCPServerConfig(name="test-server")
        assert cfg.name == "test-server"
        assert cfg.transport == "stdio"
        assert cfg.command is None
        assert cfg.args is None
        assert cfg.env is None
        assert cfg.url is None
        assert cfg.auth is None

    def test_stdio_config(self):
        cfg = MCPServerConfig(
            name="brave",
            transport="stdio",
            command="npx",
            args=["-y", "@anthropic/brave-mcp"],
            env={"BRAVE_API_KEY": "key123"},
        )
        assert cfg.command == "npx"
        assert len(cfg.args) == 2
        assert cfg.env["BRAVE_API_KEY"] == "key123"

    def test_http_config(self):
        cfg = MCPServerConfig(
            name="remote",
            transport="http",
            url="https://api.example.com/mcp",
            auth={"type": "bearer", "token": "abc"},
        )
        assert cfg.transport == "http"
        assert cfg.url == "https://api.example.com/mcp"
        assert cfg.auth["type"] == "bearer"


# --- MCPClient ---


class TestMCPClient:
    def _make_client(self):
        config = MCPServerConfig(name="test")
        transport = MagicMock()
        transport.connect = AsyncMock()
        transport.send_request = AsyncMock()
        transport.send_notification = AsyncMock()
        transport.close = AsyncMock()
        return MCPClient(config, transport), transport

    def test_init(self):
        client, transport = self._make_client()
        assert client._initialized is False
        assert client.config.name == "test"

    @pytest.mark.asyncio
    async def test_start_success(self):
        client, transport = self._make_client()
        transport.send_request.return_value = {
            "result": {"protocolVersion": "2024-11-05"}
        }

        await client.start()
        assert client._initialized is True
        transport.connect.assert_called_once()
        transport.send_notification.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_failure_error_response(self):
        client, transport = self._make_client()
        transport.send_request.return_value = {"error": "bad protocol"}

        with pytest.raises(RuntimeError, match="MCP initialization failed"):
            await client.start()
        assert client._initialized is False

    @pytest.mark.asyncio
    async def test_start_failure_transport_error(self):
        client, transport = self._make_client()
        transport.connect.side_effect = Exception("connection refused")

        with pytest.raises(Exception):
            await client.start()
        assert client._initialized is False

    @pytest.mark.asyncio
    async def test_stop(self):
        client, transport = self._make_client()
        client._initialized = True
        await client.stop()
        assert client._initialized is False
        transport.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_tools_success(self):
        client, transport = self._make_client()
        client._initialized = True
        transport.send_request.return_value = {
            "result": {
                "tools": [
                    {"name": "search", "description": "Search the web"},
                    {"name": "calc", "description": "Calculator"},
                ]
            }
        }

        tools = await client.list_tools()
        assert len(tools) == 2
        assert tools[0]["name"] == "search"

    @pytest.mark.asyncio
    async def test_list_tools_not_initialized(self):
        client, _ = self._make_client()
        with pytest.raises(RuntimeError, match="not initialized"):
            await client.list_tools()

    @pytest.mark.asyncio
    async def test_list_tools_error_response(self):
        client, transport = self._make_client()
        client._initialized = True
        transport.send_request.return_value = {"error": "server error"}

        with pytest.raises(RuntimeError, match="tools/list failed"):
            await client.list_tools()

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        client, transport = self._make_client()
        client._initialized = True
        transport.send_request.return_value = {"result": {"output": "42"}}

        result = await client.call_tool("calc", {"expression": "6*7"})
        assert result == {"output": "42"}

    @pytest.mark.asyncio
    async def test_call_tool_not_initialized(self):
        client, _ = self._make_client()
        with pytest.raises(RuntimeError, match="not initialized"):
            await client.call_tool("calc", {})

    @pytest.mark.asyncio
    async def test_call_tool_error_response(self):
        client, transport = self._make_client()
        client._initialized = True
        transport.send_request.return_value = {"error": "tool not found"}

        with pytest.raises(RuntimeError, match="execution failed"):
            await client.call_tool("bad_tool", {})


# --- MCPToolManager ---


class TestMCPToolManager:
    def test_init(self):
        mgr = MCPToolManager(config={"server1": {}})
        assert mgr.config == {"server1": {}}
        assert mgr.clients == {}
        assert mgr.user_clients == {}
        assert mgr._tool_registry == {}

    def test_create_transport_stdio(self):
        mgr = MCPToolManager(config={})
        transport = mgr._create_transport(
            {"transport": "stdio", "command": "npx", "args": ["-y", "server"]}
        )
        from tool_module.transports import StdioTransport

        assert isinstance(transport, StdioTransport)

    def test_create_transport_http(self):
        mgr = MCPToolManager(config={})
        transport = mgr._create_transport(
            {"transport": "http", "url": "https://example.com/mcp"}
        )
        from tool_module.transports import HTTPTransport

        assert isinstance(transport, HTTPTransport)

    def test_create_transport_default_is_stdio(self):
        mgr = MCPToolManager(config={})
        transport = mgr._create_transport({"command": "npx", "args": []})
        from tool_module.transports import StdioTransport

        assert isinstance(transport, StdioTransport)

    def test_create_transport_unsupported_raises(self):
        mgr = MCPToolManager(config={})
        with pytest.raises(ValueError, match="Unsupported transport"):
            mgr._create_transport({"transport": "grpc"})

    @pytest.mark.asyncio
    async def test_list_all_tools(self):
        mgr = MCPToolManager(config={})

        mock_client = MagicMock()
        mock_client.list_tools = AsyncMock(
            return_value=[
                {"name": "search", "description": "Search"},
                {"name": "calc", "description": "Calculator"},
            ]
        )
        mgr.clients = {"brave": mock_client}

        result = await mgr.list_all_tools()
        assert "brave" in result
        assert "search" in result["brave"]
        assert result["brave"]["search"]["_server"] == "brave"
        assert result["brave"]["search"]["_id"] == "brave.search"

    @pytest.mark.asyncio
    async def test_list_all_tools_empty(self):
        mgr = MCPToolManager(config={})
        result = await mgr.list_all_tools()
        assert result == {}

    @pytest.mark.asyncio
    async def test_call_tool_routes_to_correct_server(self):
        mgr = MCPToolManager(config={})
        mock_client = MagicMock()
        mock_client.call_tool = AsyncMock(return_value={"result": "ok"})
        mgr.clients = {"brave": mock_client}
        mgr._tool_registry = {"search": "brave"}

        result = await mgr.call_tool("search", {"q": "test"})
        assert result == '{"result": "ok"}'
        mock_client.call_tool.assert_called_once_with("search", {"q": "test"})

    @pytest.mark.asyncio
    async def test_call_tool_unknown_raises(self):
        mgr = MCPToolManager(config={})
        mgr._tool_registry = {}
        with pytest.raises(ValueError, match="Unknown tool"):
            await mgr.call_tool("nonexistent", {})

    @pytest.mark.asyncio
    async def test_call_tool_server_not_connected(self):
        mgr = MCPToolManager(config={})
        mgr._tool_registry = {"search": "disconnected-server"}
        mgr.clients = {}
        with pytest.raises(RuntimeError, match="not connected"):
            await mgr.call_tool("search", {})

    def test_get_user_service_status_no_token_store(self):
        mgr = MCPToolManager(config={}, token_store=None)
        status = mgr.get_user_service_status("user1")
        for svc in PER_USER_SERVICES:
            assert status[svc]["connected"] is False

    def test_get_user_service_status_with_token(self):
        mock_store = MagicMock()
        mock_store.has_token.return_value = True
        mgr = MCPToolManager(config={}, token_store=mock_store)
        status = mgr.get_user_service_status("user1")
        assert status["google-calendar"]["connected"] is True

    def test_get_missing_services(self):
        mock_store = MagicMock()
        mock_store.has_token.return_value = False
        mgr = MCPToolManager(config={}, token_store=mock_store)
        missing = mgr.get_missing_services("user1")
        assert len(missing) == len(PER_USER_SERVICES)
        assert all(not s["connected"] for s in missing)

    def test_get_missing_services_none_when_connected(self):
        mock_store = MagicMock()
        mock_store.has_token.return_value = True
        mgr = MCPToolManager(config={}, token_store=mock_store)
        missing = mgr.get_missing_services("user1")
        assert len(missing) == 0

    @pytest.mark.asyncio
    async def test_shutdown(self):
        mgr = MCPToolManager(config={})

        mock_client1 = MagicMock()
        mock_client1.stop = AsyncMock()
        mock_client2 = MagicMock()
        mock_client2.stop = AsyncMock()

        mgr.clients = {"s1": mock_client1}
        mgr.user_clients = {"u1": {"s2": mock_client2}}

        await mgr.shutdown()
        mock_client1.stop.assert_called_once()
        mock_client2.stop.assert_called_once()
        assert mgr.clients == {}
        assert mgr.user_clients == {}
        assert mgr._tool_registry == {}
