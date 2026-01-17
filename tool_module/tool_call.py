"""
Remote MCP (Model Context Protocol) Integration for ARKOS.

This module manages connections to external MCP servers, handles tool discovery,
and executes tool calls via JSON-RPC 2.0 over stdio.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""

    name: str
    command: str
    args: list[str]
    env: dict[str, str] | None = None


class MCPClient:
    """
    Manages a single MCP server connection via subprocess.

    Handles JSON-RPC 2.0 communication over stdin/stdout and implements
    the MCP protocol for tool discovery and execution.

    Parameters
    ----------
    config : MCPServerConfig
        Configuration for the MCP server connection

    Attributes
    ----------
    process : Optional[asyncio.subprocess.Process]
        The running subprocess for the MCP server
    request_id : int
        Counter for JSON-RPC request IDs
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.process: asyncio.subprocess.Process | None = None
        self.request_id = 0
        self._lock = Lock()
        self._initialized = False

    async def start(self) -> None:
        """
        Start the MCP server subprocess and perform initialization handshake.

        Raises
        ------
        RuntimeError
            If the server fails to start or initialize
        """
        logger.info(f"Starting MCP server: {self.config.name}")

        # Build environment
        env = dict(self.config.env) if self.config.env else {}

        try:
            # Start subprocess
            self.process = await asyncio.create_subprocess_exec(
                self.config.command,
                *self.config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            # Initialize MCP connection
            init_response = await self._send_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "arkos", "version": "1.0.0"},
                },
            )

            if "error" in init_response:
                raise RuntimeError(f"MCP initialization failed: {init_response['error']}")

            # Send initialized notification
            await self._send_notification("notifications/initialized", {})

            self._initialized = True
            logger.info(f"MCP server '{self.config.name}' initialized successfully")

        except Exception as e:
            logger.error(f"Failed to start MCP server '{self.config.name}': {e}")
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop the MCP server subprocess gracefully."""
        if self.process:
            logger.info(f"Stopping MCP server: {self.config.name}")
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"Force killing MCP server: {self.config.name}")
                self.process.kill()
                await self.process.wait()
            finally:
                self.process = None
                self._initialized = False

    async def list_tools(self) -> list[dict[str, Any]]:
        """
        Request list of available tools from the MCP server.

        Returns
        -------
        List[Dict[str, Any]]
            List of tool definitions with name, description, and input schema

        Raises
        ------
        RuntimeError
            If server is not initialized or request fails
        """
        if not self._initialized:
            raise RuntimeError(f"MCP server '{self.config.name}' not initialized")

        response = await self._send_request("tools/list", {})

        if "error" in response:
            raise RuntimeError(f"tools/list failed: {response['error']}")

        tools = response.get("result", {}).get("tools", [])
        logger.debug(f"Server '{self.config.name}' has {len(tools)} tools")
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """
        Execute a tool on the MCP server.

        Parameters
        ----------
        name : str
            Name of the tool to execute
        arguments : Dict[str, Any]
            Arguments to pass to the tool

        Returns
        -------
        Any
            Tool execution result

        Raises
        ------
        RuntimeError
            If server is not initialized or tool execution fails
        """
        if not self._initialized:
            raise RuntimeError(f"MCP server '{self.config.name}' not initialized")

        logger.info(f"Calling tool '{name}' on server '{self.config.name}'")
        logger.debug(f"Arguments: {arguments}")

        response = await self._send_request("tools/call", {"name": name, "arguments": arguments})

        if "error" in response:
            error_msg = response["error"]
            logger.error(f"Tool call failed: {error_msg}")
            raise RuntimeError(f"Tool '{name}' execution failed: {error_msg}")

        result = response.get("result", {})
        logger.debug(f"Tool result: {result}")
        return result

    async def _send_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """
        Send a JSON-RPC 2.0 request and wait for response.

        Parameters
        ----------
        method : str
            JSON-RPC method name
        params : Dict[str, Any]
            Method parameters

        Returns
        -------
        Dict[str, Any]
            JSON-RPC response
        """
        with self._lock:
            self.request_id += 1
            req_id = self.request_id

        request = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}

        logger.debug(f"[{self.config.name}] >> {json.dumps(request)}")

        # Send request
        request_line = json.dumps(request) + "\n"
        assert self.process is not None and self.process.stdin is not None
        self.process.stdin.write(request_line.encode())
        await self.process.stdin.drain()

        # Read response
        assert self.process.stdout is not None
        response_line = await self.process.stdout.readline()
        if not response_line:
            raise RuntimeError(f"MCP server '{self.config.name}' closed connection")

        response = json.loads(response_line.decode())
        logger.debug(f"[{self.config.name}] << {json.dumps(response)}")

        return response

    async def _send_notification(self, method: str, params: dict[str, Any]) -> None:
        """Send a JSON-RPC 2.0 notification (no response expected)."""
        notification = {"jsonrpc": "2.0", "method": method, "params": params}

        logger.debug(f"[{self.config.name}] >> {json.dumps(notification)}")

        notification_line = json.dumps(notification) + "\n"
        assert self.process is not None and self.process.stdin is not None
        self.process.stdin.write(notification_line.encode())
        await self.process.stdin.drain()


class MCPToolManager:
    """
    Manages multiple MCP server connections and provides unified tool interface.

    Coordinates tool discovery across all servers and routes tool execution
    to the appropriate server.

    Parameters
    ----------
    config : Dict[str, Dict[str, Any]]
        MCP servers configuration from config file

    Attributes
    ----------
    clients : Dict[str, MCPClient]
        Active MCP client connections by server name
    """

    def __init__(self, config: dict[str, dict[str, Any]]):
        self.config = config
        self.clients: dict[str, MCPClient] = {}
        self._tool_registry: dict[str, str] = {}  # tool_name -> server_name

    async def initialize_servers(self) -> None:
        """
        Initialize all configured MCP server connections.

        Starts each server, performs handshake, and builds tool registry.

        Raises
        ------
        RuntimeError
            If any server fails to initialize
        """
        logger.info(f"Initializing {len(self.config)} MCP servers")

        for server_name, server_config in self.config.items():
            try:
                config = MCPServerConfig(
                    name=server_name,
                    command=server_config["command"],
                    args=server_config["args"],
                    env=server_config.get("env"),
                )

                client = MCPClient(config)
                await client.start()

                # Discover tools
                tools = await client.list_tools()
                for tool in tools:
                    tool_name = tool["name"]
                    self._tool_registry[tool_name] = server_name
                    logger.info(f"Registered tool '{tool_name}' from '{server_name}'")

                self.clients[server_name] = client

            except Exception as e:
                logger.error(f"Failed to initialize server '{server_name}': {e}")
                # Continue with other servers

        if not self.clients:
            raise RuntimeError("No MCP servers successfully initialized")

        logger.info(
            f"Initialized {len(self.clients)} servers with {len(self._tool_registry)} total tools"
        )

    async def list_all_tools(self) -> list[dict[str, Any]]:
        """
        Get all available tools from all servers.

        Returns
        -------
        List[Dict[str, Any]]
            Combined list of all tools with server name added
        """
        all_tools = []

        for server_name, client in self.clients.items():
            try:
                tools = await client.list_tools()
                for tool in tools:
                    tool["_server"] = server_name  # Add server metadata
                    all_tools.append(tool)
            except Exception as e:
                logger.error(f"Failed to list tools from '{server_name}': {e}")

        return all_tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        Execute a tool by name, routing to the correct server.

        Parameters
        ----------
        tool_name : str
            Name of the tool to execute
        arguments : Dict[str, Any]
            Tool arguments

        Returns
        -------
        Any
            Tool execution result

        Raises
        ------
        ValueError
            If tool is not found in registry
        RuntimeError
            If tool execution fails
        """
        server_name = self._tool_registry.get(tool_name)
        if not server_name:
            raise ValueError(f"Unknown tool: {tool_name}")

        client = self.clients.get(server_name)
        if not client:
            raise RuntimeError(f"Server '{server_name}' not connected")

        return await client.call_tool(tool_name, arguments)

    async def shutdown(self) -> None:
        """Gracefully shutdown all MCP server connections."""
        logger.info("Shutting down all MCP servers")

        for client in self.clients.values():
            try:
                await client.stop()
            except Exception as e:
                logger.error(f"Error stopping server: {e}")

        self.clients.clear()
        self._tool_registry.clear()
