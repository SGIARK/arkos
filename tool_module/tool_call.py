"""
Remote MCP (Model Context Protocol) Integration for ARKOS.

This module manages connections to external MCP servers, handles tool discovery,
and executes tool calls via JSON-RPC 2.0 over various transports (stdio, HTTP).
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .transports import MCPTransport, StdioTransport, HTTPTransport

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""

    name: str
    transport: str = "stdio"  # "stdio" or "http"

    # STDIO-specific
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None

    # HTTP-specific
    url: Optional[str] = None
    auth: Optional[Dict[str, Any]] = None


class MCPClient:
    """
    Manages a single MCP server connection.

    Handles JSON-RPC 2.0 communication and implements the MCP protocol
    for tool discovery and execution. Transport-agnostic - works with
    stdio, HTTP, or any transport implementing MCPTransport.

    Parameters
    ----------
    config : MCPServerConfig
        Configuration for the MCP server connection
    transport : MCPTransport
        Transport layer for communication (stdio, HTTP, etc.)

    Attributes
    ----------
    transport : MCPTransport
        The active transport connection
    _initialized : bool
        Whether the MCP handshake has completed
    """

    def __init__(self, config: MCPServerConfig, transport: MCPTransport):
        self.config = config
        self.transport = transport
        self._initialized = False

    async def start(self) -> None:
        """
        Connect to MCP server and perform initialization handshake.

        Raises
        ------
        RuntimeError
            If the server fails to start or initialize
        """
        logger.info(f"Starting MCP server: {self.config.name}")

        try:
            # Connect transport
            await self.transport.connect()

            # Initialize MCP connection
            init_response = await self.transport.send_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "arkos", "version": "1.0.0"},
                },
            )

            if "error" in init_response:
                raise RuntimeError(
                    f"MCP initialization failed: {init_response['error']}"
                )

            # Send initialized notification
            await self.transport.send_notification("notifications/initialized", {})

            self._initialized = True
            logger.info(f"MCP server '{self.config.name}' initialized successfully")

        except Exception as e:
            logger.error(f"Failed to start MCP server '{self.config.name}': {e}")
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop the MCP server connection gracefully."""
        logger.info(f"Stopping MCP server: {self.config.name}")
        try:
            await self.transport.close()
        finally:
            self._initialized = False

    async def list_tools(self) -> List[Dict[str, Any]]:
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

        response = await self.transport.send_request("tools/list", {})

        if "error" in response:
            raise RuntimeError(f"tools/list failed: {response['error']}")

        tools = response.get("result", {}).get("tools", [])
        logger.debug(f"Server '{self.config.name}' has {len(tools)} tools")
        return tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
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

        response = await self.transport.send_request(
            "tools/call", {"name": name, "arguments": arguments}
        )

        if "error" in response:
            error_msg = response["error"]
            logger.error(f"Tool call failed: {error_msg}")
            raise RuntimeError(f"Tool '{name}' execution failed: {error_msg}")

        result = response.get("result", {})
        logger.debug(f"Tool result: {result}")
        return result


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

    def __init__(self, config: Dict[str, Dict[str, Any]]):
        self.config = config
        self.clients: Dict[str, MCPClient] = {}
        self._tool_registry: Dict[str, str] = {}  # tool_name -> server_name

    def _create_transport(self, server_config: Dict[str, Any]) -> MCPTransport:
        """
        Create appropriate transport based on configuration.

        Parameters
        ----------
        server_config : Dict[str, Any]
            Server configuration dictionary

        Returns
        -------
        MCPTransport
            Configured transport instance

        Raises
        ------
        ValueError
            If transport type is unsupported
        """
        transport_type = server_config.get("transport", "stdio")

        if transport_type == "stdio":
            return StdioTransport(
                command=server_config["command"],
                args=server_config["args"],
                env=server_config.get("env"),
            )
        elif transport_type == "http":
            return HTTPTransport(
                url=server_config["url"],
                auth_config=server_config.get("auth")
            )
        else:
            raise ValueError(f"Unsupported transport type: {transport_type}")

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
                # Create config object
                config = MCPServerConfig(
                    name=server_name,
                    transport=server_config.get("transport", "stdio"),
                    command=server_config.get("command"),
                    args=server_config.get("args"),
                    env=server_config.get("env"),
                    url=server_config.get("url"),
                    auth=server_config.get("auth"),
                )

                # Create appropriate transport
                transport = self._create_transport(server_config)

                # Create client with transport
                client = MCPClient(config, transport)
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

    async def list_all_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all available tools from all servers.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            {
                server_name: {
                    tool_name: tool_spec_with_metadata
                }
            }
        """
        all_tools: Dict[str, Dict[str, Any]] = {}

        for server_name, client in self.clients.items():
            try:
                tools = await client.list_tools()
                server_tools: Dict[str, Any] = {}

                for tool in tools:
                    tool_name = tool.get("name")
                    if not tool_name:
                        continue

                    tool["_server"] = server_name
                    tool["_id"] = f"{server_name}.{tool_name}"

                    server_tools[tool_name] = tool

                all_tools[server_name] = server_tools

            except Exception as e:
                logger.error(f"Failed to list tools from '{server_name}': {e}")

        return all_tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
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
