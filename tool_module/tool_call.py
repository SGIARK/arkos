"""
Remote MCP (Model Context Protocol) Integration for ARKOS.

This module manages connections to external MCP servers, handles tool discovery,
and executes tool calls via JSON-RPC 2.0 over various transports (stdio, HTTP).
"""

import os
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

from .transports import MCPTransport, StdioTransport, HTTPTransport

logger = logging.getLogger(__name__)

# Services that require per-user authentication
PER_USER_SERVICES = {
    "google-calendar": {
        "name": "Google Calendar",
        "auth_path": "/auth/google/login",
        "scopes": ["calendar"],
    }
}


class AuthRequiredError(Exception):
    """Raised when a tool requires user authentication."""

    def __init__(self, service: str, user_id: str, message: str = None):
        self.service = service
        self.user_id = user_id
        self.service_info = PER_USER_SERVICES.get(service, {})
        self.connect_url = f"{self.service_info.get('auth_path', '/auth/connect')}?user_id={user_id}"
        self.message = message or f"Please connect {self.service_info.get('name', service)} to continue"
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Return structured error for API responses."""
        return {
            "error": "auth_required",
            "service": self.service,
            "service_name": self.service_info.get("name", self.service),
            "connect_url": self.connect_url,
            "message": self.message,
        }


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
    to the appropriate server. Supports per-user authentication for services
    like Google Calendar.

    Parameters
    ----------
    config : Dict[str, Dict[str, Any]]
        MCP servers configuration from config file
    token_store : Optional[UserTokenStore]
        Token store for per-user authentication

    Attributes
    ----------
    clients : Dict[str, MCPClient]
        Active MCP client connections by server name (shared/agent-level)
    user_clients : Dict[str, Dict[str, MCPClient]]
        Per-user MCP clients: {user_id: {server_name: MCPClient}}
    """

    def __init__(self, config: Dict[str, Dict[str, Any]], token_store=None):
        self.config = config
        self.token_store = token_store
        self.clients: Dict[str, MCPClient] = {}
        self.user_clients: Dict[str, Dict[str, MCPClient]] = {}
        self._tool_registry: Dict[str, str] = {}  # tool_name -> server_name
        self._user_token_dir = Path.home() / ".arkos" / "user_tokens"

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
        Per-user services (like google-calendar) are skipped during init
        but their tools are still registered for later per-user instantiation.

        Raises
        ------
        RuntimeError
            If any server fails to initialize
        """
        logger.info(f"Initializing {len(self.config)} MCP servers")

        for server_name, server_config in self.config.items():
            # Skip per-user services during agent-level init
            if server_name in PER_USER_SERVICES:
                logger.info(f"Skipping per-user service '{server_name}' (will init per-user)")
                # Register placeholder so we know this service exists
                self._per_user_configs = getattr(self, '_per_user_configs', {})
                self._per_user_configs[server_name] = server_config
                continue

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

        if not self.clients and not getattr(self, '_per_user_configs', {}):
            raise RuntimeError("No MCP servers successfully initialized")

        logger.info(
            f"Initialized {len(self.clients)} shared servers, "
            f"{len(getattr(self, '_per_user_configs', {}))} per-user services"
        )

    async def list_all_tools(self, include_per_user: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Get all available tools from all servers.

        Parameters
        ----------
        include_per_user : bool
            If True, also discover tools from per-user services

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

        # Discover tools from per-user services
        if include_per_user:
            per_user_configs = getattr(self, '_per_user_configs', {})
            for server_name, server_config in per_user_configs.items():
                if server_name in all_tools:
                    continue
                try:
                    tools = await self._discover_per_user_tools(server_name, server_config)
                    if tools:
                        all_tools[server_name] = tools
                except Exception as e:
                    logger.error(f"Failed to discover per-user tools from '{server_name}': {e}")

        return all_tools

    async def _discover_per_user_tools(self, server_name: str, server_config: dict) -> Dict[str, Any]:
        """Temporarily start a per-user service to discover its tools."""
        transport = StdioTransport(
            command=server_config["command"],
            args=server_config["args"],
            env=server_config.get("env"),
        )
        config = MCPServerConfig(
            name=f"{server_name}:discovery",
            transport="stdio",
            command=server_config.get("command"),
            args=server_config.get("args"),
        )
        client = MCPClient(config, transport)
        try:
            await client.start()
            tools = await client.list_tools()
            server_tools: Dict[str, Any] = {}
            for tool in tools:
                tool_name = tool.get("name")
                if not tool_name:
                    continue
                tool["_server"] = server_name
                tool["_id"] = f"{server_name}.{tool_name}"
                tool["_requires_auth"] = True
                self._tool_registry[tool_name] = server_name
                server_tools[tool_name] = tool
            return server_tools
        finally:
            await client.stop()

    async def _get_user_client(self, user_id: str, server_name: str) -> Optional[MCPClient]:
        """
        Get or create a per-user MCP client for a service.

        Parameters
        ----------
        user_id : str
            The user ID
        server_name : str
            Name of the MCP server (e.g., 'google-calendar')

        Returns
        -------
        Optional[MCPClient]
            User-specific client, or None if user has no token
        """
        # Check if we already have a client for this user
        if user_id in self.user_clients and server_name in self.user_clients[user_id]:
            return self.user_clients[user_id][server_name]

        # Check if user has a token
        if not self.token_store or not self.token_store.has_token(user_id, server_name):
            return None

        # Write user's token to a file
        self._user_token_dir.mkdir(parents=True, exist_ok=True)
        token_file = self._user_token_dir / f"{user_id}_{server_name}.json"
        self.token_store.write_token_file(user_id, server_name, str(token_file))

        # Create a new MCP client with user's token
        server_config = self.config.get(server_name, {})
        if not server_config:
            return None

        # Build env: start with system env, then config env, then user token
        env = os.environ.copy()
        if server_config.get("env"):
            env.update(server_config.get("env"))
        # Point to user's token file
        env["GOOGLE_CALENDAR_MCP_TOKEN_PATH"] = str(token_file)

        config = MCPServerConfig(
            name=f"{server_name}:{user_id}",
            transport=server_config.get("transport", "stdio"),
            command=server_config.get("command"),
            args=server_config.get("args"),
            env=env,
        )

        transport = StdioTransport(
            command=server_config["command"],
            args=server_config["args"],
            env=env,
        )

        client = MCPClient(config, transport)
        try:
            await client.start()

            # Discover and register tools from this per-user service
            tools = await client.list_tools()
            for tool in tools:
                tool_name = tool["name"]
                self._tool_registry[tool_name] = server_name
                logger.info(f"Registered per-user tool '{tool_name}' from '{server_name}'")

            # Cache the client
            if user_id not in self.user_clients:
                self.user_clients[user_id] = {}
            self.user_clients[user_id][server_name] = client
            return client
        except Exception as e:
            logger.error(f"Failed to start per-user client for {user_id}/{server_name}: {e}")
            return None

    async def call_tool(
        self, tool_name: str, arguments: Dict[str, Any], user_id: Optional[str] = None
    ) -> Any:
        """
        Execute a tool by name, routing to the correct server.

        Parameters
        ----------
        tool_name : str
            Name of the tool to execute
        arguments : Dict[str, Any]
            Tool arguments
        user_id : Optional[str]
            User ID for per-user authenticated services

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

        # If tool not in registry, check if it might be from a per-user service
        if not server_name:
            per_user_configs = getattr(self, '_per_user_configs', {})
            if per_user_configs:
                # Check if user needs to auth first
                for service_name in per_user_configs:
                    if not self.token_store or not self.token_store.has_token(user_id or "", service_name):
                        # User hasn't connected this service - raise auth error
                        raise AuthRequiredError(
                            service=service_name,
                            user_id=user_id or "unknown",
                        )
                    # User has token, try to connect and discover tools
                    client = await self._get_user_client(user_id, service_name)
                    if client and tool_name in self._tool_registry:
                        server_name = self._tool_registry[tool_name]
                        break

        if not server_name:
            raise ValueError(f"Unknown tool: {tool_name}")

        # Check if this is a per-user service
        if server_name in PER_USER_SERVICES:
            if not user_id:
                raise AuthRequiredError(
                    service=server_name,
                    user_id="unknown",
                    message=f"Tool '{tool_name}' requires user authentication"
                )
            client = await self._get_user_client(user_id, server_name)
            if client:
                return await client.call_tool(tool_name, arguments)
            else:
                raise AuthRequiredError(
                    service=server_name,
                    user_id=user_id,
                )

        # Fall back to shared client
        client = self.clients.get(server_name)
        if not client:
            raise RuntimeError(f"Server '{server_name}' not connected")

        return await client.call_tool(tool_name, arguments)

    def get_user_service_status(self, user_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Check which per-user services a user has connected.

        Returns
        -------
        Dict[str, Dict]
            {service_name: {connected: bool, connect_url: str, name: str}}
        """
        status = {}
        for service, info in PER_USER_SERVICES.items():
            connected = bool(
                self.token_store and self.token_store.has_token(user_id, service)
            )
            status[service] = {
                "connected": connected,
                "name": info.get("name", service),
                "connect_url": f"{info.get('auth_path', '/auth/connect')}?user_id={user_id}",
            }
        return status

    def get_missing_services(self, user_id: str) -> List[Dict[str, Any]]:
        """Get list of services user hasn't connected yet."""
        status = self.get_user_service_status(user_id)
        return [
            {"service": svc, **info}
            for svc, info in status.items()
            if not info["connected"]
        ]

    async def shutdown(self) -> None:
        """Gracefully shutdown all MCP server connections."""
        logger.info("Shutting down all MCP servers")

        # Shutdown shared clients
        for client in self.clients.values():
            try:
                await client.stop()
            except Exception as e:
                logger.error(f"Error stopping server: {e}")

        # Shutdown per-user clients
        for user_id, user_clients in self.user_clients.items():
            for server_name, client in user_clients.items():
                try:
                    await client.stop()
                except Exception as e:
                    logger.error(f"Error stopping user client {user_id}/{server_name}: {e}")

        self.clients.clear()
        self.user_clients.clear()
        self._tool_registry.clear()
