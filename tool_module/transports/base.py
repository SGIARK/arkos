"""
Base transport interface for MCP protocol.

All MCP transports (stdio, HTTP) implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class MCPTransport(ABC):
    """Abstract base class for MCP transport mechanisms."""

    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to the MCP server.
        """
        pass

    @abstractmethod
    async def send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a JSON-RPC request and wait for response.
        """
        pass

    @abstractmethod
    async def send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """
        Send a JSON-RPC notification (no response expected).
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Close the transport connection gracefully.
        """
        pass
