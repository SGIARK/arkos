# tool_module/tool.py

import requests
from typing import Dict, Any, Optional

class Tool:
    """
    Base class for all tools that communicate via MCP.
    Provides a standardized interface for constructing and sending requests.
    """

    def __init__(self, name: str, endpoint: str, method: str = "POST", timeout: int = 10):
        """
        :param name: Tool name (used in logs and tool resolution)
        :param endpoint: Full URL of the remote MCP tool endpoint
        :param method: HTTP method (typically POST)
        :param timeout: Timeout in seconds for the request
        """
        self.name = name
        self.endpoint = endpoint
        self.method = method.upper()
        self.timeout = timeout

    def call(self, tool_input: Dict[str, Any], session_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Sends a request to the MCP tool with structured input.

        :param tool_input: Dict matching the tool's expected schema
        :param session_state: Optional state/context to pass (e.g., user ID, prior inputs)
        :return: JSON-decoded response
        """
        payload = {
            "tool_name": self.name,
            "parameters": tool_input,
            "session_state": session_state or {}
        }

        try:
            if self.method == "POST":
                response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {self.method}")
        except requests.RequestException as e:
            return {
                "error": "tool_request_failed",
                "detail": str(e),
                "tool_name": self.name
            }

        try:
            return response.json()
        except Exception as e:
            return {
                "error": "invalid_response_format",
                "detail": str(e),
                "tool_name": self.name
            }
    
    
