"""
STDIO transport for MCP protocol.

Communicates with MCP servers via standard input/output (subprocess).
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, List, Optional
from threading import Lock

from .base import MCPTransport

logger = logging.getLogger(__name__)


class StdioTransport(MCPTransport):
    """
    STDIO-based MCP transport using subprocess communication.

    Launches the MCP server as a subprocess and communicates via
    stdin/stdout pipes using newline-delimited JSON-RPC.

    Parameters
    ----------
    command : str
        Command to execute (e.g., "npx", "docker")
    args : List[str]
        Command arguments
    env : Optional[Dict[str, str]]
        Environment variables for the subprocess
    """

    def __init__(
        self, command: str, args: List[str], env: Optional[Dict[str, str]] = None
    ):
        self.command = command
        self.args = args
        self.env = env or {}
        self.process: Optional[asyncio.subprocess.Process] = None
        self.request_id = 0
        self._lock = Lock()
        self._stderr_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Start the MCP server subprocess."""
        logger.info(f"Starting STDIO transport: {self.command} {' '.join(self.args)}")

        # Build environment - merge with current environment
        full_env = dict(os.environ)
        full_env.update(self.env)

        # Ensure system paths are always present so subprocesses (like sh) can be found
        system_paths = ["/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
        current_path = full_env.get("PATH", "")
        for p in system_paths:
            if p not in current_path:
                current_path = current_path + ":" + p
        full_env["PATH"] = current_path

        try:
            self.process = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=full_env,
            )
            logger.info("STDIO subprocess started successfully")

            # Start background stderr reader so we can see server output in real-time
            self._stderr_task = asyncio.create_task(self._read_stderr())

        except Exception as e:
            logger.error(f"Failed to start STDIO subprocess: {e}")
            raise RuntimeError(f"STDIO transport connection failed: {e}")

    async def send_request(
        self, method: str, params: Dict[str, Any], timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Send JSON-RPC request via stdin and read response from stdout."""
        if not self.process:
            raise RuntimeError("STDIO transport not connected")

        # Generate unique request ID
        with self._lock:
            self.request_id += 1
            req_id = self.request_id

        # Build JSON-RPC request
        request = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}

        logger.debug(f"STDIO >> {json.dumps(request)}")

        # Send request via stdin
        request_line = json.dumps(request) + "\n"
        self.process.stdin.write(request_line.encode())
        await self.process.stdin.drain()

        # Read response from stdout with timeout to prevent hanging
        try:
            response_line = await asyncio.wait_for(
                self.process.stdout.readline(), timeout=timeout
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"STDIO server timed out after {timeout}s waiting for response to '{method}'. "
                "Check server stderr logs above for details (e.g. OAuth callback URL)."
            )

        if not response_line:
            raise RuntimeError("STDIO server closed connection unexpectedly")

        response = json.loads(response_line.decode())
        logger.debug(f"STDIO << {json.dumps(response)}")

        return response

    async def send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """Send JSON-RPC notification via stdin (no response expected)."""
        if not self.process:
            raise RuntimeError("STDIO transport not connected")

        notification = {"jsonrpc": "2.0", "method": method, "params": params}

        logger.debug(f"STDIO >> {json.dumps(notification)}")

        notification_line = json.dumps(notification) + "\n"
        self.process.stdin.write(notification_line.encode())
        await self.process.stdin.drain()

    async def _read_stderr(self) -> None:
        """Continuously read and log stderr from the subprocess."""
        try:
            while self.process and self.process.stderr:
                line = await self.process.stderr.readline()
                if not line:
                    break
                logger.warning(f"[{self.command}] stderr: {line.decode().rstrip()}")
        except Exception:
            pass

    async def close(self) -> None:
        """Terminate the subprocess gracefully."""
        if self._stderr_task:
            self._stderr_task.cancel()
            self._stderr_task = None
        if self.process:
            logger.info("Stopping STDIO subprocess")
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Force killing STDIO subprocess")
                self.process.kill()
                await self.process.wait()
            finally:
                self.process = None
