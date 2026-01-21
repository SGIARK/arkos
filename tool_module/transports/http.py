"""
HTTP transport for MCP protocol with OAuth 2.1 support.

Implements Streamable HTTP transport as defined in MCP specification.
Supports OAuth 2.1 with PKCE for authentication.
"""

import asyncio
import aiohttp
import json
import logging
import hashlib
import secrets
import base64
import webbrowser
from typing import Dict, Any, Optional
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlencode, urlparse, parse_qs

from .base import MCPTransport

logger = logging.getLogger(__name__)


class HTTPTransport(MCPTransport):
    """
    HTTP-based MCP transport using Streamable HTTP protocol.

    Supports OAuth 2.1 authentication with PKCE for secure access
    to remote MCP servers.

    Parameters
    ----------
    url : str
        MCP server endpoint URL (e.g., "https://api.example.com/mcp")
    auth_config : Optional[Dict[str, Any]]
        OAuth configuration containing:
        - type: "oauth" or "bearer"
        - client_id: OAuth client ID
        - scopes: List of scopes to request
        - token: Bearer token (if type is "bearer")
    """

    def __init__(self, url: str, auth_config: Optional[Dict[str, Any]] = None):
        self.url = url
        self.auth_config = auth_config or {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.oauth_manager: Optional[OAuthManager] = None

        # If OAuth is configured, create OAuth manager
        if auth_config and auth_config.get("type") == "oauth":
            self.oauth_manager = OAuthManager(
                server_url=url,
                client_id=auth_config.get("client_id", "arkos_mcp_client"),
                scopes=auth_config.get("scopes", ["mcp:tools", "mcp:resources"]),
            )
        # If bearer token provided directly
        elif auth_config and auth_config.get("type") == "bearer":
            self.access_token = auth_config.get("token")

    async def connect(self) -> None:
        """Establish HTTP session and authenticate if needed."""
        logger.info(f"Connecting to HTTP MCP server: {self.url}")

        # Create HTTP session
        self.session = aiohttp.ClientSession()

        # If OAuth is configured, get token
        if self.oauth_manager:
            try:
                self.access_token = await self.oauth_manager.get_token()
                logger.info("OAuth authentication successful")
            except Exception as e:
                logger.error(f"OAuth authentication failed: {e}")
                raise RuntimeError(f"HTTP transport authentication failed: {e}")

        logger.info("HTTP transport connected")

    async def send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send JSON-RPC request via HTTP POST."""
        if not self.session:
            raise RuntimeError("HTTP transport not connected")

        # Build JSON-RPC request
        request = {
            "jsonrpc": "2.0",
            "id": 1,  # Simple ID for now
            "method": method,
            "params": params,
        }

        # Build headers
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"

        logger.debug(f"HTTP >> POST {self.url}")
        logger.debug(f"HTTP >> {json.dumps(request)}")

        try:
            async with self.session.post(
                self.url, json=request, headers=headers
            ) as response:
                # Handle 401 - need authentication
                if response.status == 401:
                    if self.oauth_manager:
                        logger.info("Token expired, re-authenticating...")
                        self.access_token = await self.oauth_manager.authenticate()
                        # Retry with new token
                        return await self.send_request(method, params)
                    else:
                        raise RuntimeError(
                            "Authentication required but no OAuth configured"
                        )

                # Handle other errors
                if response.status >= 400:
                    error_text = await response.text()
                    raise RuntimeError(f"HTTP error {response.status}: {error_text}")

                # Parse response
                result = await response.json()
                logger.debug(f"HTTP << {json.dumps(result)}")

                return result

        except aiohttp.ClientError as e:
            logger.error(f"HTTP request failed: {e}")
            raise RuntimeError(f"HTTP transport error: {e}")

    async def send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """Send JSON-RPC notification via HTTP POST (no response expected)."""
        if not self.session:
            raise RuntimeError("HTTP transport not connected")

        notification = {"jsonrpc": "2.0", "method": method, "params": params}

        headers = {"Content-Type": "application/json"}

        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"

        logger.debug(f"HTTP >> POST {self.url} (notification)")

        try:
            async with self.session.post(
                self.url, json=notification, headers=headers
            ) as response:
                # Don't wait for response for notifications
                pass
        except aiohttp.ClientError as e:
            logger.warning(f"Notification failed: {e}")

    async def close(self) -> None:
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("HTTP transport closed")


class OAuthManager:
    """
    Manages OAuth 2.1 authentication flow with PKCE.

    Implements the MCP OAuth specification for secure authentication
    to remote MCP servers.

    Parameters
    ----------
    server_url : str
        MCP server URL
    client_id : str
        OAuth client ID
    scopes : List[str]
        OAuth scopes to request
    """

    def __init__(self, server_url: str, client_id: str, scopes: list):
        self.server_url = server_url
        self.client_id = client_id
        self.scopes = scopes
        self.token_cache_file = Path.home() / ".arkos" / "mcp_tokens.json"
        self.redirect_uri = "http://localhost:8765/callback"

    async def get_token(self) -> str:
        """Get access token (from cache or new authentication)."""
        # Try to load cached token
        cached = self._load_cached_token()
        if cached and not self._is_token_expired(cached):
            logger.info("Using cached OAuth token")
            return cached["access_token"]

        # Try to refresh if we have refresh token
        if cached and cached.get("refresh_token"):
            try:
                logger.info("Refreshing OAuth token")
                return await self._refresh_token(cached["refresh_token"])
            except Exception as e:
                logger.warning(f"Token refresh failed: {e}, re-authenticating")

        # Need fresh authentication
        return await self.authenticate()

    async def authenticate(self) -> str:
        """Perform full OAuth 2.1 flow with PKCE."""
        logger.info("Starting OAuth authentication flow")

        # 1. Discover OAuth endpoints
        auth_server = await self._discover_oauth_server()

        # 2. Generate PKCE challenge
        code_verifier = self._generate_code_verifier()
        code_challenge = self._generate_code_challenge(code_verifier)

        # 3. Get authorization code from user
        state = secrets.token_urlsafe(32)
        auth_code = await self._get_authorization_code(
            auth_server, code_challenge, state
        )

        # 4. Exchange code for tokens
        tokens = await self._exchange_code_for_token(
            auth_server, auth_code, code_verifier
        )

        # 5. Cache tokens
        self._save_tokens(tokens)

        logger.info("OAuth authentication complete")
        return tokens["access_token"]

    async def _discover_oauth_server(self) -> Dict[str, str]:
        """Discover OAuth endpoints from server metadata."""
        logger.info("Discovering OAuth server endpoints")

        # Parse base URL
        parsed = urlparse(self.server_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        # Try to discover authorization server metadata
        metadata_url = f"{base_url}/.well-known/oauth-authorization-server"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(metadata_url) as response:
                    if response.status == 200:
                        metadata = await response.json()
                        return {
                            "authorization_endpoint": metadata[
                                "authorization_endpoint"
                            ],
                            "token_endpoint": metadata["token_endpoint"],
                        }
            except Exception as e:
                logger.warning(f"Metadata discovery failed: {e}")

        # Fallback to default endpoints
        logger.info("Using default OAuth endpoints")
        return {
            "authorization_endpoint": f"{base_url}/authorize",
            "token_endpoint": f"{base_url}/token",
        }

    def _generate_code_verifier(self) -> str:
        """Generate PKCE code verifier."""
        return (
            base64.urlsafe_b64encode(secrets.token_bytes(32))
            .decode("utf-8")
            .rstrip("=")
        )

    def _generate_code_challenge(self, verifier: str) -> str:
        """Generate PKCE code challenge from verifier."""
        digest = hashlib.sha256(verifier.encode()).digest()
        return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")

    async def _get_authorization_code(
        self, auth_server: Dict[str, str], code_challenge: str, state: str
    ) -> str:
        """Open browser for user authorization and capture code."""
        # Build authorization URL
        auth_params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(self.scopes),
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }

        auth_url = f"{auth_server['authorization_endpoint']}?{urlencode(auth_params)}"

        # Start local server to receive callback
        auth_code_future = asyncio.Future()

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                # Parse query parameters
                query = parse_qs(urlparse(self.path).query)

                if "code" in query:
                    code = query["code"][0]
                    returned_state = query.get("state", [None])[0]

                    # Verify state matches
                    if returned_state == state:
                        auth_code_future.set_result(code)
                        self.send_response(200)
                        self.end_headers()
                        self.wfile.write(
                            b"Authorization successful! You can close this window."
                        )
                    else:
                        auth_code_future.set_exception(RuntimeError("State mismatch"))
                        self.send_response(400)
                        self.end_headers()
                        self.wfile.write(b"Authorization failed: state mismatch")
                else:
                    error = query.get("error", ["unknown"])[0]
                    auth_code_future.set_exception(
                        RuntimeError(f"Authorization error: {error}")
                    )
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(f"Authorization failed: {error}".encode())

            def log_message(self, format, *args):
                pass  # Suppress log messages

        # Start server
        server = HTTPServer(("localhost", 8765), CallbackHandler)

        # Open browser
        print("\n🔐 Opening browser for authentication...")
        print(f"If browser doesn't open, visit: {auth_url}\n")
        webbrowser.open(auth_url)

        # Wait for callback (with timeout)
        try:
            # Handle one request
            await asyncio.get_event_loop().run_in_executor(None, server.handle_request)

            # Get the authorization code
            auth_code = await asyncio.wait_for(
                auth_code_future, timeout=300
            )  # 5 min timeout

            return auth_code

        except asyncio.TimeoutError:
            raise RuntimeError("Authorization timeout - no response received")
        finally:
            server.server_close()

    async def _exchange_code_for_token(
        self, auth_server: Dict[str, str], code: str, verifier: str
    ) -> Dict[str, Any]:
        """Exchange authorization code for access token."""
        token_data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "code_verifier": verifier,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                auth_server["token_endpoint"], data=token_data
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    raise RuntimeError(f"Token exchange failed: {error}")

                return await response.json()

    async def _refresh_token(self, refresh_token: str) -> str:
        """Refresh access token using refresh token."""
        # Discover token endpoint
        auth_server = await self._discover_oauth_server()

        token_data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.client_id,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                auth_server["token_endpoint"], data=token_data
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    raise RuntimeError(f"Token refresh failed: {error}")

                tokens = await response.json()
                self._save_tokens(tokens)
                return tokens["access_token"]

    def _load_cached_token(self) -> Optional[Dict[str, Any]]:
        """Load token from cache file."""
        if not self.token_cache_file.exists():
            return None

        try:
            with open(self.token_cache_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cached token: {e}")
            return None

    def _save_tokens(self, tokens: Dict[str, Any]) -> None:
        """Save tokens to cache file."""
        self.token_cache_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.token_cache_file, "w") as f:
            json.dump(tokens, f, indent=2)

        # Set restrictive permissions
        self.token_cache_file.chmod(0o600)

    def _is_token_expired(self, token_data: Dict[str, Any]) -> bool:
        """Check if token is expired (simple check, could be improved)."""
        # For now, just return False and let the server tell us if expired
        # Could implement proper expiry checking with timestamps
        return False
