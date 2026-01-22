"""
OAuth authentication endpoints for per-user MCP access.
"""

import os
import json
import secrets
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlencode

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials

router = APIRouter(prefix="/auth", tags=["auth"])

# In-memory state storage (use Redis in production)
_oauth_states: dict = {}

# Google OAuth scopes for Calendar
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/calendar.events",
]


def get_google_flow(redirect_uri: str) -> Flow:
    """Create Google OAuth flow from credentials file."""
    creds_path = os.environ.get("GOOGLE_OAUTH_CREDENTIALS")
    if not creds_path or not os.path.exists(creds_path):
        raise HTTPException(500, "Google OAuth credentials not configured")

    flow = Flow.from_client_secrets_file(
        creds_path,
        scopes=GOOGLE_SCOPES,
        redirect_uri=redirect_uri,
    )
    return flow


@router.get("/google/login")
async def google_login(request: Request, user_id: str):
    """
    Initiate Google OAuth flow for a user.

    Query params:
        user_id: The user ID to associate the token with
    """
    if not user_id:
        raise HTTPException(400, "user_id is required")

    # Generate state token
    state = secrets.token_urlsafe(32)
    _oauth_states[state] = {
        "user_id": user_id,
        "created_at": datetime.utcnow(),
    }

    # Build redirect URI
    redirect_uri = str(request.url_for("google_callback"))
    flow = get_google_flow(redirect_uri)

    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        state=state,
        prompt="consent",
    )

    return RedirectResponse(auth_url)


@router.get("/google/callback")
async def google_callback(request: Request, code: str = None, state: str = None, error: str = None):
    """Handle Google OAuth callback."""
    # Import here to avoid circular imports
    from tool_module.token_store import UserTokenStore
    from config_module.loader import config

    if error:
        return HTMLResponse(f"<h1>Authorization Failed</h1><p>{error}</p>", status_code=400)

    if not state or state not in _oauth_states:
        return HTMLResponse("<h1>Invalid State</h1><p>Authorization expired or invalid.</p>", status_code=400)

    state_data = _oauth_states.pop(state)
    user_id = state_data["user_id"]

    # Exchange code for tokens
    redirect_uri = str(request.url_for("google_callback"))
    flow = get_google_flow(redirect_uri)

    try:
        flow.fetch_token(code=code)
    except Exception as e:
        return HTMLResponse(f"<h1>Token Exchange Failed</h1><p>{e}</p>", status_code=400)

    credentials = flow.credentials

    # Store token in database
    token_store = UserTokenStore(config.get("database.url"))
    token_store.set_token(
        user_id=user_id,
        service="google-calendar",
        access_token=credentials.token,
        refresh_token=credentials.refresh_token,
        expires_at=credentials.expiry,
        token_data={
            "token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "token_uri": credentials.token_uri,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "scopes": list(credentials.scopes) if credentials.scopes else GOOGLE_SCOPES,
        },
    )

    return HTMLResponse(f"""
        <h1>Authorization Successful!</h1>
        <p>Google Calendar connected for user: <strong>{user_id}</strong></p>
        <p>You can close this window.</p>
    """)


@router.get("/google/status")
async def google_status(user_id: str):
    """Check if a user has Google Calendar connected."""
    from tool_module.token_store import UserTokenStore
    from config_module.loader import config

    if not user_id:
        raise HTTPException(400, "user_id is required")

    token_store = UserTokenStore(config.get("database.url"))
    has_token = token_store.has_token(user_id, "google-calendar")

    return {"user_id": user_id, "connected": has_token}


@router.delete("/google/disconnect")
async def google_disconnect(user_id: str):
    """Disconnect Google Calendar for a user."""
    from tool_module.token_store import UserTokenStore
    from config_module.loader import config

    if not user_id:
        raise HTTPException(400, "user_id is required")

    token_store = UserTokenStore(config.get("database.url"))
    deleted = token_store.delete_token(user_id, "google-calendar")

    return {"user_id": user_id, "disconnected": deleted}
