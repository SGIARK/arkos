"""
Demo-style auth: pick or type a username, the server finds-or-creates
a user row, returns a JWT. Swap for email+password or Supabase GoTrue
later without breaking tasks.user_id (UUID).
"""

from __future__ import annotations

import re
import uuid
from typing import Any

import psycopg2
import psycopg2.extras
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from base_module.jwt_utils import CurrentUser, issue_token
from config_module.loader import config

router = APIRouter(prefix="/auth", tags=["auth"])

_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{2,64}$")


class DemoLoginRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=64)


class LoginResponse(BaseModel):
    token: str
    user_id: str
    username: str


class MeResponse(BaseModel):
    user_id: str
    username: str
    slack_user_id: str | None = None


class SlackConnectRequest(BaseModel):
    slack_user_id: str = Field(..., pattern=r"^U[A-Z0-9]{8,}$")


def _connect():
    return psycopg2.connect(config.get("database.url"))


def _find_or_create_user(username: str) -> tuple[str, str]:
    """Return (user_id, username). Creates the row if missing."""
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id, username FROM users WHERE username = %s", (username,))
            row = cur.fetchone()
            if row:
                cur.execute("UPDATE users SET last_seen = now() WHERE id = %s", (row["id"],))
                conn.commit()
                return str(row["id"]), row["username"]

            new_id = str(uuid.uuid4())
            cur.execute(
                "INSERT INTO users (id, username) VALUES (%s, %s) RETURNING id, username",
                (new_id, username),
            )
            created = cur.fetchone()
            conn.commit()
            return str(created["id"]), created["username"]
    finally:
        conn.close()


@router.post("/demo-login", response_model=LoginResponse)
async def demo_login(req: DemoLoginRequest) -> LoginResponse:
    """
    POST /auth/demo-login
    Body: {"username": "nate"}
    Finds or creates the user, returns a JWT. No password.
    """
    username = req.username.strip()
    if not _USERNAME_RE.match(username):
        raise HTTPException(400, "username must match [a-zA-Z0-9_.-]{2,64}")

    try:
        user_id, uname = _find_or_create_user(username)
    except psycopg2.Error as e:
        raise HTTPException(500, f"db error: {e}") from e

    token = issue_token(user_id, uname)
    return LoginResponse(token=token, user_id=user_id, username=uname)


def _get_slack_user_id(user_id: str) -> str | None:
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT slack_user_id FROM users WHERE id = %s", (user_id,))
            row = cur.fetchone()
            return row["slack_user_id"] if row else None
    finally:
        conn.close()


@router.get("/me", response_model=MeResponse)
async def me(current: dict[str, Any] = CurrentUser) -> MeResponse:
    """GET /auth/me. Returns the authenticated user from the Bearer token."""
    slack_user_id = _get_slack_user_id(current["user_id"])
    return MeResponse(user_id=current["user_id"], username=current["username"], slack_user_id=slack_user_id)


@router.post("/slack-connect", status_code=204, response_model=None)
async def slack_connect(
    req: SlackConnectRequest,
    current: dict[str, Any] = CurrentUser,
) -> None:
    """
    POST /auth/slack-connect
    Body: {"slack_user_id": "U012AB3CD"}
    Links the calling user's ARKOS account to their Slack member ID for task notifications.
    """
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET slack_user_id = %s WHERE id = %s",
                (req.slack_user_id, current["user_id"]),
            )
        conn.commit()
    except psycopg2.Error as e:
        raise HTTPException(500, f"db error: {e}") from e
    finally:
        conn.close()
