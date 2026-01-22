"""
Per-user OAuth token storage for MCP servers.
"""

import os
import json
import psycopg2
from typing import Optional, Dict, Any
from datetime import datetime, timezone


class UserTokenStore:
    """
    Stores and retrieves OAuth tokens per user in PostgreSQL.

    Table schema (create if not exists):
        CREATE TABLE IF NOT EXISTS user_oauth_tokens (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            service VARCHAR(255) NOT NULL,
            access_token TEXT NOT NULL,
            refresh_token TEXT,
            expires_at TIMESTAMP,
            token_data JSONB,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(user_id, service)
        );
    """

    def __init__(self, db_url: str):
        self.db_url = db_url
        self._ensure_table()

    def _ensure_table(self):
        """Create the tokens table if it doesn't exist."""
        conn = psycopg2.connect(self.db_url)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_oauth_tokens (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                service VARCHAR(255) NOT NULL,
                access_token TEXT NOT NULL,
                refresh_token TEXT,
                expires_at TIMESTAMP,
                token_data JSONB,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(user_id, service)
            )
        """)
        conn.commit()
        cur.close()
        conn.close()

    def get_token(self, user_id: str, service: str) -> Optional[Dict[str, Any]]:
        """Get stored token for a user and service."""
        conn = psycopg2.connect(self.db_url)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT access_token, refresh_token, expires_at, token_data
            FROM user_oauth_tokens
            WHERE user_id = %s AND service = %s
            """,
            (user_id, service),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            return None

        return {
            "access_token": row[0],
            "refresh_token": row[1],
            "expires_at": row[2].isoformat() if row[2] else None,
            "token_data": row[3] or {},
        }

    def set_token(
        self,
        user_id: str,
        service: str,
        access_token: str,
        refresh_token: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        token_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store or update token for a user and service."""
        conn = psycopg2.connect(self.db_url)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO user_oauth_tokens
                (user_id, service, access_token, refresh_token, expires_at, token_data, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (user_id, service)
            DO UPDATE SET
                access_token = EXCLUDED.access_token,
                refresh_token = COALESCE(EXCLUDED.refresh_token, user_oauth_tokens.refresh_token),
                expires_at = EXCLUDED.expires_at,
                token_data = EXCLUDED.token_data,
                updated_at = NOW()
            """,
            (
                user_id,
                service,
                access_token,
                refresh_token,
                expires_at,
                json.dumps(token_data) if token_data else None,
            ),
        )
        conn.commit()
        cur.close()
        conn.close()
        return True

    def delete_token(self, user_id: str, service: str) -> bool:
        """Delete token for a user and service."""
        conn = psycopg2.connect(self.db_url)
        cur = conn.cursor()
        cur.execute(
            """
            DELETE FROM user_oauth_tokens
            WHERE user_id = %s AND service = %s
            """,
            (user_id, service),
        )
        deleted = cur.rowcount > 0
        conn.commit()
        cur.close()
        conn.close()
        return deleted

    def has_token(self, user_id: str, service: str) -> bool:
        """Check if user has a token for a service."""
        return self.get_token(user_id, service) is not None

    def list_user_services(self, user_id: str) -> list:
        """List all services a user has tokens for."""
        conn = psycopg2.connect(self.db_url)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT service FROM user_oauth_tokens WHERE user_id = %s
            """,
            (user_id,),
        )
        services = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()
        return services

    def write_token_file(self, user_id: str, service: str, file_path: str) -> bool:
        """
        Write user's token to a file for MCP subprocess consumption.
        Returns True if token was written, False if no token exists.
        """
        token = self.get_token(user_id, service)
        if not token:
            return False

        token_data = token.get("token_data", {})

        # Convert to format expected by google-calendar-mcp
        # Uses 'access_token' instead of 'token'
        formatted = {
            "access_token": token_data.get("token"),
            "refresh_token": token_data.get("refresh_token"),
            "token_uri": token_data.get("token_uri"),
            "client_id": token_data.get("client_id"),
            "client_secret": token_data.get("client_secret"),
            "scopes": token_data.get("scopes"),
        }

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(formatted, f)
        os.chmod(file_path, 0o600)
        return True
