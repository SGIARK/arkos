#!/usr/bin/env python3
"""
Migration runner for Arkos database schema.

Applies every pending migration in db/migrations/ in lexical order.
Tracks applied migrations in a small `schema_migrations` table.
Reads connection from DB_URL env var or constructs from POSTGRES_PASSWORD.
"""

import os
import sys
from pathlib import Path

import psycopg2

# Load the same .env the app uses before reading DB_URL.
_PROJECT_ROOT = Path(__file__).parent.parent
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)
except Exception:
    pass


def get_connection_url():
    """
    Resolve the Postgres connection URL the same way the running backend does:
      1. DB_URL env var (from shell or .env)
      2. config_module.loader's ConfigLoader (substitutes ${DB_URL} from .env)
      3. Constructed default from POSTGRES_* env vars
    """
    db_url = os.environ.get("DB_URL")
    if db_url:
        return db_url

    # Fall back to ConfigLoader so we stay in sync with the api layer (Task 4)
    try:
        sys.path.insert(0, str(_PROJECT_ROOT))
        from config_module.loader import config  # type: ignore

        resolved = config.get("database.url")
        if resolved and "${" not in str(resolved):
            return resolved
    except Exception as e:
        print(f"(config_module loader unavailable: {e})", file=sys.stderr)

    password = os.environ.get("POSTGRES_PASSWORD", "postgres")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    dbname = os.environ.get("POSTGRES_DB", "postgres")
    user = os.environ.get("POSTGRES_USER", "postgres")

    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


def ensure_migrations_table(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                name        TEXT        PRIMARY KEY,
                applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
    conn.commit()


def already_applied(conn, name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM schema_migrations WHERE name = %s", (name,))
        return cur.fetchone() is not None


def apply_migration(conn, path: Path) -> None:
    sql = path.read_text()
    with conn.cursor() as cur:
        cur.execute(sql)
        cur.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (path.name,))
    conn.commit()


def main():
    try:
        db_url = get_connection_url()
        conn = psycopg2.connect(db_url)

        ensure_migrations_table(conn)

        migrations_dir = Path(__file__).parent / "migrations"
        if not migrations_dir.is_dir():
            print(f"No migrations directory at {migrations_dir}", file=sys.stderr)
            return 1

        files = sorted(migrations_dir.glob("*.sql"))
        if not files:
            print("No migration files found.")
            return 0

        applied = 0
        for path in files:
            if already_applied(conn, path.name):
                print(f"- {path.name} (already applied)")
                continue
            print(f"+ applying {path.name}")
            apply_migration(conn, path)
            applied += 1

        conn.close()
        print(f"Done. {applied} migration(s) applied.")
        return 0

    except Exception as e:
        print(f"Migration failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
