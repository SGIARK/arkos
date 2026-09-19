#!/usr/bin/env python3
"""Migration runner: apply pending db/migrations/*.sql in lexical order.

IT SAYS WHERE IT IS GOING AND REFUSES A REMOTE TARGET WITHOUT BEING TOLD (G4,
2026-09-13). `load_dotenv` plus `DB_URL` meant a bare `python db/migrate.py`
aimed at the Supabase project, with no target argument, no printed destination
and no confirmation, so an operator applying a local migration LOCALLY reached
production instead. That happened, and it was benign only because the table
involved held no rows.

The local path stays one variable, since `override=False` means an exported
`DB_URL` wins over `.env`, and the dangerous path grows a deliberate gesture:
`--production`. There is only ONE database and it is production (owner,
2026-09-12), so this is not a staging mechanism; it is the difference between
meaning it and reaching it.
"""

import os
import sys
import urllib.parse
from pathlib import Path

import psycopg2

_PROJECT_ROOT = Path(__file__).parent.parent
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)
except Exception:
    pass


def get_connection_url():
    """Resolve the Postgres URL from DB_URL, else config.yaml, else POSTGRES_* vars."""
    db_url = os.environ.get("DB_URL")
    if db_url:
        return db_url

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


# Hosts that cannot be anybody's production database. Anything else needs
# `--production`, including a name that merely looks internal: the point is that
# the operator said so, not that the runner guessed well.
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", ""})


def describe(db_url: str) -> tuple[str, str]:
    """The host and database a DSN names, for printing. Never the credentials."""
    parsed = urllib.parse.urlsplit(db_url)
    return parsed.hostname or "", (parsed.path or "").lstrip("/") or "?"


def check_target(db_url: str, *, production_intended: bool) -> str | None:
    """Return why this target is refused, or None to proceed.

    FAIL CLOSED ON THE REMOTE CASE, which is the asymmetry that matters: a local
    apply that is refused costs a retry with a flag, and a production apply
    nobody meant costs whatever the migration did to real rows.
    """
    host, name = describe(db_url)
    if host in _LOCAL_HOSTS or production_intended:
        return None
    return (
        f"refusing to migrate {name} at {host}: that is not a local database.\n"
        "There is one database and it is production, so this needs saying out loud:\n"
        "  python db/migrate.py --production\n"
        "To migrate a local database instead, name it explicitly, which also keeps\n"
        "`.env` out of it:\n"
        "  DB_URL=postgresql://test:test@localhost:5432/test python db/migrate.py"
    )


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
        production_intended = "--production" in sys.argv[1:]
        db_url = get_connection_url()

        # PRINTED BEFORE ANYTHING IS TOUCHED, and printed whatever the verdict,
        # because the operator who was about to be surprised is the one who
        # needs it. Host and database only: a DSN carries a password.
        host, name = describe(db_url)
        print(f"target: {name} at {host or '(local socket)'}")

        refusal = check_target(db_url, production_intended=production_intended)
        if refusal:
            print(refusal, file=sys.stderr)
            return 2

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
