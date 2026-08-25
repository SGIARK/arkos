"""Connection-URL resolution and migration helpers in db/migrate.py.

The Postgres-touching functions run against MagicMock connections.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from db import migrate

# ---------------------------------------------------------------------------
# get_connection_url
# ---------------------------------------------------------------------------


class TestGetConnectionUrl:
    def test_env_var_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("DB_URL", "postgresql://from-env/db")
        assert migrate.get_connection_url() == "postgresql://from-env/db"

    def test_falls_back_to_constructed_default(self, monkeypatch):
        monkeypatch.delenv("DB_URL", raising=False)
        monkeypatch.setenv("POSTGRES_PASSWORD", "secret")
        monkeypatch.setenv("POSTGRES_HOST", "db.example.com")
        monkeypatch.setenv("POSTGRES_PORT", "6543")
        monkeypatch.setenv("POSTGRES_DB", "arkos")
        monkeypatch.setenv("POSTGRES_USER", "supabase")

        # Force the ConfigLoader path to fail so the constructed default runs.
        with patch("config_module.loader.config.get", side_effect=RuntimeError("no config")):
            url = migrate.get_connection_url()

        assert url == "postgresql://supabase:secret@db.example.com:6543/arkos"

    def test_constructed_default_fills_missing_values(self, monkeypatch):
        monkeypatch.delenv("DB_URL", raising=False)
        for var in ("POSTGRES_PASSWORD", "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER"):
            monkeypatch.delenv(var, raising=False)

        with patch("config_module.loader.config.get", side_effect=RuntimeError("no config")):
            url = migrate.get_connection_url()

        assert url == "postgresql://postgres:postgres@localhost:5432/postgres"

    def test_uses_config_loader_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("DB_URL", raising=False)
        with patch("config_module.loader.config.get", return_value="postgresql://from-config/x"):
            assert migrate.get_connection_url() == "postgresql://from-config/x"

    def test_skips_unresolved_config_value(self, monkeypatch):
        # A literal "${DB_URL}" means substitution never happened.
        monkeypatch.delenv("DB_URL", raising=False)
        monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
        with patch("config_module.loader.config.get", return_value="${DB_URL}"):
            url = migrate.get_connection_url()
        assert url.startswith("postgresql://postgres:postgres@localhost:5432/")


# ---------------------------------------------------------------------------
# Migration helpers
# ---------------------------------------------------------------------------


class TestMigrationHelpers:
    def _conn_with_cursor(self, fetchone_value=None):
        conn = MagicMock()
        cur = MagicMock()
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)
        cur.fetchone.return_value = fetchone_value
        conn.cursor.return_value = cur
        return conn, cur

    def test_ensure_migrations_table_creates_and_commits(self):
        conn, cur = self._conn_with_cursor()
        migrate.ensure_migrations_table(conn)
        sql = cur.execute.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS schema_migrations" in sql
        conn.commit.assert_called_once()

    def test_already_applied_true(self):
        conn, cur = self._conn_with_cursor(fetchone_value=(1,))
        assert migrate.already_applied(conn, "0001_init.sql") is True
        cur.execute.assert_called_once_with(
            "SELECT 1 FROM schema_migrations WHERE name = %s",
            ("0001_init.sql",),
        )

    def test_already_applied_false(self):
        conn, cur = self._conn_with_cursor(fetchone_value=None)
        assert migrate.already_applied(conn, "0002_users.sql") is False

    def test_apply_migration_executes_sql_and_records_name(self, tmp_path: Path):
        sql_file = tmp_path / "0042_demo.sql"
        sql_file.write_text("CREATE TABLE demo (id SERIAL PRIMARY KEY);")
        conn, cur = self._conn_with_cursor()

        migrate.apply_migration(conn, sql_file)

        # Two execute calls: the migration SQL, then the bookkeeping insert.
        first_call_sql, *_ = cur.execute.call_args_list[0][0]
        assert "CREATE TABLE demo" in first_call_sql

        second_call = cur.execute.call_args_list[1]
        assert "INSERT INTO schema_migrations" in second_call[0][0]
        assert second_call[0][1] == ("0042_demo.sql",)

        conn.commit.assert_called_once()


# ---------------------------------------------------------------------------
# main() smoke
# ---------------------------------------------------------------------------


class TestMainSmoke:
    def test_returns_1_on_db_failure(self, monkeypatch):
        monkeypatch.setenv("DB_URL", "postgresql://nonexistent-host:1/db")
        with patch("db.migrate.psycopg2.connect", side_effect=Exception("boom")):
            assert migrate.main() == 1

    def test_returns_0_when_no_migrations(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DB_URL", "postgresql://stub")
        empty_dir = tmp_path / "migrations"
        empty_dir.mkdir()

        with (
            patch("db.migrate.psycopg2.connect"),
            patch("db.migrate.ensure_migrations_table"),
            patch("db.migrate.Path") as mock_path,
        ):
            # Path(__file__).parent / "migrations" must resolve to empty_dir.
            mock_path.return_value.parent.__truediv__.return_value = empty_dir
            assert migrate.main() == 0


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")
