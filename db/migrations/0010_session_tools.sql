-- Which MCP servers a session may reach.
--
-- The default is ours alone: a connected server is not a reachable one until it
-- is toggled into a session. An absent row is therefore OFF, and the only rows
-- here are the ones a human turned on — which is what makes an accidental
-- over-budget request impossible rather than unlikely.
--
-- Keyed by `mcp_url`, not by the config label, for the same reason
-- `user_connections` and `shared_connections` are: the key under config.yaml's
-- `mcp_servers:` is an in-process label rebuilt at every startup, so renaming it
-- is free and nothing durable may reference it. The url is the identity. The
-- table this replaces was keyed by the label, which is the bug.
--
-- `enabled` is kept rather than deleting the row on a toggle-off, so a write is
-- one upsert and a read never has to tell "turned off" from "never touched" by
-- absence alone. `updated_at` moves on every write and IS the drop order: when
-- the tool cap forces a server out, the most recently enabled one goes first.
--
-- WHY 0010 AND NOT 0009. A stopped first build of Task 11.5 applied a migration
-- named `0009_session_tools.sql`, then had its code reverted — leaving an empty
-- table keyed the wrong way and a `schema_migrations` row for a file no longer
-- in the tree. Re-issuing under the SAME name could never work: the runner
-- records by filename and skips anything already recorded, so the replacement
-- was silently never applied and every read of the new column failed with
-- `UndefinedColumnError`. A new number is the only version of this that a plain
-- `python db/migrate.py` actually performs. The stale row is deleted below
-- rather than left pointing at a missing file — in the same transaction, so a
-- half-done cleanup is not a state this can stop in.

DROP TABLE IF EXISTS session_tools CASCADE;

CREATE TABLE session_tools (
    session_id  UUID        NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    mcp_url     TEXT        NOT NULL,
    enabled     BOOLEAN     NOT NULL DEFAULT TRUE,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, mcp_url)
);

DELETE FROM schema_migrations WHERE name = '0009_session_tools.sql';
