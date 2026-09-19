-- Connections are keyed by the Arcade app, not by a url (Task 11.10).
--
-- Smithery is gone, and with it the premise these tables were drawn on: that a
-- server IS a url, one url per server, and that we mint a `connection_id` that
-- Smithery accepts as a path segment. All seven servers now live behind ONE
-- Arcade MCP Gateway, where a "server" is a tool-name PREFIX (`Gmail_*`,
-- `Linear_*`) inside a single flat tool list. Keeping `mcp_url` would have made
-- six rows share one value under a primary key that forbids it.
--
-- The key is the PREFIX, not the gateway url and not the url plus a column.
-- The gateway slug is infrastructure and can change — recreate the gateway and
-- every url-keyed row orphans — while the grants themselves live Arcade-side
-- keyed by user id and survive that untouched. A row's honest meaning is now
-- "user X connected Gmail", so it is keyed (user_id, server) and the gateway
-- url lives only in config, where infrastructure belongs.
--
-- `connection_id` and `tools_cache` go with the premise. There is no connection
-- object at Arcade for us to name, so nothing is minted and the write-before-PUT
-- rule has no PUT to precede; and the gateway's tool list is one list for the
-- whole gateway rather than a per-user fact, so caching it per user stored six
-- copies of the same thing. It is cached in-process now, on the same TTL.
--
-- EVERY EXISTING ROW IS DELETED. Not a precaution — every one of them is keyed
-- to a dead `*.run.tools` url and holds a Smithery connection id for a grant
-- that no longer exists. There is nothing here to carry across, and leaving the
-- rows would strand exactly the remnants this card exists to remove. Users
-- reconnect from the settings panel, and sessions re-enable their servers by
-- hand, the same terms 11.5 set when the toggles first landed.

BEGIN;

-- --- the per-user connections ------------------------------------------------

DELETE FROM user_connections;

ALTER TABLE user_connections DROP CONSTRAINT user_connections_pkey;
ALTER TABLE user_connections DROP COLUMN connection_id;
ALTER TABLE user_connections DROP COLUMN tools_cache;
ALTER TABLE user_connections RENAME COLUMN mcp_url TO server;
ALTER TABLE user_connections ADD PRIMARY KEY (user_id, server);

COMMENT ON COLUMN user_connections.server IS
    'The Arcade app prefix (Gmail, Linear, ...), which is what a tool name is prefixed with.';

-- --- the shared half, which now has no possible writer ------------------------
--
-- Every connector behind the gateway is per-user, Slack (the only
-- `requires_auth: false` server) is dropped entirely, and Google Search is one
-- of OUR tools with no row anywhere. A table nothing can write is the "no
-- remnants" case; git remembers the DDL if a shared-server concept returns.

DROP TABLE shared_connections;

-- --- the session toggles ------------------------------------------------------
--
-- Keyed by the same identity as the connections, for the same reason it was
-- keyed by url before: a `mcp_servers:` config key is an in-process label and
-- nothing durable may reference it. An Arcade prefix is the vendor's own name
-- for the app, not ours to rename.

DELETE FROM session_tools;

ALTER TABLE session_tools DROP CONSTRAINT session_tools_pkey;
ALTER TABLE session_tools RENAME COLUMN mcp_url TO server;
ALTER TABLE session_tools ADD PRIMARY KEY (session_id, server);

COMMENT ON COLUMN session_tools.server IS
    'The Arcade app prefix this session was given; absent means off.';

COMMIT;
