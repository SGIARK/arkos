-- Connections move from Arcade to Composio managed auth (Task 11.10.2).
--
-- The KEY does not change shape: a "server" is still a tool-name prefix inside
-- one flat list, and it is still what `user_connections` and `session_tools` are
-- keyed by. What changes is the vendor's spelling of it. Arcade prefixed tools
-- `Gmail_ListEmails`; Composio prefixes them `GMAIL_FETCH_EMAILS`. So `Gmail`
-- becomes `GMAIL`, `MicrosoftOutlookMail` becomes `OUTLOOK`, and the prefix is
-- once again the vendor's own name for the toolkit rather than a label of ours.
--
-- `connected_account_id` comes back, and it is NOT the `connection_id` 0019
-- deleted. That column existed because Smithery made us mint an id and PUT it;
-- this one is Composio's own id for a grant, handed to us at the end of the
-- consent flow in `/connections/done?connected_account_id=...` and needed again
-- to DELETE the grant on disconnect. It is nullable because a row is written the
-- moment consent starts, before any id exists.
--
-- `reconnect` joins the status vocabulary. Composio reports ERRORED for a grant
-- that has expired or been revoked provider-side, and a call against one comes
-- back as a SUCCESSFUL tools/call carrying `isError` and "No connected account
-- found for user ID ...". Mapping that to `pending` would read as "never
-- connected" and mapping it to `connected` would loop the model into a wall;
-- `reconnect` says what it is — the human authorized this once and has to again.
--
-- EVERY EXISTING ROW IS DELETED, on the same reasoning 0019 used. Every row is
-- keyed to an Arcade prefix naming a grant that lives at Arcade, and no Arcade
-- grant is reachable through Composio. There is nothing to carry across. Users
-- reconnect from the settings panel; sessions re-enable their toggles by hand.

BEGIN;

-- --- the per-user connections ------------------------------------------------

DELETE FROM user_connections;

ALTER TABLE user_connections ADD COLUMN connected_account_id TEXT;

COMMENT ON COLUMN user_connections.server IS
    'The Composio toolkit prefix, upper snake (GMAIL, LINEAR, GOOGLEDRIVE) -- what a tool name is prefixed with.';
COMMENT ON COLUMN user_connections.connected_account_id IS
    'Composio''s id for the grant. Arrives with the consent callback; needed to revoke. Null until consent completes.';
COMMENT ON COLUMN user_connections.status IS
    'pending | connected | reconnect. `reconnect` is Composio ERRORED: authorized once, needs authorizing again.';

-- --- the session toggles ------------------------------------------------------
--
-- Same identity as the connections, so the same sweep. A toggle naming `Gmail`
-- would silently never match a manifest built from `GMAIL_*` tools: the session
-- would look enabled and ship no tools, which is the worst of both readings.

DELETE FROM session_tools;

COMMENT ON COLUMN session_tools.server IS
    'The Composio toolkit prefix this session was given; absent means off.';

COMMIT;
