-- Migration 0: the whole schema. A fresh cutover, not a history migration.
--
-- Carried over untouched: waitlist (real landing-page signups) and repeat_tasks.
-- Everything else is dropped, including the old users table: user_id is a Supabase
-- auth UUID now, so its locally minted ids do not mean the same thing.

-- WARNING: user_sandboxes is dropped, not migrated (user_id was TEXT, is UUID now).
-- Every live e2b handle is lost, so users' sandbox filesystems become unreachable
-- and the orphans keep running and billing until their own idle timeout. Pause or
-- reap all sandboxes BEFORE running this.

-- ---------------------------------------------------------------- teardown --
DROP TABLE IF EXISTS conversation_context  CASCADE;  -- merged into session_events
DROP TABLE IF EXISTS task_events           CASCADE;  -- merged into session_events
DROP TABLE IF EXISTS computer_task_events  CASCADE;  -- merged into session_events
DROP TABLE IF EXISTS computer_tasks        CASCADE;  -- dropped with the old chain
DROP TABLE IF EXISTS task_approvals        CASCADE;  -- -> approvals
DROP TABLE IF EXISTS user_sandboxes        CASCADE;  -- recreated, new shape
DROP TABLE IF EXISTS tasks                 CASCADE;  -- -> sessions
DROP TABLE IF EXISTS users                 CASCADE;  -- -> users, Supabase-keyed

-- Pre-Smithery OAuth storage, plaintext tokens, unread since Smithery took over
-- the credential vault. Nothing in the new schema replaces it.
DROP TABLE IF EXISTS user_oauth_tokens     CASCADE;

-- mem0's pgvector collection, which lived in THIS database. The real vectors are
-- in the vecs schema, so dropping public.memories alone leaves them behind.
-- public.memories is a view when mem0 went through `vecs` and a table when it
-- did not. IF EXISTS suppresses "does not exist", NOT "is not a table", so
-- either bare form aborts on the other shape; dispatch on relkind instead.
DO $$
DECLARE kind "char";
BEGIN
    SELECT c.relkind INTO kind
      FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
     WHERE n.nspname = 'public' AND c.relname = 'memories';
    IF    kind = 'v' THEN EXECUTE 'DROP VIEW public.memories CASCADE';
    ELSIF kind = 'm' THEN EXECUTE 'DROP MATERIALIZED VIEW public.memories CASCADE';
    ELSIF kind IN ('r', 'p', 'f') THEN EXECUTE 'DROP TABLE public.memories CASCADE';
    END IF;
END $$;
DROP TABLE IF EXISTS vecs.memories          CASCADE;
DROP TABLE IF EXISTS vecs.memories_entities CASCADE;
DROP TABLE IF EXISTS vecs.mem0migrations    CASCADE;
-- vecs.urop_benchmark* stay: different project, not mem0's, not ours to drop.

DROP TYPE IF EXISTS task_status     CASCADE;
DROP TYPE IF EXISTS approval_kind   CASCADE;
DROP TYPE IF EXISTS approval_status CASCADE;

-- Orphaned with the tasks trigger that was its only caller.
DROP FUNCTION IF EXISTS set_updated_at() CASCADE;

-- The drops above are unconditional: a DB may have stopped at 0006, so
-- computer_tasks may or may not exist.

-- The deleted migrations must not look applied on a DB that already ran them.
DELETE FROM schema_migrations WHERE name ~ '^000[1-7]_';

-- ---------------------------------------------------------------- identity --
CREATE TABLE users (
    id          UUID        PRIMARY KEY,   -- = Supabase auth sub
    email       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- -------------------------------------------------------------- containers --
CREATE TABLE projects (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID        NOT NULL REFERENCES users(id),
    title       TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE project_files (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id    UUID        NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name          TEXT        NOT NULL,
    size_bytes    BIGINT      NOT NULL,
    -- Supabase Storage {user_id}/{project_id}/{file_id}; copied into the sandbox
    -- at lease acquisition, not live-mounted: e2b has no network filesystem.
    storage_path  TEXT        NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ------------------------------------ a task IS a session: one row, one id --
CREATE TABLE sessions (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id          UUID        NOT NULL REFERENCES users(id),
    project_id       UUID        REFERENCES projects(id),
    -- A PHASE, not a type: it flips when the human says go.
    mode             TEXT        NOT NULL CHECK (mode IN ('attended', 'unattended')),
    title            TEXT,
    goal             TEXT,
    -- idle = alive, waiting for a human. CHECKed because an out-of-vocabulary
    -- write would strand the session: transition()'s WHERE status=expected stops matching.
    status           TEXT        NOT NULL CHECK (status IN (
                         'pending', 'idle', 'running', 'awaiting_approval',
                         'completed', 'failed', 'cancelled')),
    terminal_reason  TEXT,                  -- = done.reason, verbatim
    cursor_seq       BIGINT      NOT NULL DEFAULT 0,
    hops_used        INT         NOT NULL DEFAULT 0,
    lease_owner      TEXT,                  -- worker id holding this session
    lease_expires    TIMESTAMPTZ,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at         TIMESTAMPTZ
);

-- Operator-console seams: cheap now, painful later.
CREATE INDEX idx_sessions_triage ON sessions (status, terminal_reason, ended_at);
CREATE INDEX idx_sessions_user   ON sessions (user_id, ended_at);
-- the new_sessions_per_hour quota query, which ended_at cannot serve
CREATE INDEX idx_sessions_created ON sessions (user_id, created_at);

-- ----------------------------------- the transcript: audit, never pruned ----
CREATE TABLE session_events (
    -- One global counter serving as ordering, resume cursor and event id; per-session
    -- gaps are fine because every read is "after N", never "count". BIGSERIAL assigns
    -- before commit, so append() must hold pg_advisory_xact_lock(session_id) or a
    -- reader can skip a seq that commits late.
    seq         BIGSERIAL   PRIMARY KEY,
    session_id  UUID        NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    kind        TEXT        NOT NULL CHECK (kind IN (
                    'user', 'content', 'reasoning', 'tool_call', 'tool_result',
                    'status', 'todo', 'budget', 'lifecycle', 'view_transform', 'done')),
    version     INT         NOT NULL DEFAULT 1,  -- readers upcast; rows never rewritten
    payload     JSONB       NOT NULL,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_session_events_cursor ON session_events (session_id, seq);

-- ------------------------------------------------- oversized tool outputs --
CREATE TABLE result_blobs (
    ref         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID        NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    content     TEXT        NOT NULL,   -- full text; the event holds a preview
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ----------------------------------------- operational log: internal, pruned
-- Best-effort and batched: a failed write loses a diagnostic, never halts a run.
CREATE TABLE system_events (
    id          BIGSERIAL   PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
    level       TEXT        NOT NULL,   -- info warn error
    event       TEXT        NOT NULL,
    session_id  UUID,                   -- correlates noise back to a session
    user_id     UUID,
    fields      JSONB       NOT NULL DEFAULT '{}'
);

CREATE INDEX idx_system_events_ts      ON system_events (ts);
CREATE INDEX idx_system_events_session ON system_events (session_id);

-- ------------------------------------------------------------------ pauses --
CREATE TABLE approvals (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id    UUID        NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    -- Stays OPEN across the park; the response event closes it.
    tool_call_id  TEXT        NOT NULL,
    kind          TEXT        NOT NULL CHECK (kind IN ('approval', 'ask')),
    prompt        TEXT        NOT NULL,
    answer        TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    answered_at   TIMESTAMPTZ
);

CREATE INDEX idx_approvals_pending ON approvals (session_id) WHERE answered_at IS NULL;
-- a double-park is impossible, not merely unlikely
CREATE UNIQUE INDEX idx_approvals_one_open_per_call
    ON approvals (session_id, tool_call_id) WHERE answered_at IS NULL;

-- ---------------------------------------------------------- stateful hands --
CREATE TABLE resource_leases (
    resource_key  TEXT        PRIMARY KEY,  -- 'sandbox:{user}' | 'browser:{user}'
    session_id    UUID        NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    acquired_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at    TIMESTAMPTZ NOT NULL
);

CREATE TABLE user_sandboxes (
    user_id       UUID        PRIMARY KEY REFERENCES users(id),
    sandbox_id    TEXT,                   -- e2b handle; cattle, may be respawned
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at  TIMESTAMPTZ
);

-- No server_name column: the config.yaml mcp_servers key is an in-process label
-- rebuilt at startup, and nothing durable references it.
CREATE TABLE user_connections (
    user_id        UUID        NOT NULL REFERENCES users(id),
    -- THE identity. Smithery knows the server by its url; so do we.
    mcp_url        TEXT        NOT NULL,
    -- Issued once at connect time and stored, never recomputed, so it cannot drift.
    connection_id  TEXT        NOT NULL,
    -- The fact of connection: what makes "restart = 0 Smithery PUTs" true.
    status         TEXT        NOT NULL,
    tools_cache    JSONB,
    refreshed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, mcp_url)
);

-- No user column: no-auth servers have no user, and user_connections.user_id is NOT NULL.
CREATE TABLE shared_connections (
    mcp_url        TEXT        PRIMARY KEY,
    connection_id  TEXT        NOT NULL,
    status         TEXT        NOT NULL,
    tools_cache    JSONB,
    refreshed_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
