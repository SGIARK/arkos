-- Migration 0 — the whole schema (Task 0c).
--
-- Fresh cutover, not a history migration. 0001-0007 are deleted, not archived:
-- they built tables the redesign does not build, and nothing migrates forward.
-- This file is now the truth; docs/schema.md describes it and goes away.
--
-- Carried over, deliberately untouched:
--   waitlist      - real signups from the landing page
--   repeat_tasks  - watching is scrapped for this redesign, but the rows stay
--
-- Everything else is dropped. Every user_id is a Supabase auth UUID now, which
-- is why the old users table (username/last_seen, locally minted id) cannot be
-- kept: its ids do not mean the same thing.

-- ---------------------------------------------------------------- teardown --
DROP TABLE IF EXISTS conversation_context CASCADE;  -- merged into session_events
DROP TABLE IF EXISTS task_events          CASCADE;  -- merged into session_events
DROP TABLE IF EXISTS task_approvals       CASCADE;  -- -> approvals
DROP TABLE IF EXISTS user_sandboxes       CASCADE;  -- recreated, new shape
DROP TABLE IF EXISTS tasks                CASCADE;  -- -> sessions
DROP TABLE IF EXISTS users                CASCADE;  -- -> users, Supabase-keyed

DROP TYPE IF EXISTS task_status     CASCADE;
DROP TYPE IF EXISTS approval_kind   CASCADE;
DROP TYPE IF EXISTS approval_status CASCADE;

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
    -- Supabase Storage, one bucket, {user_id}/{project_id}/{file_id}. COPIED
    -- into the sandbox at lease acquisition, not live-mounted: e2b has no
    -- network filesystem.
    storage_path  TEXT        NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ------------------------------------ a task IS a session: one row, one id --
CREATE TABLE sessions (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id          UUID        NOT NULL REFERENCES users(id),
    project_id       UUID        REFERENCES projects(id),
    -- 'attended' | 'unattended'. A PHASE, not a type: it flips when the human
    -- says go.
    mode             TEXT        NOT NULL,
    title            TEXT,
    goal             TEXT,
    -- pending idle running awaiting_approval completed failed cancelled.
    -- idle = alive, waiting for a human; survives restarts.
    status           TEXT        NOT NULL,
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

-- ----------------------------------- the transcript: audit, never pruned ----
CREATE TABLE session_events (
    -- ONE global counter: ordering + resume cursor + event id. Gaps per session
    -- are fine; every read is "after N", never "count".
    seq         BIGSERIAL   PRIMARY KEY,
    session_id  UUID        NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    -- user content reasoning tool_call tool_result status todo budget
    -- lifecycle view_transform done
    kind        TEXT        NOT NULL,
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
-- Best-effort and batched: a failed write here is a lost diagnostic, never a
-- halted run. That is the whole reason it is not session_events.
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
    kind          TEXT        NOT NULL,   -- 'approval' | 'ask'
    prompt        TEXT        NOT NULL,
    answer        TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    answered_at   TIMESTAMPTZ
);

CREATE INDEX idx_approvals_pending ON approvals (session_id) WHERE answered_at IS NULL;

-- ---------------------------------------------------------- stateful hands --
CREATE TABLE resource_leases (
    resource_key  TEXT        PRIMARY KEY,  -- 'sandbox:{user}' | 'browser:{user}'
    session_id    UUID        NOT NULL REFERENCES sessions(id),
    acquired_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at    TIMESTAMPTZ NOT NULL
);

CREATE TABLE user_sandboxes (
    user_id       UUID        PRIMARY KEY REFERENCES users(id),
    sandbox_id    TEXT,                   -- e2b handle; cattle, may be respawned
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at  TIMESTAMPTZ
);

-- No server_name column. The key under config.yaml's mcp_servers: ('linear',
-- 'gmail') is an in-process label for logs and the tool registry, rebuilt from
-- config at every startup. Nothing durable references it, so renaming it is
-- free. Display name likewise comes from config at render time.
CREATE TABLE user_connections (
    user_id        UUID        NOT NULL REFERENCES users(id),
    -- THE identity. Smithery knows the server by its url; so do we.
    mcp_url        TEXT        NOT NULL,
    -- Issued ONCE at connect time and stored. Never recomputed from a formula,
    -- so there is no formula to drift against.
    connection_id  TEXT        NOT NULL,
    -- The fact of connection: what makes "restart = 0 Smithery PUTs" true.
    status         TEXT        NOT NULL,
    tools_cache    JSONB,
    refreshed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, mcp_url)
);

-- Same rule, no user column: no-auth servers (Slack's workspace bot token) have
-- no user, and user_connections.user_id is NOT NULL.
CREATE TABLE shared_connections (
    mcp_url        TEXT        PRIMARY KEY,
    connection_id  TEXT        NOT NULL,
    status         TEXT        NOT NULL,
    tools_cache    JSONB,
    refreshed_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
