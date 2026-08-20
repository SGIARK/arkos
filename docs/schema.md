# Migration 0 — the whole schema

The DB is rebuilt from scratch (Task 0c). No history migration, no bridge
columns. `db/migrations/0001`-`0007` are deleted, not archived — they built tables
the redesign does not build. `waitlist` is the one table carried over; it has real
signups. Every `user_id` is a Supabase UUID, so the `_user_uuid()` hash bridge in
`task_store.py` is deleted with them.

Two logging tables, deliberately different in nature:
**`session_events`** is the transcript (user-visible, audit, never pruned, a
failed append halts the run). **`system_events`** is operational (internal,
best-effort, pruned at 30 days, a failed write is just a lost diagnostic).

---

```sql
-- identity ------------------------------------------------------------------
users (
  id               uuid primary key,     -- = Supabase auth sub
  email            text,
  -- The chat the app lands in, made on first login and set once. ON DELETE SET
  -- NULL: deleting a session must never delete the person whose chat it was.
  home_session_id  uuid references sessions(id) on delete set null,
  created_at       timestamptz not null default now()
)

-- containers ----------------------------------------------------------------
projects (
  id            uuid primary key default gen_random_uuid(),
  user_id       uuid not null references users(id),
  title         text not null,           -- a LABEL. Renaming changes only this.
  slug          text not null,           -- the folder: ~/projects/<slug>/, set ONCE
                                         -- at creation and never moved by a rename,
                                         -- because it is the only durable path the
                                         -- agent has. Defaulted uniquely so an
                                         -- insert that does not care still gets one.
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now()
)
create unique index on projects (user_id, slug);  -- ~/projects/ is flat

-- the tree half of the store; the bytes are in object storage (D27)
project_files (
  id            uuid primary key default gen_random_uuid(),
  project_id    uuid not null references projects(id) on delete cascade,
  path          text not null,           -- relative to the project root
  content_hash  text,                    -- sha256; the blob is at
                                         -- {prefix}/blobs/{hh}/{sha256}
  size          bigint not null,
  mtime         timestamptz not null default now(),
  created_at    timestamptz not null default now()
)
create unique index on project_files (project_id, path);
-- No bytes column and no storage_path: a row names a path and the hash of its
-- content. Two projects holding the same content share one blob.

-- a saved copy of a project's tree; the bytes are already immutable
project_snapshots (
  id          uuid primary key default gen_random_uuid(),
  project_id  uuid not null references projects(id) on delete cascade,
  label       text,
  taken_at    timestamptz not null default now()
)
create index on project_snapshots (project_id, taken_at desc);

snapshot_files (
  snapshot_id   uuid not null references project_snapshots(id) on delete cascade,
  path          text not null,
  content_hash  text not null,
  size          bigint not null,
  mtime         timestamptz not null,
  primary key (snapshot_id, path)
)
-- A snapshot costs rows, not bytes. Any future blob GC must walk these as well
-- as project_files, or restoring a snapshot stops working.

-- the user's memory: the curated core and the notes appended to it (D8 amended)
memory_files (
  id            uuid primary key default gen_random_uuid(),
  user_id       uuid not null references users(id) on delete cascade,
  path          text not null,           -- 'MEMORY.md' | 'notes/<stamp>-<rand>.md'
  content_hash  text not null,           -- sha256; the blob holds the bytes
  size          bigint not null,
  body          text not null,           -- the same text, where the FTS query runs
  tsv           tsvector generated always as (to_tsvector('english', body)) stored,
  mtime         timestamptz not null default now(),
  created_at    timestamptz not null default now()
)
create unique index on memory_files (user_id, path);
create index on memory_files using gin (tsv);
-- Keyed by user, not by project: memory is not a project tree. A note is written
-- once; the core is replaced whole under an advisory lock. Whether this may ever
-- be mounted into a sandbox is D30, open — no claim can name it today.

-- what a session may see, and what it locks (D29)
session_claims (
  session_id  uuid not null references sessions(id) on delete cascade,
  project_id  uuid not null references projects(id) on delete cascade,
  subpath     text not null default '/',
  mode        text not null check (mode in ('read','write')),
  primary key (session_id, project_id, subpath)
)
create index on session_claims (project_id);
-- Declared at session creation and fixed for its life. Write claims take
-- project:{id}; read claims mount without a lease and discard their edits.

-- a task IS a session: one row, one id --------------------------------------
sessions (
  id              uuid primary key default gen_random_uuid(),
  user_id         uuid not null references users(id),
  project_id      uuid references projects(id),
  mode            text not null check (mode in ('attended','unattended')),
                                          -- a PHASE, not a type; flips when the
                                          -- human says go
  title           text,
  goal            text,
  status          text not null check (status in ('pending','idle','running',
                                          'awaiting_approval','completed',
                                          'failed','cancelled')),
                                          -- idle = alive, waiting for a human;
                                          -- survives restarts. CHECKed because an
                                          -- out-of-vocabulary write makes
                                          -- transition()'s WHERE status=expected
                                          -- stop matching and strands the session.
  terminal_reason text,                   -- = done.reason, verbatim
  cursor_seq      bigint not null default 0,
  hops_used       int  not null default 0,
  lease_owner     text,                   -- worker id holding this session
  lease_expires   timestamptz,
  created_at      timestamptz not null default now(),
  ended_at        timestamptz
)
-- operator-console seams (cheap now, painful later):
create index on sessions (status, terminal_reason, ended_at);
create index on sessions (user_id, ended_at);
-- the new_sessions_per_hour quota query, which ended_at cannot serve:
create index on sessions (user_id, created_at);

-- the transcript: audit, never pruned ---------------------------------------
session_events (
  seq         bigserial primary key,      -- ONE global counter: ordering +
                                          -- resume cursor + event id. Gaps
                                          -- per session are fine; every read
                                          -- is "after N", never "count".
  session_id  uuid not null references sessions(id) on delete cascade,
  kind        text not null check (kind in ('user','content','reasoning',
                              'tool_call','tool_result','status','todo','budget',
                              'lifecycle','view_transform','done')),
  version     int  not null default 1,    -- readers upcast; rows never rewritten
  payload     jsonb not null,
  ts          timestamptz not null default now()
)
create index on session_events (session_id, seq);

-- oversized tool outputs ----------------------------------------------------
result_blobs (
  ref         uuid primary key default gen_random_uuid(),
  session_id  uuid not null references sessions(id) on delete cascade,
  content     text not null,              -- full text; the event holds a preview
  created_at  timestamptz not null default now()
)

-- operational log: internal, pruned -----------------------------------------
system_events (
  id          bigserial primary key,
  ts          timestamptz not null default now(),
  level       text not null,              -- info warn error
  event       text not null,
  session_id  uuid,                       -- correlates noise back to a session
  user_id     uuid,
  fields      jsonb not null default '{}'
)
create index on system_events (ts);
create index on system_events (session_id);
-- pruned at 30 days; batched best-effort writes; never blocks the loop

-- pauses --------------------------------------------------------------------
approvals (
  id           uuid primary key default gen_random_uuid(),
  session_id   uuid not null references sessions(id) on delete cascade,
  tool_call_id text not null,             -- stays OPEN across the park; the
                                          -- response event closes it
  kind         text not null check (kind in ('approval','ask','call')),
  prompt       text not null,
  answer       text,           -- prose for ask/approval; 'approve'|'decline' for call
  tool_name    text,           -- `call` only: the call that runs if approved, so
  tool_args    jsonb,          -- consent binds to it and not to a description
  consumed_at  timestamptz,    -- the exactly-once latch; consumed-but-open = repair
  created_at   timestamptz not null default now(),
  answered_at  timestamptz
)
create index on approvals (session_id) where answered_at is null;
-- a double-park is impossible, not merely unlikely:
create unique index on approvals (session_id, tool_call_id) where answered_at is null;

-- stateful hands ------------------------------------------------------------
resource_leases (
  resource_key text primary key,          -- 'browser:{user}' | 'project:{id}'
  session_id   uuid not null references sessions(id) on delete cascade,
  acquired_at  timestamptz not null default now(),
  expires_at   timestamptz not null
)

-- The box follows the session, and the row is also its slot in the user's
-- pool: the row count per user is what sandbox.max_concurrent_per_user caps.
session_sandboxes (
  session_id   uuid primary key references sessions(id) on delete cascade,
  user_id      uuid not null references users(id) on delete cascade,
  sandbox_id   text,                      -- e2b handle; null between slot and boot
  created_at   timestamptz not null default now(),
  last_used_at timestamptz not null default now()
)

user_connections (
  user_id       uuid not null references users(id),
  mcp_url       text not null,            -- THE identity. Smithery knows the
                                          -- server by its url; so do we.
  connection_id text not null,            -- issued ONCE at connect time and
                                          -- stored. Never recomputed from a
                                          -- formula, so there is no formula to
                                          -- drift against.
  status        text not null,            -- the fact of connection: what makes
                                          -- "restart = 0 Smithery PUTs" true
  tools_cache   jsonb,
  refreshed_at  timestamptz not null default now(),
  primary key (user_id, mcp_url)
)
-- No server_name column. The key under config.yaml's mcp_servers: ('linear',
-- 'gmail') is an in-process label for logs and the tool registry, rebuilt from
-- config at every startup. Nothing durable references it, so renaming it is free.
-- Display name likewise comes from config at render time.

shared_connections (
  mcp_url      text primary key,         -- no-auth servers have no user, and
  connection_id text not null,           -- user_connections.user_id is NOT NULL.
  status       text not null,            -- Slack's credential is a workspace bot
  tools_cache  jsonb,                    -- token, not a user token.
  refreshed_at timestamptz not null default now()
)

-- what one session may reach ------------------------------------------------
-- Absent row = OFF. The default is ours alone, so a fresh session cannot be the
-- one that puts 164 schemas in a request. Keyed by mcp_url, never by the
-- config label, for the reason stated under the connection tables above.
session_tools (
  session_id   uuid not null references sessions(id) on delete cascade,
  mcp_url      text not null,
  enabled      boolean not null default true,
  updated_at   timestamptz not null default now(),
  primary key (session_id, mcp_url)
)

-- carried over / out of scope -----------------------------------------------
waitlist      -- preserved: real signups from the landing page
repeat_tasks  -- untouched; watching is scrapped for this redesign
```

---

`users` stays `{id, email, created_at}`. `email` comes from the Supabase token
and identifies the account to a human reading the operator console; nothing
sends mail. `slack_user_id`, `username` and `last_seen` are gone with no
successor. A session waiting on a human is surfaced by its `awaiting_approval`
status — the project dot and `GET /attention` — not by an outbound message. Any
push channel would be a new feature with its own spec.

**Config keys are labels, never durable keys.** A connection is identified by its
`mcp_url` and addressed by a stored `connection_id`. Change the url and it is
genuinely a different server, which is exactly what a new row means. Rename or
restyle the config key and nothing at all happens.

Why it looks like this: see `decisions.md` (D13 global seq, D14 one id,
D17 two log tables, D18 leases). Once migration 0 exists, the SQL file is the
truth and this doc goes away.
