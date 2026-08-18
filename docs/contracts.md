# ARKOS Contracts

The stable core. Changes here need owner sign-off; everything else churns freely.
Levels: **backend modules (incl. db) → files → frontend → auth**.
Every contract states its **IO** and what it **replaces/consolidates**.

Settled: ONE agent (`run_turn`). ComputerAgent is deleted; `computer_module`
becomes hands only (sandbox + tools). The browser stays a tool (`browser_task`)
wrapping the third-party `browser_use` specialist — rented brain behind the tool
boundary, not a second agent of ours. Watching/triggers: scrapped (future feature;
`repeat_tasks` table untouched, out of scope). Chat transcripts are sessions too:
they ride `session_events` like any task.

---

## Events vs messages (read this first)

Two representations, one system:

- **Messages** (OpenAI format) are what the MODEL sees. Built in memory per run,
  rebuilt from the log on wake. Never stored as-is.
- **Events** (vocabulary below) are what EVERYONE ELSE sees: stored in the db,
  streamed to the UI, asserted in tests. Store-shape = wire-shape: the object we
  save is the object we push. No translation layer.

Three UI moments, one mechanism: just-opened → read recent events (snapshot);
watching → events arrive over SSE; reconnected → replay after `Last-Event-ID`,
then live. Same query, three moments.

### Event vocabulary (final)

| Event | Why it exists |
|---|---|
| `user{text, source}` | the human's text. `source ∈ human \| system` so the nudge and system injections have a home. Without this kind the fold cannot rebuild a user turn and every attended conversation loses half of itself on reload |
| `content{text}` | model's text, streamed to the screen as written |
| `reasoning{text}` | the model's `<think>` output, streamed the same way. Separate from `content` because the FOLD treats them differently: `content` replays into the message list, `reasoning` never does (Qwen3's template strips thinking from prior turns, so replaying it is both wrong and a context tax). Rendered collapsed |
| `tool_call{id, name, args}` | "→ browser_task" row; `id` pairs it with its result |
| `tool_result{id, ok, error_kind, content, total_chars?, ref?}` | the result card; `content` is view-capped, `ref` pages the full blob |
| `status{label, url?}` | "using the browser…" — UI is never silently frozen. `url` is the ephemeral browser frame stream, announced not stored-for-replay |
| `todo{items}` | feeds the TODO panel |
| `budget{hops_used, hops_max}` | feeds the "hop 7/15" meter |
| `lifecycle{from, to, reason}` | every status transition, appended by `transition()`. Feeds the status pill; without a payload it renders as a generic row |
| `view_transform{rung, dropped_refs[]}` | a context-ladder drop. View-only: the log is never rewritten |
| `done{reason}` | run over. The only writer of a TERMINAL status from `running` (not the only writer of status: create, lease claim, park, respond, and cancel all move it too) |

`done.reason ∈ turn_end | completed | max_hops | wall_clock | model_error |
context_overflow | cancelled | interrupted`.
`turn_end` is NON-terminal: it is the attended "I have said my piece" and the only
trigger for `running -> idle`, leaving `terminal_reason` and `ended_at` NULL.
`interrupted` is written by the startup sweep for a session the process died
underneath.
Transcript invariant: every `tool_call.id` is closed by exactly one `tool_result.id`
before any later assistant/terminal event. Abort paths synthesize
`{ok:false, error_kind:interrupted}`. Events carry a `version` field; readers
upcast, rows are never rewritten.

---

## Backend

### db (fate map)

| Table | Fate |
|---|---|
| `task_events`, `computer_task_events`, `conversation_context` | merge → **`session_events`** `{seq, session_id, kind, version, payload, ts}` — append-only, per-session monotonic `seq` (DB-assigned) |
| `tasks` → **`sessions`** | one id (a task IS a session); `mode` (attended\|unattended), `status` incl. `idle`, `terminal_reason`, `cursor_seq`, lease cols. Full DDL: `schema.md` |
| `task_approvals` → **`approvals`** | kept, renamed to match `schema.md` |
| `user_sandboxes` → **`session_sandboxes`** | rekeyed by session: the box follows the session, and the row is also its slot in the user's pool |
| `users` | rebuilt, zero rows (Task 0c clears every user). Columns per `schema.md`, not today's table |
| `computer_tasks` | dropped with the rest of the old chain |
| `repeat_tasks` | out of scope (watching scrapped) |
| new | **`system_events`** (operational log: batched, best-effort, pruned 30d); **`result_blobs`** `{ref, session_id, content, created_at}` — full oversized tool outputs, events carry preview+ref; **`projects`**, **`project_files`**, **`user_connections`** `{user_id, mcp_url, connection_id, status, tools_cache, refreshed_at}` — keyed by `(user_id, mcp_url)`; `connection_id` is minted BY US (Smithery accepts a path segment we choose), stored, never recomputed. No `server_name`: config keys are in-process labels, not durable keys. No credentials (they stay behind Smithery). **Write the row first:** mint a UUID, INSERT `status='pending'`, PUT to Smithery, then UPDATE from the response. A crash between mint and PUT leaves a pending row whose id is reused on retry, so the id survives every failure; minting after the PUT strands the old connection holding the user's live OAuth grant, which is the drift D24 exists to kill, just moved to crash-time. **`shared_connections`** `{mcp_url pk, connection_id, status, tools_cache, refreshed_at}` — same rule, no user column, because no-auth servers (Slack's workspace bot token) have no user and `user_connections.user_id` is NOT NULL. |

### harness_module (owns control plane — lifecycle sits NEXT TO the agent, never inside it)

**`lifecycle.py`** — consolidates `task_store.set_task_status` /
`mark_task_completed` / `mark_task_failed` / PATCH-status route.
```
transition(session_id, expected: Status, new: Status, reason: str,
           mode: Mode | None = None) -> bool
# Status: pending idle running awaiting_approval completed failed cancelled
# Mode:   attended | unattended
# conditional UPDATE WHERE status=expected; False = lost race; cancel wins.
# Never LLM-gated. Every transition is also appended as a `lifecycle` event.
# mode moves in the SAME conditional UPDATE as status. A mode flip without a
# status change is illegal, and both flips (approve, unattended done) already
# change status. This closes the window where a session is unattended for budget
# accounting but still recorded attended.
```

Every transition, and its ONE trigger:

| from | to | trigger |
|---|---|---|
| — | `pending` | `POST /sessions` |
| `pending` | `running` | runner claims the lease |
| `running` | `idle` | `done{turn_end}` — attended, model stopped calling tools |
| `idle` | `running` | human sends a message |
| `idle` | `running` | `POST /sessions/{id}/approve` — also flips `mode` to unattended |
| `running` | `awaiting_approval` | park tool (`request_approval` / `ask`) |
| `awaiting_approval` | `running` | respond endpoint wakes it |
| `running` | `completed` | `done{completed}` — also flips `mode` back to attended |
| `running` | `failed` | `done{max_hops \| wall_clock \| model_error \| context_overflow \| interrupted}` |
| `running` | `cancelled` | cancel signals the loop → exits via `done{cancelled}` |
| `pending`/`idle`/`awaiting_approval` | `cancelled` | cancel written directly (no loop to signal) |
| `completed`/`failed`/`cancelled` | `running` | **human restarts it**, or sends a message on a terminal session. Clears `terminal_reason` and `ended_at` |

**Nothing auto-resumes.** A session the process died underneath becomes
`failed{interrupted}` at startup (the sweep synthesizes the `done` event first, so
the transcript says why it stopped). It is not requeued: a blind retry re-executes
an unclosed side effect. Restarting is a human act, and it uses the reopen row
above, so "the session you keep coming back to" (D5) and "a human decides whether
to restart" (D16) are the same mechanism, not two.

Reopening makes `terminal` a current-state cache rather than history. That is
fine: the `done{...}` event stays in the log forever (D3), which is where history
actually lives.

`transition()` is the single writer; `done{reason}` owns terminal-from-`running`.
`terminal_reason` = `done.reason` verbatim — no second vocabulary; `status` is its
coarse bucket (four failure reasons all map to `failed`, reason preserved).

**`idle` is a real state.** After a session replies and before the human types it
is alive and waiting — not running, not completed, not parked, not pending. An
attended session may sit `idle` forever; terminals come from unattended runs.
Guard rail: **every state must change what a person sees.** A proposed state that
changes nothing visible means control flow is creeping into the status column.

**`session_log.py`** — consolidates `log_event`, `log_computer_event`,
`conversation_context` writes.
```
# The fold (runner.wake) maps kinds to messages. Three answers, which is why
# there are three kinds:
#   user      -> role:"user"
#   content   -> role:"assistant"
#   reasoning -> DROPPED. Never replayed into context.
#   tool_call / tool_result -> the paired assistant/tool messages
#   status / todo / budget / lifecycle / view_transform -> UI only, not context
append(session_id, event) -> seq        # seq = ONE global BIGSERIAL: ordering,
                                        # resume cursor, event id. Several things
                                        # append (loop, api, lifecycle) and an
                                        # idle session has no loop to funnel
                                        # through, so per-session numbering would
                                        # need a lock. Gaps are fine: every read
                                        # is "after N", never "count".
                                        # Enforces the transcript invariant.
# append() takes pg_advisory_xact_lock(session_id) before inserting. BIGSERIAL
# hands out values BEFORE commit, so without it an api txn holding seq 100 can
# commit after the loop's 101, and an SSE reader that already sent 101 never
# emits 100: a silently dropped event on a live stream. The lock makes commit
# order equal seq order WITHIN a session. D13 is untouched; it argued against
# per-session counters, not per-session locking.
# NOT covered: the user-scoped grid feed (G27) spans sessions, so it needs a
# hold-back window when it is built.
get_events(session_id, after_seq=0, limit=500) -> list[Event]
save_blob(session_id, content) -> ref
read_blob(ref, offset=0, limit=2000) -> str
# append() failure is LOUD: the run halts with a terminal event rather than
# continuing unrecorded. Nothing executes off the record.
```

**View & context (invariant).** Context-view transforms (dropping/demoting old
tool results) are recorded as `view_transform` events; the log itself is NEVER
rewritten; context assembly is deterministic given (log, config) — same log,
same view, byte-identical (prompt cache + replay). Ladder mechanics live in the
spec; this invariant is the contract.

**Rung 1 clears by REF, not by tool** (owner sign-off 2026-08-17, replacing the
`clearable_tools` whitelist). Clear the oldest results that hold a blob ref,
regardless of producer; a result without a ref is never cleared. The whitelist
promised re-derivability — re-run `grep` and get the same answer — which was
already false for `browser_task`, because the web moved. A ref promises exact
recoverability instead, and exact recoverability is what the fold actually
needs. It also reaches the two largest producers of context pressure,
`browser_task` and every `mcp_*` result, which the whitelist excluded by
construction. Everything else is unchanged: view-only, the log is never
rewritten, every drop appends a `view_transform` carrying the dropped refs, and
replay stays deterministic.

**Logging — two tables, split by AUDIENCE.**
- **`session_events`** = the transcript. What a human should see about the
  agent's behaviour: messages, tool calls and results, lifecycle, todos, budgets.
  User-visible (Looking Glass renders it), audit-grade, never pruned. A failed
  append HALTS the run — nothing executes off the record.
- **`system_events`** = operational. Retries, lease churn, timing internals, DB
  hiccups, startup. Batched best-effort writes that never block; pruned at 30
  days; a failed write is a lost diagnostic, never a stopped run. Carries an
  optional `session_id` so internal noise joins back to a session without
  polluting the transcript.
- **stderr** covers exactly one case: the database is unreachable, which cannot
  be written to the database. A small `logging` module owns the JSON formatter,
  the session-id context var, and secret redaction. `print()` is banned.
- **Metrics worth watching** are already events, so they are SQL: malformed
  tool-call rate (the load-bearing assumption; drifts with prompts/models) and
  terminal-reason distribution (are budgets too tight, is the nudge working).
  TTFT is a CI test, not production telemetry.

**Config is the source of truth (no magic numbers).** Every limit, budget, and
threshold is read from config at call time. Nothing is hardcoded in the loop,
the client, or the ladder; a value that exists in config and is ignored by the
code is a bug (see `short_term_turns` in violations).

```yaml
llm:
  context_window: 40960        # total in+out the served model accepts
  max_tokens: 8192             # output reserve  -> input budget = window - reserve
  timeout_s: 90                # per model call
  max_retries: 3               # the ONE retry layer
context:
  recovery_threshold: 0.8      # rung 0 trips at this fraction of input budget
  chars_per_token: 4           # estimator ratio; there is no tokenizer for the served model
  turns: 50                    # short-term transcript window (today: ignored)
budgets:                       # keyed by sessions.mode — one vocabulary, nothing to map
  attended:   {max_hops: 6,  wall_clock_s: 300}
  unattended: {max_hops: 15, wall_clock_s: 1800}
  per_tool_attempts: 3
tools:
  result_view_cap_chars: 4000  # over this -> blob + ref
sandbox:                       # promoted out of the deleted computer_agent: block.
  template: base               # deleting that block wholesale would break the LIVE
  timeout_seconds: 300         # sandbox, since Task 8 (which relocates it) is the
  idle_timeout_seconds: 900    # last deletion. Only computer_agent.llm dies.
app:
  public_url: "https://..."    # the one origin /app and the API are served from
quotas:
  max_unattended_sessions: 5   # per user; only UNATTENDED runs count — an idle
                               # or attended conversation consumes no worker
  new_sessions_per_hour: 20    # per user; sliding window
  upload_max_mb: 25            # per file, POST /projects/{id}/files
```

This block is the redesign DELTA, not the whole file. It does not restate
`llm.base_url`, `llm.model_name`, `database`, `smithery`, or `mcp_servers`.

Quotas are enforced in `api.py` at command time, before any state change.
`POST /sessions` checks `new_sessions_per_hour` only; `POST /sessions/{id}/approve`
checks `max_unattended_sessions`, because every session is created attended (D5) so
a check at create time always sees zero unattended load and never fires. The upload
route checks size. Over
quota returns the standard `{code, message, retryable}` error. The
count-then-act race on the concurrency check is real and unhandled in v1.

`context_window` must match how the SGLang server was actually launched
(`--context-length`), not the model's theoretical max. Budgets differ per mode,
which is why they are passed INTO `run_turn` rather than read inside it.

**One predicate, derived not restated: is a human waiting?** `sessions.mode` is
the only input. Budgets key off it directly (`Budgets.load(mode)`), and the
client's `source` derives from it in one line — `background` when unattended, so
an unattended run yields the GPU slot on overload instead of queueing for it.
`source` stays `interactive | background` because `model_module` must not know
what a session is; that is a layer boundary, not a second vocabulary.

**Concurrency model (the scalability law).** One asyncio event loop; every
session is a coroutine, never an OS thread. NO blocking I/O in any async path —
one sync DB call freezes every user. **asyncpg**, one pool created at startup — the harness is new code, so there is
nothing to migrate (`$1` placeholders; `statement_cache_size=0` against
Supabase's transaction pooler). The GPU (SGLang continuous batching) is the shared
bottleneck; `client.py` is the seam for adding replicas. Isolation between
sessions is data-level (own log, own folded context per `generate` call) —
crosstalk is structurally impossible, not merely prevented.

**`runner.py`** — replaces `task_runner.py` and `computer_module/runner.py`.
```
start(task_id) -> None      # lease-claim, then wake
wake(task_id) -> None       # fold events from cursor_seq -> messages,
                            # run run_turn, translate events -> append()+transition()
cancel(task_id) -> bool
# verify-on-wake: dangling tool_call -> interrupted result surfaced to the model
# ("outcome unknown, verify before retrying"); never silently re-executed.
```

**`api.py`** — replaces the 4x status-polling endpoints, approvals poll, bespoke
chat plumbing. One error shape everywhere: `{code, message, retryable}`.

| Endpoint | In | Out |
|---|---|---|
| `POST /auth/session` | Supabase JWT as `Authorization: Bearer` | 204 + `Set-Cookie`. Verifies it once, upserts `sub` → `users`. The ONLY endpoint that reads a bearer token |
| `DELETE /auth/session` | — | 204, cookie cleared |
| `GET /auth/me` | — | `{user_id, email}` |
| `GET /sessions/{id}` | — | `{title, project_id, status, hops_used/max, recent_events[]}` |
| `GET /sessions/{id}/events` | `Last-Event-ID?` | SSE of events, `id:<seq>` each |
| `POST /sessions` | `{goal, steps?, project_id?}` | `{session_id, project_id}` (new project unless given; `steps` seed the todo list) |
| `POST /sessions/{id}/messages` | `{text}` | 202 — appended as user event, read next hop |
| `POST /sessions/{id}/cancel` | — | 202 |
| `POST /approvals/{id}/respond` | `{answer}` | 202 — appends event, wakes at cursor |
| `GET /projects` | — | `[{id, title, status_rollup, updated_at}]` |
| `GET /attention` | `project_id?` | pending approvals/asks (same query, any scope) |
| `POST /projects/{id}/files` | multipart | `{file_id, name, size}` |
| `GET /results/{ref}` | `offset&limit` | blob slice (ownership-checked) |
| `GET /sessions/{id}/browser/frames` | — | SSE JPEG side-channel, keyed (user, session), announced by a `status` event, rendered in the canvas panel (not a corner overlay) |
| `POST /sessions/{id}/approve` | — | 202 — attended → **unattended**; the run begins |
| `GET /connections` | — | `[{server, name, mcp_url, requires_auth, status, tool_count, refreshed_at}]` — `mcp_servers:` joined to `user_connections` + `shared_connections` |
| `POST /connections/{server}/connect` | — | `{setup_url}` — mints the id, writes the `pending` row, PUTs to Smithery. Idempotent: reconnect reuses the stored id |
| `DELETE /connections/{server}` | — | 204 |
| `GET /oauth/callback/{server}` | Smithery's redirect | HTML that `postMessage`s the opener and closes. Identity from the **cookie**, never a query param |

The callback never *blocks* on Smithery: it authenticates by cookie, responds,
and fires one verification after the response. Verification is idempotent
`connect()`, run again on any read of `GET /connections`; a row not yet connected
keeps its `connection_id` and `tools_cache` (D24). The callback firing is the
proof OAuth completed, so it is the trigger — read-repair alone would strand a
connection whose popup closed early, since dispatch never re-verifies and
revalidation skips unconnected rows.

**Auth on every endpoint above: the session cookie.** httpOnly + Secure +
SameSite=Lax, set by us after verifying Supabase's JWT once. The browser attaches
it automatically, so `EventSource` needs no token in the URL and `Last-Event-ID`
reconnect works out of the box. Mutations additionally check `Origin`. No stream
tokens.

**Same-site is a deployment REQUIREMENT, not an aside.** `/app` and the API are
served from one origin behind Caddy. SameSite=Lax sends the cookie only on
top-level GET navigations, so if they were cross-site every `EventSource` (a
subresource request) would arrive with no cookie and both SSE streams would 401 on
every connect, silently killing the entire push architecture while
`test_cookie_session` still passed. CORS is therefore decoration: `allow_origins`
is pinned to the single app origin as defence-in-depth, not as the thing making it
work. A cross-site frontend would need SameSite=None plus explicit CSRF tokens.
We are not doing that. `EventSource` sets `withCredentials: true` regardless,
because it costs nothing and fails closed.

### agent_module (pure brain — the ONLY agent)

**`loop.py`** — replaces `Agent.step`/`step_stream`/`choose_transition`, ALL of
`state_module/`, and `ComputerAgent.run` (generalized from it; the class is deleted).
```
run_turn(messages: list[Message],
         tools: list[ToolSpec],
         budgets: {max_hops, per_tool_attempts, wall_clock_s, model_retries},
         mode: "attended"|"unattended",
         *, dispatch: (name, args) -> ResultEnvelope,
         hops_used: int = 0,
         options: dict|None = None) -> AsyncIterator[Event]
# dispatch is INJECTED and required. Importing tool_module here would give the
# loop MCP sessions, the sandbox and the browser, and "pure brain" would be false.
# hops_used is CALLER-SUPPLIED for the same reason: hops are cumulative across a
# resume and counted from the log, which the loop cannot read.
# No `source` param: unattended IS the background source, derived from mode.
# Never touches Postgres. Only caller of the model.
# Hops cumulative across resume (counted from the log); wall clock bounds the
# WHOLE hop, model plus tools, not just the gap between hops.
# Per-tool attempts cap 3, then the failure stands. The count is CONSECUTIVE
# FAILURES, not calls, and a success clears the streak: a tool that recovers is
# not punished for an earlier bad patch. In-flight calls count against the cap,
# so a parallel fan-out of five cannot outrun a cap of two.
# Malformed tool args buy ONE repair round trip per turn, NOT charged as a hop:
# the hop produced no usable work, and charging it would let bad JSON eat budget
# the task needs.
# Completion is set from the finish_task RESULT, never from the call. A
# finish_task that errors, has bad args, or is capped does NOT complete the run.
# model_retry semantics: the client retries WITHIN one generate() call; the loop
# re-attempts the HOP at most 3 consecutive times (with backoff) after a
# retryable exhaustion, then done{model_error}. Two bounded layers, not nested
# unbounded ones.
# TERMINATION (one rule): a run ends when the model stops calling tools AND the
#   exit is safe. attended -> safe always (the human is the continuation);
#   unattended -> safe only if finish_task was called. Bare text when unattended
#   appends, gets ONE low-budget nudge near the cap, and loops.
# Errors are model input, not control flow; only cancellation propagates.
# Streaming is structural: client deltas re-yielded as content events
# immediately; no accumulation step exists (pinned by test).
```

**`events.py`** — formalizes `step_stream`'s ad-hoc dicts into the vocabulary above.
Consumed by: session_log (store), SSE (wire), runner (lifecycle), tests, frontend.

### model_module (the model lives here)

**`client.py`** — replaces `ArkModelNew.py:ArkModelLink` AND
`computer_module/model.py:ToolCallingModel` (two clients → one); shrinks `llm_json.py`.
```
generate(messages, tools|None, source, options: dict|None) -> AsyncIterator[Delta]
# Delta = text_delta | reasoning_delta | tool_call_delta | finish{usage}
# reasoning_delta carries SGLang's reasoning_content (--reasoning-parser qwen3).
# The loop re-yields it as a `reasoning` event, streamed like `content`: Qwen3
# reasons for hundreds of tokens before answering, and dropping it means a frozen
# screen during the part of the turn that takes longest.
# options is a passthrough of per-call model params (temperature for tool turns,
# chat_template_kwargs to disable thinking). Written ONLY from config, never from
# model output.
# One cached AsyncOpenAI(timeout=llm.timeout_s, max_retries=0). The ONLY retry layer:
# timeout|connect|rate_limit|server_error retry ≤3 backoff+jitter;
# bad_request|auth fail fast; background source: no overload retries.
# Raises ModelError{kind, retryable} and nothing else.
# Not promised: which model/endpoint (Qwen3-8B on SGLang today; swap is config).
```

### tool_module (ALL hands — MCP, sandbox, browser)

**`envelope.py`** — standardizes what `state_tool.py` (raises),
`ComputerAgent._call_mcp` (error strings), and `tools.py:dispatch` (already right)
do inconsistently.
```
execute(name, args, ctx) -> ResultEnvelope     # never raises except cancellation
ResultEnvelope{ok, content, error_kind: none|invalid_args|not_found|auth_required|
               timeout|upstream_error|interrupted, retryable, ref?}
ToolSpec{name, description, input_schema, readonly, requires_approval}
# Failure content is written FOR the model. auth_required carries the setup URL.
# readonly drives concurrency (parallel reads, serial writes);
# requires_approval checked at execute time.
```

**`manifest.py`** — replaces `_make_agent` tool assembly,
`format_tools_for_system_prompt`, `create_tool_option_class`; deletes `scoped.py`.
```
manifest(user_id) -> list[ToolSpec]
# ONE manifest — there are no session types (D5). Every session may reach for
# every hand; lazy provisioning makes an unused tool free.
#   control : finish_task · request_approval · ask · todo_write · read_result
#   world   : list_projects · get_project · list_sessions · get_session · list_files
#   sandbox : run_command · read_file · write_file · edit_file · list_dir · grep · glob
#   browser : browser_task
#   MCP     : whatever the user connected, exposed as mcp_{name}
# Ours (~20) are authored here and always loaded. MCP tools are namespaced
# mcp_* (stripped on dispatch, so a remote read_file cannot shadow ours) and are
# the ONLY ones deferred when the schema budget is tight (load_tools fetches).
```

**Tool layout — folder per tool.** Each tool owns its schema, its description
(own file, a function — it is the model's whole manual for that tool), its
`validate` (preconditions before work, e.g. read-before-edit), its `call`, and
its `render` (its row in Looking Glass). `registry.py` auto-discovers; adding a
tool is adding a folder.

**kept:** `smithery.py` (transport under envelope; one ClientSession, TTL,
vault+proxy — creds never in brain or sandbox); `tool_module/browser/`
(browser_tool, browser_stream, browser_actions — leashed); `tool_module/sandbox/`
(e2b manager + sandbox toolset, moved from computer_module; lazy provision on
first sandbox-tool call).

Browser contract, precisely: progress = ordinary `status{label}` events in the
session stream (wired via browser_use's `register_new_step_callback`); video
frames are NOT events — they stay an ephemeral side-channel, keyed by
`(user_id, task_id)`, ownership-checked like every endpoint, and announced by a
`status` event carrying the stream URL so the UI mounts the pane from the event
stream. The final `tool_result` is a real envelope built from the run history
(`ok` from `is_successful()`/`has_errors()`, error summary on failure, `ref` to a
compact run record) — never a bare string. Budgets enforced via the graceful
stop callback (partial results on timeout), hard `wait_for` only as backstop.
Any silently dropped `register_*` kwarg on a browser_use version bump must log
at WARNING — progress events silently disappearing is the opacity this outlaws.

Resource contract, generally: **serialize a resource iff it is BOTH shared
across a user's concurrent sessions AND carries mutable state between calls.**
That test, not a list, decides future resources.

| Resource | Shared? | Stateful? | Model |
|---|---|---|---|
| sandbox | no (one box per session) | no (the disk is a cache, D27) | **capped**, not leased: `sandbox.max_concurrent_per_user` |
| project subtree (`project:{id}`) | yes | yes (the tree it writes) | **leased**, one per write claim |
| browser (`browser:{user}`) | yes | yes (profile, cookies, logins) | **leased** |
| MCP via Smithery | yes | no (stateless per call) | no lease, runs free |
| session log | no (per session) | — | serialized by the appender, a different race |

**The sandbox is capacity, not a lease.** A box belongs to one session and is
destroyed once that session's flush lands, so there is nothing to serialize:
overlapping writes are already ordered by the `project:{id}` claims, and a box is
never compared to another box, only to the store subtrees its session claimed.
What `sandbox.max_concurrent_per_user` protects is spend, and a session over the
cap waits exactly as a lease waiter does. The slot is a row in
`session_sandboxes`, taken before the box boots and dropped when it is reaped, so
capacity cannot outlive the box that used it. A session's box is reaped ONLY
after its flush lands.

**A lease wait is not a park.** A session waiting on a held lease stays `running`
and emits `status{label:"waiting for a computer"}`, which is exactly what `status`
exists for. There is no `waiting` status and there must not be one: borrowing
`awaiting_approval` would turn the project dot ochre and put a phantom row in
`/attention`. No hops burn, because no model call happens. The wall clock excludes
lease-wait time, using the same active-segment accounting park already needs. On
lease-wait timeout the tool returns
`ResultEnvelope{ok:false, error_kind:timeout, retryable:true}` and the model routes
around it, per "errors are model input, not control flow".

A lease is held for the **whole session, not per tool call** — per-call leasing
lets session B interleave session A's half-finished write, which is the exact
corruption the lease exists to prevent. Released on terminal or on park (a parked
session is not acting), re-acquired on resume. Contended, a session parks as
"waiting for {resource}" and wakes when free; a lease-wait timeout parks rather
than fails.

**State persists, runtime is cattle.** The store and the browser profile
directory are durable; the running instance is not. Lazily booted on first use,
torn down at the end of the session, rebuilt from the store next time. "Persistent
browser" means the logins survive, not that a browser stays warm.

### store — the agent's filesystem

Bytes in object storage we own, the tree in Postgres, the sandbox disk a cache
(D27). `harness_module/store.py` owns it, because the store is the harness's and
never the sandbox's (D28).

```
put_blob(content) -> sha256          # content-addressed, immutable, write-once
get_blob(sha256) -> bytes
read_tree(project_id) -> [TreeEntry{path, content_hash, size, mtime}]
commit_tree(project_id, entries)     # blobs FIRST, rows LAST, in one transaction
diff_tree(a, b) -> paths that differ by hash
append_note(user_id, text) -> path   # one file per note
```

**Layout is fixed.** `{prefix}/blobs/{hh}/{sha256}` for bytes;
`{user}/memory/{MEMORY.md, notes/}` and `{user}/projects/{project-id}/` for the
tree. Keyed by project id so a rename moves nothing; mounted under the project
slug, because the model reads paths as context and a mounted name should be
human.

**Blobs first, rows last.** A commit uploads every missing blob before it flips
tree rows, and flips them in one transaction. A crash between the two leaves the
previous tree intact and whole: an orphan blob costs storage, a tree row
pointing at a blob that is not there costs a file. Commits are idempotent, so a
retry after a partial upload is safe.

**The cache fills when the session takes its box and empties before it gives it
back.** Taking a box materializes the session's claimed subtrees; release, park
and terminal flush them back and then reap the box. A flush that fails is loud in
`system_events`, retried on the reaper's backoff, and **the sandbox is not killed
until the flush lands** — the reap is downstream of the commit, so the cache is
never destroyed while it holds the only copy of an edit.

**Memory never mounts.** `{user}/memory/` is excluded structurally, not by a
filter: project subtrees are the only mountable thing, and the mount path has no
branch that can reach memory. The sandbox executes model-authored code, and
memory is the most sensitive distillate in the system. Sessions may only
`append_note`; `MEMORY.md` is rewritten by the compaction job alone.

**Claims are the unit of conflict** (D29). A session declares
`(project_id, subpath, mode)` at creation. The set is the sole source of which
leases it takes (`project:{id}` for each write claim) and what appears in its
sandbox. Nothing unclaimed is mounted. A read claim materializes without a lease
and its flush is a no-op, with discarded edits logged.

**No store credentials enter the sandbox.** Bytes flow store → harness → e2b
API, never sandbox → store.

### (computer_module: dissolved)

With its agent deleted, `computer_module` had no reason to be a module. The
sandbox manager and toolset move to `tool_module/sandbox/`; `agent.py`,
`model.py`, `runner.py`, `computer_router.py` are deleted/folded into the
unified task path. Backend module map is now role-per-module: `harness_module`
(control plane) · `agent_module` (brain) · `model_module` (model) ·
`tool_module` (all hands).

### memory_module — REMOVED for now; reimplementation TBD

Long-term memory is cut from the redesign scope. mem0 is dropped entirely
(its embedder service, extraction LLM calls, config, and deps go with it).
Short-term context is fully covered by `session_log` (the transcript IS the
memory within a session). No memory tools ship in v1 manifests.

What this deletes: `retrieve_long_memory` auto-injection (embed+search per turn),
background `add_memory` extraction, the mem0 `memories` collection, `memory_module`
itself save for whatever thin glue `session_log` absorbs.

When reimplemented (TBD, own spec), the direction already workshopped: tools over
a schema we own (`save_memory` / `search_memory` / `pinned` standing rules;
explicit writes, no hot-path retrieval; FTS before vectors). Nothing in the
contracts above depends on memory existing — the seam is just two more ToolSpecs
in the manifest when it returns.

---

## Frontend

Consumes ONLY the endpoint table + event vocabulary. Types in a hand-maintained
`types.ts` mirroring `events.py` (CI check; codegen later if the surface grows).

- No polling. Per open surface: one snapshot request + one `EventSource`
  (cookie-authenticated — no token in the URL); reconnect via `Last-Event-ID`;
  apply events as deltas.
- Commands are optimistic: apply locally, reconcile on the event.
- Renderer switches on `event.type`; unknown types render as a generic row, never crash.
- Product is a subsection: Looking Glass = project-bubble grid (status dot =
  lifecycle rollup: green running, ochre awaiting_approval — approvals AND unanswered asks, red failed, neutral grey idle, gray completed/cancelled)
  → detail stream. Command Center / project attention / in-window pause are the
  same `GET /attention` at user/project/task scope; resolving anywhere writes the
  same respond event. Sessions never return values (no join).

---

## Auth (summary; own spec when built)

Today's hole is token ISSUANCE: `demo-login` hands a valid token to anyone who
types a username.

**Supabase Auth (GoTrue) + a session cookie.** Frontend logs in with supabase-js;
we verify that JWT **once** and set our own httpOnly + Secure + SameSite=Lax
session cookie. The browser then attaches it automatically, which is why
`EventSource` needs no stream tokens and `Last-Event-ID` reconnect works free,
and why XSS cannot read the session. `sub` → users row. DELETE `demo-login`,
DELETE `ARK_DEMO_MODE`, pin CORS to the real origin with credentials allowed,
and origin-check mutations (the CSRF cost of cookies). Non-browser clients (CLI,
mobile) would need bearer alongside — a small addition when one appears.

Invariants the rest of the system relies on: every endpoint scoped to the token's
user; ownership checked on sessions, projects, files, and blob refs; per-user cap
on concurrently running sessions. None of this exists in prod until it does —
`/app` stays down publicly until then.

---

## Conformance tests (definition of done for the spine)

| Test | Pins |
|---|---|
| `test_transcript_invariant` | every tool_call.id closed once; interrupted synthesis on abort |
| `test_lifecycle_transitions` | ALLOWED map only; cancel wins a live race |
| `test_streaming_first_token` | first content event before the mocked model finishes |
| `test_retry_budget_bounded` | ≤3 attempts, bounded wall clock; background no-retry |
| `test_append_ordering_under_concurrency` | concurrent api and loop appends; an SSE reader never skips a seq |
| `test_resume_verify_on_wake` | dangling call → interrupted; no silent re-execution |
| `test_event_replay_deterministic` | same log ⇒ identical context assembly, ladder included: a log over the input budget folds to a view under it, and folds byte-identically twice |
| `test_authz_scoping` | user A cannot read/steer user B's sessions, refs, files |

---

## Current violations (bugs the contracts outlaw — fix targets, not contracts)

| Violation | Breaks | Where |
|---|---|---|
| browser frame queue keyed by user only; concurrent tasks clobber each other | (user, task) stream keying | `tool_module/browser_stream.py:39,74` |
| `/v1/browser/stream` trusts client-supplied user_id, no auth | every-endpoint ownership check | `harness_module/browser_routes.py:31-34` |
| browser returns bare string; failure and empty are both `""` | tool_result envelope | `tool_module/browser_tool.py:499` |
| zero step callbacks wired; 3-min silent tool call | progress-as-status-events | `tool_module/browser_tool.py:443-463` |
| ~~`run.sh` launches with NO parser flags; `llm.model_name` is `"tgi"`~~ FIXED in Task 0a(i) | one authoritative model: **Qwen3-8B**, `--tool-call-parser qwen25`, `--reasoning-parser qwen3` | `config_module/config.yaml`, `model_module/run.sh` |
| ~~`logging_module` mandated by CLAUDE.md does not exist; 35 `print()`s in prod~~ RESOLVED 2026-08-13 | two-logs contract above | the 35 were 27 in a misplaced test, 7 in `db/migrate.py` (a CLI, where print is correct) and 1 in a build script. Zero in production paths. `logging_module` itself is still unbuilt (D17) |

**Resolved by deletion, 2026-08-13 (Tasks 7+8).** These were tracked rather than
fixed because their files were scheduled corpses; the files are now gone, so the
violations are gone with them. Kept as a record of what the redesign was for, not
as open items: `demo-login` minting credential-free tokens (`users.py`); task
status written from 4+ call sites with no legality check (`task_store.py`,
`tasks.py`); runaway tasks marked **completed** via the `completion_signal`
fallback (`task_runner.py:213`); fake streaming, full completion then per-char
replay (`agent_module/agent.py:494`); two model clients with no timeout and
nested retries (`ArkModelNew.py:83`, `computer_module/model.py:46`);
`ArkModelLink` built with no `model_name` so the live path ignored
`llm.model_name` (`app.py:67`); mem0 hardcoding Qwen2.5-7B against a port serving
Qwen3-8B (`memory_module/memory.py:58`); context overflow classified
`bad_request` into terminal death (`ArkModelNew.py:147`); `short_term_turns`
config never read; and the docstrings routing implementers to the superseded
specs.

Settled 2026-08-13: `base_module` is renamed **`harness_module`** — the four
control-plane files (`lifecycle` · `session_log` · `runner` · `api`) land there
in Tasks 4-5.

Settled 2026-08-16: **`version` is per-event**, not per-session. A session
outlives a deploy and rows are never rewritten, so one transcript legally holds
events of mixed versions; a per-session version cannot describe that without a
rewrite the append-only rule forbids. Already the shipped shape —
`events.py:47` stamps every event and `parse_event` upcasts, migration 0:127 is
a per-row column.

Settled 2026-08-16: **`mcp_*` tools require approval by default.** A remote
server does not tell us whether a tool mutates. That unknown already resolves
conservatively for concurrency (`readonly=False`, so writes never batch), and it
now resolves the same way for consent, rather than being conservative about
latency and permissive about consequences. Attended runs hardly notice; an
unattended one is fifteen hops of unsupervised Gmail, Slack, GitHub and Linear
otherwise, which is the tool class this whole approval surface exists for. Waive
per server with `mcp_servers.<label>.auto_approve` — `true`, or a list of tool
names. Loosening a default is a config edit; tightening one after unattended
runs have been sending mail is an incident.
