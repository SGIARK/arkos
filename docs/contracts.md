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

**Every appended event is published, and published after its transaction
commits.** An append that skips the publish is a UI that only updates when
someone reconnects; a publish before the commit hands a subscriber a seq that a
`Last-Event-ID` reader cannot fetch yet. Where an append shares a transaction
with a state change — `lifecycle.transition` is the one — the publish belongs to
that function, after the block, not to each caller who must remember it. And the
replay is paged until a read comes back short: one page is not the backlog, and
a reader rejoining a long session is exactly the case that exceeds it.

### Event vocabulary (final)

| Event | Why it exists |
|---|---|
| `user{text, source}` | the human's text. `source ∈ human \| system` so the nudge and system injections have a home. Without this kind the fold cannot rebuild a user turn and every attended conversation loses half of itself on reload. **`source: system` is NEVER RENDERED** — the play button's handoff, the plan-reply instruction, the continuation and the finish nudge are the harness talking to the model, and showing them puts what we send in among the words the human said |
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

`done.reason ∈ turn_end | stopped | completed | max_hops | wall_clock |
stalled_progress | model_error | internal_error | context_overflow | cancelled |
interrupted`. `turn_end` and `stopped` are NON-terminal and are the two triggers
for `running -> idle`, both leaving `terminal_reason` and `ended_at` NULL. They
differ only in who ended the hop: `turn_end` is the attended "I have said my
piece", `stopped` is a human pressing Stop.
`interrupted` is written by the startup sweep for a session the process died
underneath.

**Each failure reason names ONE thing, and the pill is read by people.**
`model_error` is a real `ModelError` after both retry layers, and nothing else.
`stalled_progress` is the model producing no work: an empty reply, or a third
consecutive bare-text hop in an unattended run after the continuation and the
finish nudge were both injected. `internal_error` is the harness — a setup
failure raised in `_drive`'s catch-all, or a loop that ended with no `done`.
Before 11.8.5 all three were `model_error`, so a Postgres blip, an OpenAI outage
and a model that said nothing were indistinguishable on the status pill, and the
2026-08-20 Marketplace run reported `failed{model_error}` though nothing errored.
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
| `task_approvals` → **`approvals`** | kept, renamed to match `schema.md`. THREE kinds, and they are not interchangeable: `ask`/`approval` are prose the model reads back; `call` is a gated tool call carrying `(tool_name, tool_args)`; `plan` is a proposed plan carrying the plan itself in `tool_args`. A stopped run has NO row — 11.8.6 gave it kind `resume` and 11.8.7 deleted it, because a stop is not a question and a held run waits on no consent |
| `user_sandboxes` → **`session_sandboxes`** | rekeyed by session: the box follows the session, and the row is also its slot in the user's pool |
| `users` | rebuilt, zero rows (Task 0c clears every user). Columns per `schema.md`, not today's table |
| `computer_tasks` | dropped with the rest of the old chain |
| `repeat_tasks` | out of scope (watching scrapped) |
| new | **`system_events`** (operational log: batched, best-effort, pruned 30d); **`result_blobs`** `{ref, session_id, content, created_at}` — full oversized tool outputs, events carry preview+ref; **`projects`**, **`files`** `{user_id, path, content_hash, size, mtime}` unique on `(user_id, path)` — ONE flat store per user, replacing `project_files` (11.9; `project_files` is DROPPED, and a folder is `split_part(path,'/',1)`, derived and held by no table); **`project_folders`** `{project_id, folder}` — the links, so deleting a project deletes its links and no files; **`user_connections`** `{user_id, server, status, refreshed_at}` — keyed by `(user_id, server)`, where `server` is the ARCADE APP PREFIX (`Gmail`, `MicrosoftOutlookMail`), which is what every one of that app's tool names is prefixed with (11.10). Not the gateway url: one gateway serves every app, so a url cannot tell two servers apart, and the gateway slug is infrastructure that can be recreated while the grants behind it — keyed Arcade-side by user id — survive untouched. Still not a config key: a `mcp_servers:` label is in-process and nothing durable may reference it; an Arcade prefix is the vendor's own name. No credentials (they stay at Arcade). No `connection_id` and no `tools_cache`: there is no connection object at Arcade to name, and the gateway's tool roster is one roster rather than a per-user fact, so it is cached in-process on the TTL. The row is a CACHE of `POST /v1/tools/authorize`'s answer, so the panel and the toggle refusal can read without a round trip. **`shared_connections` is DROPPED** (11.10): every connector behind the gateway is per-user, Slack is gone, and Google Search is one of ours with no row anywhere — a table with no possible writer. |

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
| `running` | `idle` | `done{stopped}` — a human pressed Stop. **The mode is NOT touched**, which is the whole of what makes it gentle: the plan the run was approved from is still approved, the box is hibernated rather than reaped, and picking it up costs nothing. **`idle` + `unattended` IS a stopped run** — the pair is reachable no other way, since an ordinary idle session is attended and an unattended one is running or parked — and a message or a plain start resumes it unattended |
| `idle` | `running` | human sends a message |
| `idle` | `running` | `POST /sessions/{id}/approve` — asks for a plan. Mode does NOT move here |
| `running` | `awaiting_approval` | park tool (`request_approval` / `ask` / `propose_plan`). Mode is NOT touched |
| `awaiting_approval` | `running` | respond endpoint wakes it — **also flips `mode` to unattended when the row is a `plan` and the answer is the approve word.** The only mode flip into unattended there is |
| `awaiting_approval` | `idle` | a DECLINED plan: the park closes, nothing runs, the session is an attended chat again. The only answer that ends a park without waking the session, because it is the only one with nothing for the model to read |
| `running` | `completed` | `done{completed}` — also flips `mode` back to attended |
| `running` | `failed` | `done{max_hops \| wall_clock \| stalled_progress \| model_error \| internal_error \| context_overflow \| interrupted}` |
| `running` | `cancelled` | cancel signals the loop → exits via `done{cancelled}` |
| `pending`/`idle`/`awaiting_approval` | `cancelled` | cancel written directly (no loop to signal) — **and it flips `mode` back to attended**, the same rule terminals from `running` follow. A cancelled STOPPED run is where that started mattering: the hold leaves the session unattended on purpose, so without the flip it stayed recorded unattended forever, holding a quota slot for a run nobody was running |
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
  max_tools: 128               # hard cap on tool SCHEMAS in one request; ours are
                               # always loaded, so the human's allowance is
                               # max_tools - ours
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
  call_timeout_s: 120          # hard cap on one tool call; must exceed leases.wait_timeout_s
leases:
  ttl_s: 900                   # a lease expires if the process holding it dies
  wait_timeout_s: 90           # a contended call gives up here, inside call_timeout_s
  poll_s: 2
sandbox:
  template: base
  timeout_seconds: 300              # a box's own idle timeout, per e2b
  max_concurrent_per_user: 5        # boxes at once; >= quotas.max_unattended_sessions
  slot_ttl_s: 900                   # a slot unrenewed this long is reclaimed, its box killed
app:
  public_url: "https://..."    # the one origin /app and the API are served from
store:
memory:
  prompt_max_chars: 4000       # of MEMORY.md, injected into the system prompt at fold
  core_max_chars: 20000        # update_memory refuses a longer document
  note_max_chars: 2000         # save_memory refuses a longer note
  search_limit: 10             # default hits per search_memory call
quotas:
  max_unattended_sessions: 5   # per user; only UNATTENDED runs count — an idle
                               # or attended conversation consumes no worker
  new_sessions_per_hour: 20    # per user; sliding window
  upload_max_mb: 25            # per file, POST /projects/{id}/files
```

This block is the redesign DELTA, not the whole file. It does not restate
`llm.base_url`, `llm.model_name`, `database`, `mcp`, or `mcp_servers`.

Quotas are enforced in `api.py` at command time, before any state change.
`POST /sessions` checks `new_sessions_per_hour` only; `POST /sessions/{id}/approve`
checks `max_unattended_sessions`, because every session is created attended (D5) so
a check at create time always sees zero unattended load and never fires. The upload
route checks size. Over
quota returns the standard `{code, message, retryable}` error. The
count-then-act race on the concurrency check is real and unhandled in v1.

`max_tools` is the provider's hard cap on tool schemas in one request, not a
preference. `assert_coherent` refuses a configuration where `ours >=
llm.max_tools`: the session's own allowance would be zero and no connected
service could ever be reached.

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
start(session_id, *, mode=None, reason="woken") -> bool
                            # transition to running (mode moves in the SAME
                            # update), then drive one turn in the background
stop(session_id)  -> bool   # HOLD a live turn: cancel the dispatch tasks in
                            # flight, refuse the rest of the hop, park on a
                            # `resume` row at the hop boundary. No done, no
                            # mode flip. False = no live turn here, cancel instead
cancel(session_id) -> bool  # END it: task.cancel() on the whole turn, or write
                            # the terminal directly when nothing is running.
                            # Flips mode back to attended
save_plan(session_id, args, version) -> str | None
                            # the approved plan -> plan.md, through the STORE
                            # (the box is hibernated while the plan is parked;
                            # the next materialize carries the file in)
# verify-on-wake: dangling tool_call -> interrupted result surfaced to the model
# ("outcome unknown, verify before retrying"); never silently re-executed.
# The drive loop registers its sink per session so `stop` can reach it. The
# dispatch wrapper registers each CALL's own task, which is what lets a stop
# close the calls in flight without cancelling the turn: the loop stays pure and
# a stopped call comes back to it as an ordinary envelope.
```

**`api.py`** — replaces the 4x status-polling endpoints, approvals poll, bespoke
chat plumbing. One error shape everywhere: `{code, message, retryable}`.

| Endpoint | In | Out |
|---|---|---|
| `POST /auth/session` | Supabase JWT as `Authorization: Bearer` | 204 + `Set-Cookie`. Verifies it once, upserts `sub` → `users`. The ONLY endpoint that reads a bearer token |
| `DELETE /auth/session` | — | 204, cookie cleared |
| `GET /auth/me` | — | `{user_id, email, home_session_id}` — the home session is the chat the app lands in; the page needs it on first render and no other request would carry it |
| `GET /health` | — | `{status, database}` — `status ∈ ok \| degraded`, `database` is `ok` or the failure in words. Unauthenticated, because an uptime check has no session and a health endpoint that needs one reports the checker's problems rather than the service's. The only route that was live without a contracts row, which is how it got here |
| `GET /auth/config` | — | `{supabase_url, anon_key}` — what the sign-in view needs to reach Supabase. Public and unauthenticated: it is how a signed-out browser signs in, and the anon key authorizes nothing on its own |
| `GET /sessions/{id}` | — | `{title, project_id, project_title, folders[], status, mode, hops_used/max, claims[], plan, recent_events[]}` — `project_title` is the LABEL, and it is here so the window's header reads the same however the window was opened; it came from the grid's navigation state before, which a session opened from the desk does not have. — `folders` is what this session WRITES, in claim order, and the first of them is where `plan.md` lands. It replaced `project_slug` in 11.9: a project links folders rather than owning one, so there is no single directory to name, and the session header stopped drawing directory chips when there could be several. `plan` is the session's NEWEST plan (`{approval_id, version, goal, answer}`) or null: the collapsed card an approved run pins needs it, and counting `propose_plan` calls in `recent_events` drifts because that window is capped |
| `GET /sessions/{id}/events` | `Last-Event-ID?` | SSE of events, `id:<seq>` each |
| `POST /sessions` | `{goal, steps?, project_id?}` | `{session_id, project_id}` (new project unless given; `steps` seed the todo list) |
| `POST /sessions/{id}/messages` | `{text}` | 202 — appended as a user event and read at the next hop, including by a turn already running (LG-1.8). Delivery, never interruption: it waits for the current hop to finish, and stopping a run mid-step is `POST /cancel` |
| `POST /sessions/{id}/stop` | — | 202 `{stopped}` — the SAME teardown as cancel, landing gently: `done{stopped}`, `running -> idle`, mode KEPT, box hibernated, in-flight calls closed by the interrupted synthesis. Immediate: no hop boundary to reach, no window, no grace timer. `409 not_running` for anything else — an idle or parked session is already not acting |
| `POST /sessions/{id}/resume` | — | 202 — pick a stopped run back up with nothing added. A plain start: the mode was kept, so an idle unattended session resumes UNATTENDED from its plan, hops re-counted from the `done{stopped}` like every other `done`. No row to answer and no handoff injected — resuming is the absence of a change. Saying something instead is `POST /messages`, and the only difference is the words in the fold. `409 not_idle` otherwise |
| `POST /sessions/{id}/cancel` | — | 202 — ends the run and flips `mode` back to attended. From a stop there is no live turn, so the terminal is written directly — and that direct-write path hands the mode back too, which is what spends the plan |
| `POST /approvals/{id}/respond` | `{answer}` | 202 — wakes at cursor. `ask`/`approval` take prose and append a `user` event; `call` takes exactly `approve` or `decline`, appends nothing, and the resumed run executes or closes that call; `plan` takes three — the approve word saves `plan.md`, checks the unattended quota and starts the run unattended (**the only mode flip there is**), the decline word closes the park to `idle`, anything else is a REPLY — appended as the human's `user` event plus an unrendered `user{source: system}` instruction to propose again — which wakes the session attended and whose only valid answer is a new plan |
| `GET /projects` | — | `[{id, title, status_rollup, updated_at}]` |
| `POST /projects` | `{title, folders?:[name]}` | `{id, title, folders, files}` — a project made deliberately rather than as a side effect of `POST /sessions`. `folders` are store folders it LINKS, by name; `404 not_found` names any that are not in the caller's store. Picking none makes a folder named after the project (uniquified, kept alive by a sentinel) and links that, so it appears in the Files tab as an ordinary folder. This replaced `seed_from`, which copied tree rows between two project-scoped trees: there is one store now, so pointing two projects at one folder is linking it twice and the file stays one file |
| `POST /projects/{id}/folders` | `{folder}` | `{id, folders}` — link one more store folder. The UI shows it at once and the AGENT sees it from the NEXT session, because claims are fixed for a session's life. Linking twice is the same link. `404` for a folder that is not in the caller's store |
| `PATCH /projects/{id}` | `{title}` | `{id, title, folders, updated_at}` — rename, and it changes the TITLE ONLY. There is no longer anything a rename could reach: folders are the store's, derived from the paths of files that exist, and the Files tab's headers are those segments. `folders` comes back so a surface can show where the work actually lands |
| `GET /attention` | `project_id?` `session_id?` | pending approvals/asks, oldest first, each carrying its session and project. A `call` also carries `tool_name` and `tool_args` — the call itself, so the human decides on it rather than on a description. A `plan` carries the same two (the plan is in `tool_args`) plus `version`, which is a fact about the session's plan history rather than about this row. No diff against the previous version: a reply is answered by a whole new plan. One query at three scopes — no filter is the Command Center, `project_id` is a project's list, `session_id` is one window — because an approval is a state of the session, not something a surface owns |
| `GET /sessions` | `status?` | `[{session_id, title, status, mode, project_id, project_title, hops_used/max, last_event_at}]` — the user's sessions across every project. The rail asks `?status=running`; the per-project list does not compose into a view that spans tabs |
| `GET /projects/{id}/sessions` | — | `[{session_id, title, status, mode, hops_used/max, open_questions, last_event_at}]` — how the grid gets from a bubble to a window, most recently active first |
| `GET /folders` | — | `[{name, files}]` — every folder in the caller's store with its file count. What the create modal's checklist and the `+ link` picker read. There is no folders table: this is the first segment of every path the user has, grouped, which is why a folder cannot be stale and one appears the moment a file lands under a new first segment. The count excludes sentinels |
| `POST /folders` | `{path}` | `{path, sentinel}` — a folder is durable when it is NAMED, not when it is filled. A folder is a path segment and not a row, so what lands is a zero-byte `.keep` inside it: materialize, `_sweep` and flush carry files and only files, so a directory kept as a row of its own would be deleted by the first flush that ran. `409 already_exists` when anything is at that path already |
| `GET /files` | — | `[{file_id, path, name, folder, size, mtime}]` — the caller's whole store as tree rows, folder-first paths; no sandbox is woken. What the Files tab draws |
| `GET /files/{file_id}` | — | `{path, size, mtime, text, binary}` — one file's contents from the store, no sandbox woken. `binary: true` with `text: null` when it is not UTF-8: a reader that renders a PNG as characters is worse than one that says it cannot |
| `POST /files` | multipart (`file`, `path`) | `{file_id, name, path, folder, size}` — the path is REQUIRED to name a folder: every file in the store is in exactly one, because the folder IS the first segment, so a bare filename would be its own folder holding nothing |
| `POST /files/move` | `{from, to}` | `{from, to, moved:[{from, to}], stale_sessions}` — one file or a whole subtree, in one transaction, and moving BETWEEN folders is an ordinary move now: one namespace, a row edit, no copy. **Blobs never move**: they are content-addressed and immutable, so a rename is a row edit and nothing is re-uploaded. Every live box holding a covering claim is corrected in the SAME request, and that is load-bearing rather than polite — flush commits what is on disk, so a box left on the old path would put it back and delete the new one when the turn ended, undoing the move without saying so. A box that refuses comes back in `stale_sessions` and as an `error` in `system_events`, never swallowed. `404` when nothing is at `from`; `409 move_refused` for an occupied destination, a directory moved into itself, a TOP-LEVEL FOLDER as the source (that is a rename, and it has its own route), or a FILE sent to the top level — that level holds FOLDERS, so a file there would be its own folder holding nothing. A **DIRECTORY** sent there is legal and is how a folder is made by dragging: `triage/inbox -> inbox` promotes it, which is what "a folder is a top-level path segment" already says. File or directory is decided by the ROWS, inside the transaction, not by the string. The refusals say different things on purpose, and the SOURCE is tested first: `triage -> sorted` is a folder rename, not a file landing at the top level. A promoted directory belongs to no project until something links it — the files never left the store, they left the link |
| `DELETE /files` | `{path}` | `{path, batch, files, unlinked, folders}` — delete a file or a whole subtree. The rows go and the BLOBS do not: they are content-addressed, immutable and never collected, so undo is a RESTORE — the same content under the same id — rather than a best effort. A delete that empties a folder takes the folder with it, because a folder exists exactly as long as a file exists under it, and the project links that named it go into the same batch so they come back together; `folders` names what stopped existing. `409 folder_busy` while a live box has the folder mounted |
| `POST /files/undo` | `{batch}` | `{path, files, relinked, folders}` — put back exactly what one delete gesture removed, links included. `batch` names the GESTURE, so undo restores what that click took rather than whatever was deleted most recently. `409 already_exists` when something has since been put at one of those paths — it arrived afterwards and is not this batch's to overwrite. `404` for an unknown batch, one already undone, or one belonging to another user |
| `POST /files/rename` | `{path, name}` | `{from, to, moved:[{from, to}], stale_sessions}` — rename anything in the store: a file, a directory, or a top-level folder. `name` is a NAME; a `/` in it is `400`, because a rename changes what a thing is CALLED and moving it is the route above. A top-level folder's name is written in THREE places — the paths, the `project_folders` that link it, and the `session_claims` that mount it — and all three move in ONE transaction, so a project never links a folder that no longer exists. `409 already_exists` when the name is taken, and it is the NAME that must be free rather than merely the paths under it: renaming `triage` onto `notes` would otherwise fold two folders into one whenever their files happened not to clash, silently and with no way back. `409 folder_busy` while a live box has that folder materialized — the session's claims and manifest are in the runner's memory as well as the database, so a box left at `~/store/<old>/` would flush its work back under the old name and resurrect the folder, losing the turn; stopping the run first is the answer, and saying so beats a rename that half-happens. A file or a nested directory renames freely under a running session, because only the top-level name moves a mount |
| `GET /projects/{id}/files` | — | the same rows, narrowed to what this project LINKS — the working-files pane. A view of the store, not a tree of its own: same ids, same paths, so clicking one in the pane and finding it in the Files tab is one file rather than two listings that have to agree |
| `GET /results/{ref}` | `offset&limit` | blob slice (ownership-checked) |
| `GET /sessions/{id}/browser/frames` | — | SSE JPEG side-channel (`event: frame`, `{jpeg}` base64), keyed (user, session), ownership-checked, announced by a `status` event, rendered in the canvas panel (not a corner overlay). Nothing is captured while nobody is subscribed |
| `POST /sessions/{id}/approve` | — | 202 — **asks for a plan; it does not start an unattended run.** Appends a `user{source: system}` handoff ("draft the plan from this transcript and call `propose_plan`, ALWAYS") and starts an ordinary attended turn. Pressed on a fresh session with no conversation it still yields a plan card, opening as an intake form: the gaps arrive in `missing`, never as prose. Accepts `idle`, `pending` **and any TERMINAL status** — on a cancelled run this is the "resume" press, and the handoff makes it a continuation. `409 already_unattended` / `not_idle` (a live run, or a start that lost the status race — a 202 there would leave the handoff in the transcript with no hop to read it) |
| `GET /connections` | — | `[{server, label, name, status, tool_count, refreshed_at, setup_url, scopes, shares_with}]` — one row per `mcp_servers:` connector, `server` being the Arcade prefix. Status is read LIVE from Arcade, one `POST /v1/tools/authorize` per connector concurrently, and the rows are synced from it: `authorize` mints a consent link without invoking anything, so asking is free, and it is scope-aware, so it answers per SERVICE. The gateway's own `Arcade_ListApps` is NOT the source and cannot be — measured, it reports PROVIDERS (`arcade-google` covers Gmail, Calendar and Search), so connecting Gmail would render Calendar connected while every Calendar call still challenged. `setup_url` travels with the row so the panel opens the popup INSIDE the click, `scopes` says what that click grants, `shares_with` names the sibling services a disconnect takes. Google Search appears in no row: it is ours, with no grant to make |
| `POST /connections/{server}/connect` | — | `{server, status, setup_url, scopes}` — mints a fresh consent link when the row's has gone stale. Calls no tool and connects nothing; the popup is what connects. Idempotent: asking again before the user finishes returns the same pending authorization rather than starting a second |
| `DELETE /connections/{server}` | — | `{server, disconnected:[...]}` — revokes at Arcade (`DELETE /v1/admin/user_connections/{id}`) and drops our rows. **Not 204.** Arcade's connection is per PROVIDER ACCOUNT, so revoking Gmail revokes every service signed in through the same Google account; there is no narrower revoke to offer, so the one button says what it will take (the panel confirms from `shares_with` first) and the response is what actually went |
| `GET /sessions/{id}/tools` | — | `{max_tools, ours, budget, used, servers:[{server, label, name, status, tool_count, enabled, ...}]}` — the session window's tool meter. `budget = llm.max_tools - ours`, so it moves on its own when a tool of ours is added; `used` is the tool count of the servers this session was given. Rows are `GET /connections` plus `enabled`. `ours` counts what `registry.manifest` counts, which includes Google Search — it comes over the gateway and is still ours — or the meter would disagree with the request the model gets |
| `PUT /sessions/{id}/tools/{server}` | `{enabled}` | the same document. Refuses `409 not_connected` for a server nobody has authorized, and `409 tool_budget` — with both numbers in the message — for a toggle that would put the manifest over `llm.max_tools`. Turning something OFF is never refused |
| `GET /sessions/{id}/fs` | `path?` | `{path, entries:[{name, path, is_dir, size}]}` — the session's LIVE sandbox disk, not the store. Ownership-checked like the frame stream, and it never boots a box: a parked or reaped session is `404` |
| `GET /sessions/{id}/fs/file` | `path` | `{path, size, text, binary, truncated}` — one file from the live disk, on the same terms. Not-UTF-8 says `binary: true` with `text: null`; over `sandbox.browse_max_bytes` comes back cut short with `truncated: true` |

**Consent is PANEL-FIRST, and there is no callback of ours to land on.** The
popup goes to Arcade's OAuth flow and returns to Arcade; nothing redirects back
here, so `GET /oauth/callback/{server}` and the read-repair machinery behind it
are GONE (11.10) — there is no `pending` row minted before a PUT, because there
is no PUT. The panel re-reads on focus and on the popup closing, and each read
asks Arcade afresh. The model normally never meets a pre-consent challenge; if
one ever reaches it, the challenge arrives as a *successful* tool result
carrying `authorization_url` and `llm_instructions`, and **no envelope guard is
built for that** — a knowing decision (11.10), to be fixed live if it is ever
seen rather than designed for in advance.

**The gateway's `initialize` instructions and any `llm_instructions` in a
result are UNTRUSTED DATA.** Arcade says so about its own app list, and the same
posture is ours: nothing from the gateway is ever executed, and the system
prompt is built from `registry.manifest`, never from what the gateway says about
itself.

**A session reaches only what it was given.** Ours are always loaded and never
counted against the human's allowance; every MCP server is OFF until it is
toggled into that session, which is what makes an over-cap request impossible
rather than unlikely. The toggles are `session_tools`, keyed by `server` — the
Arcade app prefix — for the same reason the connection table is: a
`mcp_servers:` config key is an in-process label and nothing durable may
reference it. The meter the human sees is `enabled / (llm.max_tools - ours)`,
and Google Search is inside `ours`: it reaches the web over the gateway and is
still one of ours, with no grant to make and nothing to toggle.

**`registry.manifest` is the ONLY builder of a turn's tool list, and it cannot
overflow.** The cap is applied to the toggles rather than trusted to them: a
stale toggle set, or a server that grew its tool list overnight, must not be
able to produce a request the provider rejects outright — which is exactly how
164 schemas appeared without anyone changing anything. The drop rule is
specified, not implied: **whole servers only** (never a subset of one server's
tools — half a server is a model that believes it can post and finds out
mid-task that it cannot), **most-recently-enabled first**, so a session keeps
the reach it has been working with. A benched server is a visible fact: a
`status` event in the transcript and a `tools_benched` row in `system_events`.

**The agent has a clock, and every result says when it was true.** The system
prompt carries the current date-time, rebuilt every turn, and every tool result
the fold renders is prefixed `[fetched <when>]` from the event's own `ts`. Both
are PRESENTATION: `session_events` is byte-identical with or without them, no
wire shape moves, and no tool schema changes. Without them a model cannot tell a
week-old inbox read from this morning's, and cannot notice it slept between
turns. The stamp is absolute, never an age — an age rewrites every result on
every fold and changes the cached prefix each hop for nothing. The prompt states
the snapshot rule WITH its termination anchor (results fetched this turn are
fresh; re-read once before acting on an older one; one re-check is enough), or
"check it is still true" becomes a loop.

**The per-turn system prompt is generated from the manifest that SHIPPED, never
from the toggles.** This is not a stylistic preference. A prompt built from
toggles promises a server the cap quietly dropped, which reintroduces
prompt-doesn't-match-manifest through the emergency exit. It names what is
enabled, what is connected but off, and what was benched this turn, so the model
says "Slack is not enabled in this session" instead of improvising. The manifest
is therefore built BEFORE the fold — the prompt cannot describe a request that
has not been decided yet, so `fold(session, reach, now=)` takes it as an
argument and stays a pure function of (log, config, mode, memory, reach, now).
`now` is an ARGUMENT for the same reason: the fold reads no hidden clock, so
"same log ⇒ identical context assembly" still holds, restated as same log AND
same inputs. A caller comparing two folds passes the same instant to both.

**The home session.** First login creates one ordinary attended session and
records it as `users.home_session_id`, set once. Nothing else in the system
special-cases it: it moves through the lifecycle like any other and may sit idle
forever. What makes it home is that the app opens it by default, which is a
routing fact rather than a state one. It does not count against
`quotas.new_sessions_per_hour` — the server greeting someone should not spend
the allowance for what they ask for themselves.

**It has NO PROJECT** (11.9). It used to mint one called "Chat", because a
project was the only way to hold a directory and a session needed a directory.
Nothing holds a directory now, so the shadow project is not cleaned up, it is
unmade — a fresh account owns no project and no folder. The consequence is
honest rather than hidden: a chat that has not been given work claims nothing,
mounts nothing, and cannot approve a plan (`409 no_folder`). Asking it for work
makes a project, and that is when a folder appears.

**Steering reaches the turn it was typed into.** The loop is handed a callback
once per hop that returns whatever the human has said since it was last asked,
and appends it to the message list as a `user` turn. It is read at the TOP of a
hop, so a message typed mid-tool-call lands after the result that was open when
it was typed — the same ordering the fold applies on replay, and the one the
model API requires. The loop never reads the log itself: it is the brain, and
the log is the harness's.

**The approval gate parks on the CALL. Consent binds to the call, never to a
description of one.** `requires_approval` is checked in `envelope.execute`; with
no grant it raises the `approval_required` marker, the runner's sink DROPS that
result instead of queueing it, and the tool_call therefore stays OPEN in the
transcript. `park()` writes an approvals row of kind `call`, bound to that call's
own id and carrying the real `(tool_name, tool_args)`. The human approves the
thing that will run.

**Exactly one open tool call is permitted across a park**, and only across a
park. The partial unique index on `(session_id, tool_call_id) WHERE answered_at
IS NULL` enforces one row per call in the database; a second gated call in the
same hop is refused with `approval_required` and closes normally, to be
re-issued after the first is answered. Every reader is checked against this: the
fold never runs while the call is open (the resume settles it first), the
startup sweep only touches `running` sessions and a parked one is
`awaiting_approval`, LG-1.8 steering is carried because the call closes before
the fold, and the renderer draws the open call as a pending-call card.

**Answering runs the call, exactly once.** `approve`/`decline` are the whole
vocabulary for a `call` row — a decision, not prose, so nothing is appended as a
`user` event. Approve claims a `consumed_at` latch (the same conditional-update
pattern as `answer`, so concurrent wakes admit exactly one executor) and runs
that call through NORMAL dispatch with a one-shot grant; the result closes the
call. Decline closes it with "the human declined; choose another approach".
**A row consumed while its call is still open is repaired, never repeated** — the
process died mid-flight and the tool may have run, so the call closes as
`interrupted`: sending a message twice is worse than not knowing whether it sent
once.

`request_approval` and `ask` remain, for plan-level questions only. Their
answers are prose, delivered as a `user` event, and they close their own call
before parking. `approvals.attended_auto_approve` (default OFF) is the escape
hatch, and turning it on means every gated call is a silent yes.

**An unattended run starts from an approved plan, and there is no other door.**
`propose_plan` is a park tool of kind `plan`: the model may call it at any time,
its args ARE the plan (`goal`, `done_when`, `steps`, `inputs`, `missing`), and
the approvals row carries them — consent binds to the plan, never to prose about
it, exactly as it binds to a gated call. The two entries are the play button
(`POST /sessions/{id}/approve`, which appends a `user{source: system}` handoff
and starts an ordinary ATTENDED turn) and the model proposing off its own
judgement; both funnel through the one tool, so there is exactly one way a run
begins. Proposing is the model's judgement; deciding never is.

**Kind `plan` takes three answers, and only the first starts anything.** The
approve word writes the approved args to `plan.md` at the root of the session's
FIRST linked folder (first LINKED, not first alphabetically — the order folders
were linked in is the order they were chosen in). A session that claims no
folder has nowhere durable to put one and is refused `409 no_folder` BEFORE the
row is answered, so the plan survives. The write goes — through
the STORE, not the sandbox, because the box is hibernated while the plan is
parked and the next materialize carries the file in like any other —
checks the unattended quota, and flips mode and status in the SAME conditional
UPDATE that wakes the session. The two decision words are stored NORMALIZED, so
the row says what was decided rather than how it was typed; feedback is prose
and is stored verbatim. Everything that can refuse an approval is checked
BEFORE the row is answered — the quota, and a session with no project to write
`plan.md` into (`409 no_project`) — and a start that then loses the status race
REOPENS the row, because a plan stamped approved that nothing ran is a plan
nothing can approve again. The decline word closes the park
(`awaiting_approval -> idle`) and nothing ran. Anything else is a REPLY, and
what a reply does is below. `plan` joins `approval` and `call` in the composer's
409, so prose typed beside the card never answers it.

**Each call is a VERSION, and the newest one is the whole story.** Re-proposing
supersedes the open row (`answer = 'superseded'`, so it stops waiting on anybody
without pretending a human decided it) and the new row is v{n}; a plan's version
IS its position in the session's plan history. **There is no diff against the
previous version.** `/attention` sent `previous_args` and `last_ask` for a
"changed since v{n-1}" list and no longer does: edits stack, so by v3 the list
was longer than the plan and said less than reading the plan does. A reply is
answered by a whole new plan, and a whole new plan is what a surface renders.

**A reply closes the card and asks for the next plan.** It is appended as the
human's own `user` event — they typed it — followed by a `user{source: system}`
instruction to fold it in and call `propose_plan` again with every field. That
second event is load-bearing: without it the model answers the reply inline, and
the session goes idle with the card gone and nothing to approve, so the run the
human was setting up quietly stops existing. **`user{source: system}` events are
never rendered**: the handoff, this instruction, the continuation and the finish
nudge are the harness talking to the model, and showing them puts what we send
in among the words the human said.

An under-informed plan is still a plan: `missing` is how insufficiency renders
INSIDE the card, as named questions, so the card doubles as the intake form
rather than the model asking in prose beside it — and a plan proposed after a
reply fills `missing` again if the reply did not settle everything.

**Park, don't fail.** An unattended run ends by `finish_task`, by parking on a
question, an approval, a plan or a human's Stop, or by a named budget or stall
reason. There is no "confused" terminal: a run that does not know what to do
next asks, and a run that has stopped producing work is named
`stalled_progress` rather than being reported as an error that did not happen.

**ONE TEARDOWN, TWO LANDINGS.** Stop and cancel are the same path —
`task.cancel()` on the turn — and differ only in where it lands. The intent is
recorded per session before the task is signalled and read where the ending is
written; whichever press arrives first decides the landing.

- **Cancel** is terminal: `done{cancelled}`, box reaped, mode handed back to
  attended. That mode flip is what spends the plan, and the direct-write path
  (a cancel with no live turn) does it too.
- **Stop** is not: `done{stopped}`, `running -> idle`, mode KEPT, leases
  released, box HIBERNATED. In-flight calls close through the interrupted
  synthesis every ending already runs. Immediate — there is no hop boundary to
  reach, no window to miss, no grace timer.

**The run control has three faces, and never two at once: `▶` when nothing is
in flight, `■ stop` while running, `✕ cancel` while stopped.** The faces key off
STATUS, from the lifecycle stream, never off an endpoint's 202 — so the button
is right after a reload and right in a second tab. `▶` is `autopilot` on a
session with no live plan and `resume` on a cancelled one, where it drafts a
continuation rather than a fresh v1.

**Resuming is the absence of a change, so it is code that already exists.** The
stop kept the mode, so an idle session that is still unattended starts
unattended: `POST /messages` resumes it with the words in the fold,
`POST /resume` resumes it with nothing added. Hops re-budget from zero because
every `done` resets the count, which needs no special case. There is no
approvals row, no park kind, no arm in `respond` and no exemption in the
composer's 409 — 11.8.6 built all of those to express what keeping the mode
already says.

What this replaced, recorded because the shape is the lesson: 11.8.6 made stop a
second authority over how a turn ends — a sink flag, a dispatch-task registry, a
boundary wait, a wall-clock backstop that degraded into a cancel, and a `resume`
park with three answers — all coordinating with a loop that runs on event time.
Every coordination point was a race and first live use found three in one
afternoon. The complexity was the bug. `cancelled_by_user` went with it: the
per-tool streak is per-turn state and a stop ends the turn, so the exemption
protected nothing.

**A park kind is a wire fact, not a UI component.** A held run renders as an
ordinary inline transcript row that scrolls away like any other, the composer is
its prose input, and the approvals card family does not grow.

**The transcript is the only surface.** Every face of a plan — drafting, the
open card, the approved plan's one-line pin, a held run, a spent one, a
dismissed one — lives INSIDE the scrolling transcript and scrolls with it.
Nothing about a run is docked above the composer or floated over the
conversation. The plan lane was docked when it first shipped (11.8.5) and
overlaid the transcript, which is what the 11.8.6 canvas pass fixed.

**Pressing ▶ on a TERMINAL session is legal, and it is the important case.** On
a cancelled run the control reads "resume" and the same endpoint drafts a
CONTINUATION: the handoff CARRIES `plan.md`'s content and tells the model to
resume from what is verifiably done, rather than planning work the human has
already paid for a second time. The `terminal -> running` reopen row in the
transition table is the mechanism.

**A fact the harness knows is INJECTED into the transcript, never left for the
model to discover by tool call.** The handoff carries the plan itself — the
file's content when a run has happened here, and "no plan exists yet" when none
has. It used to say "read plan.md FIRST", which spent a tool call on a fact the
harness already had, and after a DECLINED plan that read was a guaranteed
FileNotFound, because nothing had written the file. The model's tools are for
what only the world knows.

What this replaced: the gate refused with a message naming `request_approval`
and promised "you may call {name} once they agree" — a promise nothing kept,
because the gate never read the approvals table. Every gated call looped
forever, consent bound to prose, and asking correctly was logged as
`invalid_args`, spending the per-tool failure cap on it.

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
# so a parallel fan-out of five cannot outrun a cap of two. No failure kind is
# exempt: the streak is per-turn state and every teardown ends the turn, so an
# exemption for a stopped call protected nothing (11.8.7).
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
#   unattended -> safe only if finish_task was called.
# UNATTENDED BARE TEXT IS ANSWERED, not looped: the prompt promises the model
#   "you will simply be asked to continue", so the first bare hop appends a
#   user{source:system} continuation, a SECOND CONSECUTIVE one draws the finish
#   nudge immediately, and a third ends the run done{stalled_progress}. An empty
#   reply ends it directly — there is nothing to continue from. A tool-calling
#   hop clears the streak. The near-cap nudge still fires on its own schedule.
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
# ONE manifest BUILDER — there are no session types (D5). Every session may reach
# for every hand it has been GIVEN; lazy provisioning makes an unused tool free.
# What differs between sessions is reach, not kind: same builder, same rules,
# different toggles (11.5).
#   control : finish_task · request_approval · ask · todo_write · read_result
#   world   : list_projects · get_project · list_sessions · get_session · list_files
#   memory  : save_memory · search_memory · read_memory · update_memory
#   sandbox : run_command · read_file · write_file · edit_file · list_dir · grep · glob
#   browser : browser_task
#   MCP     : whatever the SESSION was given, exposed as mcp_{name}
# Ours (~20) are authored here and always loaded. MCP tools are namespaced
# mcp_* (stripped on dispatch, so a remote read_file cannot shadow ours) and are
# the ONLY ones deferred when the schema budget is tight.
# ONE manifest builder, and it cannot overflow: registry.manifest(user_id,
# mcp=, session_id=) -> Manifest{specs, servers, ours, budget, used}. No
# session_id means no toggles means ours alone. See "A session reaches only what
# it was given" above for the drop rule.
```

**Tool layout — folder per tool.** Each tool owns its schema, its description
(own file, a function — it is the model's whole manual for that tool), its
`validate` (preconditions before work, e.g. read-before-edit), its `call`, and
its `render` (its row in Looking Glass). `registry.py` auto-discovers; adding a
tool is adding a folder.

**kept:** `arcade.py` (transport under envelope; one ClientSession, a gateway
session per user, TTL — the user's credentials never leave Arcade, and ours never
leave `.env`; it replaced `smithery.py` in 11.10 and there is no
`kind: smithery` path left); `tool_module/browser/`
(browser_tool, browser_stream, browser_actions — leashed); `tool_module/sandbox/`
(e2b manager + sandbox toolset, moved from computer_module; lazy provision on
first sandbox-tool call).

**The browser runs in the Browserless container (docker-compose); the app
reaches it ONLY over CDP via `browser.cdp_url`; the tool never launches a local
browser, and an unset URL is a loud refusal, not a fallback.**

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
| store folder (`folder:{user}:{name}`) | yes | yes (the tree it writes) | **leased**, one per write claim |
| browser (`browser:{user}`) | yes | yes (profile, cookies, logins) | **leased** |
| MCP via the Arcade gateway | yes | no (stateless per call) | no lease, runs free |
| session log | no (per session) | — | serialized by the appender, a different race |

**The sandbox is capacity, not a lease.** A box belongs to one session and is
destroyed once that session's last flush lands, so there is nothing to serialize:
overlapping writes are already ordered by the folder claims, and a box is
never compared to another box, only to the store subtrees its session claimed.
What `sandbox.max_concurrent_per_user` protects is spend, and a session over the
cap waits exactly as a lease waiter does. The slot is a row in
`session_sandboxes`, taken before the box boots and dropped when it is reaped, so
capacity cannot outlive the box that used it, and it carries an expiry renewed on
every call into the box, so a process that dies frees what it held. A session's
box is reaped ONLY after its flush lands.

**A park hibernates the box; only a terminal kills it.** A parked session gives
up every lease — it is not acting — but keeps its slot and its box, paused: it is
not over either, and the work outside the claimed mounts (a download, an install)
is still there when the human answers. Terminal flushes and kills; an expired
slot is reclaimed and its box killed with it.

**A lease wait is not a park.** A session waiting on a held lease stays `running`
and emits `status{label:"waiting for a computer"}`, which is exactly what `status`
exists for. There is no `waiting` status and there must not be one: borrowing
`awaiting_approval` would turn the project dot ochre and put a phantom row in
`/attention`. No hops burn, because no model call happens. The wall clock excludes
lease-wait time, using the same active-segment accounting park already needs. On
lease-wait timeout the tool returns
`ResultEnvelope{ok:false, error_kind:timeout, retryable:true}` saying the call
**never ran and is safe to retry later**, and the model routes around it, per
"errors are model input, not control flow". That wording is only true if the wait
gives up before the call around it is cut off, so `leases.wait_timeout_s` must
leave margin inside `tools.call_timeout_s`; startup refuses a config where it does
not, along with a `sandbox.max_concurrent_per_user` below
`quotas.max_unattended_sessions`.

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

**One idea per file** (11.8.8). `blobs.py` is content-addressed bytes and the
HTTP client that carries them; `store.py` is the TREE alone; `memory.py` is the
user's notes and curated core, keyed by user and mounted nowhere. Imports go one
way — blobs ← store ← workspace, memory standing alone — and `workspace.py` is a
transfer engine: materialize, flush, seal.

**An HTTP client belongs to the loop that made it.** `httpx` binds sockets to
the running loop, so a client cached in a module global outlives it and the next
loop finds a dead pool. Both clients — the store's in `blobs.py` and the model's
in `model_module/client.py` — are keyed by the RUNNING loop, and are fixed
together on purpose: fixing one and not the other moves the flake rather than
retiring it.

```
put_blob(content) -> sha256          # content-addressed, immutable, write-once
get_blob(sha256) -> bytes
read_tree(user_id, prefix="/") -> [TreeEntry{path, content_hash, size, mtime}]
folders(user_id) -> [Folder{name, files}]         # DERIVED: split_part(path,'/',1), grouped
folder_of(path) -> name                           # the first segment, and that is all a folder is
unique_folder(user_id, base) -> name              # the none-case, resolved against what exists
put_file(user_id, path, content) -> StoredFile    # refuses a path with no folder segment
move_path(user_id, src, dst) -> [(from, to)]      # one file or a subtree; rows only, never blobs.
                                                  # Refuses a TOP-LEVEL folder: that is a rename of
                                                  # what claims and mounts are keyed by
commit_tree(user_id, entries, prefix="/")         # blobs FIRST, rows LAST, in one transaction
append_note(user_id, text) -> path   # one file per note; never rewritten
update_memory(user_id, text)         # replaces MEMORY.md whole, under an advisory lock
read_memory(user_id) -> str          # the curated core, '' when there is none
read_notes(user_id) -> [Note{path, text, written_at}]
search_memory(user_id, query, limit) -> [Hit{path, text, written_at, rank}]
```

**The store is ONE FLAT NAMESPACE PER USER, and a folder is a top-level path
segment in it** (11.9). A folder is DERIVED, never a row: it exists exactly as
long as a file exists under it, its name is unique per user by construction —
because `(user_id, path)` is — and no table holds it, so there is nothing to
orphan, nothing to rename by editing, and nothing that can disagree with the
files. Every file is in exactly one folder, so a path with no folder segment is
refused rather than becoming a folder of its own holding nothing. A folder that
has been named and not filled is kept alive by its `.keep` sentinel, which is
also how the none-case of creating a project reserves one.

**A project OWNS no folder; it LINKS folders**, as many as it wants
(`project_folders`), and several projects may link the same one. Deleting a
project deletes its links and nothing else — files are never orphaned because
they were never owned. `projects.slug` survives only as the default NAME for
the folder the none-case creates. There is no "this project's directory"
anywhere.

**Layout is fixed.** `{prefix}/blobs/{hh}/{sha256}` for bytes;
`{user}/memory/{MEMORY.md, notes/}` for the memory region. The tree is keyed by
user and its paths are the folder-first store paths, which are also what the
model reads: mounted, a store path is the same path with the mount root in
front of it.

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

**A flush may only commit against a workspace that proves it was materialized.**
`materialize` writes a sentinel into the box (`/home/user/.ark/materialized.json`:
a nonce and the hash of the tree it laid down) and records the nonce against the
session's slot; `flush` reads both and aborts, loudly and without giving up the
box, unless they agree. The proof is about the box, not its contents: a session
that deleted every file still commits that deletion, while a box that was
replaced, emptied or materialized for another session commits nothing. Without
it an empty sweep is indistinguishable from a delete-all, and a box that died
mid-run replaces the project's tree with no rows at all.

**Memory is the user's, and it is written the way the transcript is.** It lives
in `memory_files`, keyed by user, not in any project tree: `MEMORY.md` is the
curated core and `notes/<stamp>-<rand>.md` is one appended note. A note is
written once and never edited, so concurrent sessions cannot collide. The core
is the one file that is replaced whole, and it is replaced under a
transaction-scoped advisory lock on the user — the gate a read-then-write
compactor will need, held from the first version so it never has to be retrofitted.
The bytes go to a blob as everywhere else; the row also carries the text, because
`search_memory` is a Postgres full-text query and the words have to be where the
query runs. Search is FTS and nothing more: no vectors, no embeddings, no
extraction, no per-turn retrieval.

**Memory does not mount, and whether it ever should is D30 — open.** Today no
claim can name the region and nothing in `materialize` reads it. That is the
default posture, not a proof: the alternative on the table is a claimable
read-only mount, which would put memory under the same uniform claims rule as
everything else, and settling D30 that way costs one additive migration rather
than a rebuild. Nothing in the schema or the code is written to foreclose it.

**The core is carried into every turn; the notes are searched on demand.** Fold
injects `MEMORY.md` into the system prompt, capped at `memory.prompt_max_chars`
with a marker naming `read_memory` for the rest. The four tools —
`save_memory`, `search_memory`, `read_memory`, `update_memory` — are the model's
only writers, and `update_memory` refuses until `read_memory` has run in that
turn, because a whole-document rewrite from the capped copy would silently drop
its tail. The model is the compactor in v1; the background job that will curate
unattended is later, and needs no entry point the model does not already use.

**Claims are the unit of conflict** (D29), and since 11.9 a claim names a
FOLDER: `(folder, subpath, mode)`, declared at creation. The set is the sole
source of which leases it takes and what appears in its sandbox. Nothing
unclaimed is mounted. A read claim materializes without a lease and its flush is
a no-op, with discarded edits logged.

**Write leases are per FOLDER**, keyed `folder:{user_id}:{name}`. Two projects
writing DIFFERENT folders never contend, and two sessions writing the SAME
folder still serialize even when they belong to different projects — which
keying it by project could express only while a project held one directory.

**The agent's durable paths are `~/store/<folder>/`, one mount per claimed
folder.** A mounted path is the store path with the mount root in front of it,
so the model reads one namespace and not two, and the per-turn system prompt
NAMES the folders the session holds — with several of them possible, "the
project directory" is not something it can infer.

**Claims are FIXED for a session's life.** A folder linked to a project
mid-session reaches the agent at the NEXT session while the UI shows it at once.
That is recorded as a fact rather than papered over: a folder appearing under a
running run would be a mount and a lease it was never told about.

**Every session spawned in a project receives its links**, all write, through
`_record_claims`. A session with no project claims nothing and mounts nothing —
that is the home chat, and it is why **the home session no longer mints a shadow
project**. A project existed only to hold a directory; no directory is held, so
the orphan is not cleaned up, it is unmade.

**An upload lands in the store, and in the boxes already holding it.**
`POST /files` writes the blob and the tree row first, so the file
exists whether or not anything is awake. It is then written through to every
running session whose box has a materialized claim covering that path, so a
session reads it the same turn; every other session gets it at its next
materialize, and a parked session is left asleep. The write-through is a plain
overwrite: on a path the session is itself editing the last write wins, which is
safe because the bytes are in the store either way and `edit_file` refuses a
stale `old_string` rather than writing over what landed. A write-through that
fails is logged and nothing more — the store already has the file. An empty file
is content like any other. `quotas.upload_max_mb` is enforced as the upload is
read, not after.

**PREVENTION, NOT SYNCHRONIZATION: a destructive store operation against a
folder whose WRITE LEASE is held is refused** (`409 folder_busy`). Move, rename,
delete and undo all ask `leases.holder("folder:{user}:{name}")` first. A session
holds that lease for the whole time it is writing the folder, so if nobody holds
it there is no live box to diverge from the store, and if somebody does, no HTTP
handler can correct them — the runner holds that session's claims and its
materialized manifest in MEMORY. A box that still held a deleted file would put
it back at its next flush; one that lost it would commit the loss.

This replaced a coherency protocol. `move_through` pushed every store-side move
into every live box with remote `mv` and `rm` so a flush would not undo it, and
reported the boxes that refused as `stale_sessions` — the scariest path in the
subsystem, chasing a moving cache to repair a divergence one lease check simply
prevents (11.8.8). Read claims take no lease and need none: their flush
discards, so they cannot put anything back.

`write_through` is the one live-box write that stays, because it is ADDITIVE:
an uploaded file put into a box already working that folder, where the store
already holds the file either way and a failure is a logged nothing.

**Nothing is destroyed, only unlisted.** Deleted rows move to `deleted_files`
(with `deleted_links` for the links the same gesture dropped) rather than
staying in `files` behind a `deleted_at` flag: a flag would sit in the way of
every tree read, of `put_file`'s `ON CONFLICT (user_id, path)`, and of the
unique index that makes folder names unique per user. `files` holds live rows
and only live rows, which is what every other statement already assumes.

**A folder's name lives in three places, and one operation moves all three.**
The paths (it IS their first segment), the `project_folders` that link it, and
the `session_claims` that mount it. `POST /files/rename` is that operation and
the only one: nothing else may write a folder name, which is what keeps
"derived, never a row" true of a thing that three tables nonetheless spell out.

**What is NOT in 11.9, and is deliberately absent rather than forgotten:**
unlinking a folder from a project; exposing per-folder read mode (claims support
it; links ship write-only); and blob GC.

**Snapshot TABLES dropped (11.8.8).** 11.7.5 removed the capability and 0015
dutifully rekeyed the dormant rows into the user namespace — which is when it
became clear what was being maintained: a correctly-migrated table for a feature
that does not exist. `project_snapshots` and `snapshot_files` are gone; when
snapshots return they get designed against the folder store natively rather than
inheriting a shape drawn for `project_files`, which no longer exists either.

**Snapshots were REMOVED (11.7.5).** `snapshot_project` / `restore_snapshot` /
`list_snapshots` / `prune_snapshots` had no caller in the tree, and
`scripts/snapshot_store.py` — named by the config comment that configured them —
has never existed in this repository's history. The design was sound (a snapshot
is tree rows, which is only true because blobs are immutable and never mutated),
and it is in git if it is ever wanted. What is not kept is 142 lines of a
capability nothing invokes.


**No store credentials enter the sandbox.** Bytes flow store → harness → e2b
API, never sandbox → store. This is also why the FUSE probe mounted rclone's
local backend rather than a bucket: what needed answering was whether
`/dev/fuse` works in the box (it does), and answering it did not require handing
a box a credential.

### (computer_module: dissolved)

With its agent deleted, `computer_module` had no reason to be a module. The
sandbox manager and toolset move to `tool_module/sandbox/`; `agent.py`,
`model.py`, `runner.py`, `computer_router.py` are deleted/folded into the
unified task path. Backend module map is now role-per-module: `harness_module`
(control plane) · `agent_module` (brain) · `model_module` (model) ·
`tool_module` (all hands).

### memory_module — deleted; memory returned as part of the store (8.8)

mem0 is gone for good, with its embedder service, extraction LLM calls, config
and deps. What came back, in `harness_module/store.py` rather than a module of
its own, is the half that was ever worth having: explicit writes over a schema
we own, FTS before vectors, and nothing on the hot path. The contract is the
memory section of the store above.

What stays deleted: `retrieve_long_memory` auto-injection (embed + search every
turn), background `add_memory` extraction, the mem0 `memories` collection, and
`memory_module` itself. The model decides what to remember and when to look, the
same way it decides to read a file.

Still not built: the background compaction job (the model curates for now), and
pinned standing rules as a first-class thing rather than lines in `MEMORY.md`.

---

## Frontend

Consumes ONLY the endpoint table + event vocabulary. Types in a hand-maintained
`types.ts` mirroring `events.py` (CI check; codegen later if the surface grows).

- No polling. Per open surface: one snapshot request + one `EventSource`
  (cookie-authenticated — no token in the URL); reconnect via `Last-Event-ID`;
  apply events as deltas.
- Commands are optimistic: apply locally, reconcile on the event.
- Renderer switches on `event.type`; unknown types render as a generic row, never crash.
- **The Files tab and a session's working-files pane are ONE component at two
  scopes** (11.9): the whole store, and the folders one project links. Same
  powers in both — open, drag to move, drop to upload, double-click to rename —
  because they are two views of one namespace and not two filesystems. What a
  caller supplies is which rows to load and what a click does; everything else
  is behaviour, and behaviour has one implementation. They were two components
  with two different sets of abilities, which is the only reason a file could be
  renamed in one pane and not the other.
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
| `test_transcript_invariant` | every tool_call.id closed once; interrupted synthesis on abort. ONE exception: a call parked for approval stays open across the park and is closed by the answer |
| `test_lifecycle_transitions` | ALLOWED map only; cancel wins a live race |
| `test_streaming_first_token` | first content event before the mocked model finishes |
| `test_retry_budget_bounded` | ≤3 attempts, bounded wall clock; background no-retry |
| `test_append_ordering_under_concurrency` | concurrent api and loop appends; an SSE reader never skips a seq |
| `test_resume_verify_on_wake` | dangling call → interrupted; no silent re-execution |
| `test_event_replay_deterministic` | same log AND same inputs ⇒ identical context assembly, ladder included: a log over the input budget folds to a view under it, and folds byte-identically twice. `now` and `reach` are inputs, not ambient reads |
| `test_authz_scoping` | user A cannot read/steer user B's sessions, refs, files |
| `test_plan_gate` | one door into an unattended run: `propose_plan` parks kind `plan` with the plan on the row; approve writes `plan.md`, checks the quota and flips mode+status in one update; decline closes the park to `idle`; anything else is a reply that closes the card and asks for the next plan; a composer message 409s; play appends the handoff without flipping mode, and is accepted on a TERMINAL session (the continuation press) |
| `test_stop_resume` | stop and cancel are one teardown with two landings: stop lands `idle` with `done{stopped}`, the mode kept and the box hibernated, its in-flight calls closed `interrupted`; a message resumes it unattended with the words in the fold and a plain start resumes it with nothing added; cancel from a stop lands `cancelled` with the mode handed back, and so does a direct-write cancel of an idle unattended session; the handoff carries plan.md's content rather than telling the model to read it |
| `test_folders_are_the_filesystem` | a folder is a derived top-level segment and no table holds it; renaming one moves its paths, its links and its claims together, is refused onto a taken name, and is refused while a live box holds it; a delete keeps the blobs so undo restores the same content under the same id, takes the folder and its links when it empties one, restores the batch it NAMES, refuses to overwrite what arrived since, and is refused while a live box holds the folder; a project links folders and owns none; `POST /projects` with links → folder claims at spawn, with none → one fresh folder; the link endpoint writes `project_folders` and reaches the NEXT session's claims, not this one's; write leases are per folder, so two projects writing different folders do not contend and the same folder does; `plan.md` lands in the first linked folder; a rename touches no folder; no home-session project is minted |

---

## Current violations (bugs the contracts outlaw — fix targets, not contracts)

| Violation | Breaks | Where |
|---|---|---|
| ~~browser frame queue keyed by user only; concurrent tasks clobber each other~~ **REBUILT in Task 9**: `FrameBroker` is keyed `(user_id, session_id)` | (user, task) stream keying | `tool_module/browser/stream.py` |
| ~~`/v1/browser/stream` trusts client-supplied user_id, no auth~~ **REBUILT in Task 9**: `GET /sessions/{id}/browser/frames`, cookie-authed and ownership-checked like everything else | every-endpoint ownership check | `harness_module/api.py` |
| ~~browser returns bare string; failure and empty are both `""`~~ **REBUILT in Task 9**: the envelope is built from the run history, `ok` from its own verdict, the record behind a `ref` | tool_result envelope | `tool_module/browser/tool.py` |
| ~~zero step callbacks wired; 3-min silent tool call~~ **REBUILT in Task 9**: every step is a `status` event, and a vendor that drops the callback logs at WARNING | progress-as-status-events | `tool_module/browser/tool.py` |
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
latency and permissive about consequences. Since 2026-08-18 an attended run notices too: the gate no longer
answers yes on the human's behalf, so a gated call sends the model to
`request_approval`, which parks and asks. An unattended run is fifteen hops of
unsupervised Gmail, Slack, GitHub and Linear otherwise, which is the tool class
this whole approval surface exists for. Waive
per server with `mcp_servers.<label>.auto_approve` — `true`, or a list of tool
names. Loosening a default is a config edit; tightening one after unattended
runs have been sending mail is an incident.
