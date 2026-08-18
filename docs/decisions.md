# Architecture Decisions

One entry per decision that would otherwise get re-argued. Format: what we
chose, why, and what it costs. Newest at the bottom.

Not a discussion log — only decisions that are made. If it is still open, it
lives in a spec's open questions, not here.

---

Grouped by module. Numbering is chronological (D1 is oldest) and never reused.


---

## Whole-system

Shape of the thing; nothing below contradicts these.

**D1 · One loop, no conversation state machine.** `run_turn` calls the model,
runs tool calls, repeats. Routing already exists in the response shape (tool
calls present or absent), so the state graph was re-implementing in 3-5 model
calls what the chat template does in one. *Cost:* the model has more rope; the
guardrails move to the tool boundary (manifest, budgets, approval flags).

**D2 · No orchestration framework.** LangGraph and friends encode assumptions
about what a model cannot do, which go stale as models improve. Our loop is
~200 lines we own. *Cost:* we maintain it.

**D3 · The log is the only durable truth.** Append-only `session_events`; the
harness holds nothing that cannot be rebuilt by folding it. *Cost:* every read
path is a fold; nothing is cached in process.

**D4 · The lifecycle FSM lives in the harness, next to the agent, never inside
it.** The loop emits `done{reason}`; the harness decides what that does to the
row. *Cost:* one more boundary to respect.

**D5 · One session, two modes — nothing spawns.** The session you chat with is
the one that executes and the one you reopen in the project tab. Planning and
doing are the same conversation. `create_session` is deleted: handing
`{goal, steps}` to a fresh worker threw away everything else that was said.
*Attended* (human present, turn-taking, ends its turn) flips to *unattended*
when you approve the plan, and back when the run finishes. Parallelism lives in
the project grid — five things at once is five projects. *Cost:* one session
cannot be in two places, so fan-out needs five projects (or `create_session`
returns later); the full manifest is always present, which lazy provisioning
makes free.

**D27 · Storage is separated from compute.** The agent's files live in object
storage we own, with the tree in Postgres: `project_files` carries
`path, content_hash, size, mtime` and no bytes, and blobs are content-addressed
by sha256. The sandbox disk is a cache, filled when the session takes its box and
flushed before it gives it back. e2b's own persistence is demoted to a warm-start optimization, so
deleting a paused sandbox loses nothing. A filesystem is bytes plus a tree; both
halves are ours. Content addressing buys dedup across projects, snapshots that
cost a row copy, and crash-safe writes, because a blob is written once and never
mutated. *Cost:* every lease acquire pays a materialize and every release pays a
flush, so a project too large for eager sync needs the lazy-mount rung that does
not exist yet; and the vendor's filesystem being a cache means a warm resume
must diff rather than trust what it finds.

**D28 · The agent lives outside the computer.** The loop, the transcript writer
and every credential stay in the harness; the sandbox holds files and runs
commands. Nothing that can read the model's context or the user's secrets is
inside a box that executes model-authored code. Inner loops are allowed only as
leashed tools behind the tool boundary, which is what `browser_task` already is.
*Cost:* bytes cross the harness on the way in and out rather than the sandbox
pulling them directly, which is a transfer we pay for and a credential we never
hand over.

**D29 · The unit of conflict is the path set, named and persisted as claims.**
A session declares its claims at creation — `(project_id, subpath, mode)` — and
that set is the sole source of two things: which leases it takes, and what
appears in its sandbox. Nothing unclaimed is mounted. Write claims take
`project:{id}`; read claims mount leaselessly and discard their edits. Memory is
shared, never leased, append-gated and compacted on its own. Fixing the set at
creation is what keeps the lease story race-free: a session cannot acquire its
way into a deadlock halfway through. *Cost:* no mid-session "add folder" without
a design for re-acquisition, and a session that guessed its claims wrongly has
to be restarted rather than widened.


**D30 · Whether memory is ever readable inside the box is OPEN.** Memory does
not mount today: no claim can name the region and nothing in `materialize`
reads it. That is a posture, not a ruling. The case for keeping it out is that
the sandbox executes model-authored code and the memory core is the most
distilled thing we hold about a user; the case against is that the agent owns
its computer and its memory both, and a claimable read-only mount would put
memory under the same uniform claims rule as everything else — declared,
visible in Looking Glass, writes still gated. Settled when there is a real
reason to read memory from inside a box; until then the default stands. Nothing
is written to foreclose either answer, and the mount answer costs one additive
migration (a region column on `session_claims`). *Cost:* the question stays
open, so anyone reaching for memory inside the box has to come back here first.


---

## agent_module — the loop

`run_turn`, events, termination.

**D9 · Native tool calling replaces enum-constrained selection.** Validate and
retry once on malformed args. *Cost:* gated on Task 0a measuring the 8B's
malformed-call rate; if it is bad, temperature or constrained decoding.

**D15 · Termination is one rule with a safety predicate.** A run ends when the
model stops calling tools AND the exit is safe: always when *attended* (a human
is the continuation), only via `finish_task` when *unattended*. Evaluated
against the current mode, not a session type. *Cost:* an unattended run that
finishes but forgets `finish_task` burns budget to `failed{max_hops}`; mitigated
by the system prompt and a low-budget nudge.

**D26 · Every pause is a tool that parks — `ask` included.** An ask was going to
be plain text ending the turn, but `/attention` and the ochre dot are driven by
`status=awaiting_approval`, which only a park event can set. Text-only asks would
have needed the runner to *infer* "that turn ended on an unanswered question,"
putting model-output interpretation back on the lifecycle path the deterministic
gate exists to keep off it. So `ask` is a real tool alongside
`request_approval`, and questions become first-class in `/attention` for free.
Corollary: **every open question glows ochre**, so the grid is a complete
"waiting on me" inbox rather than only a background-work monitor. *Cost:* one
more manifest entry, and an active chat with an unanswered question stays ochre
until you answer it. That is intended.

**D20 · Buddy plans with read tools, not a world digest.** The world is bigger
than any snapshot (every file, every browser tab, every service's state), so a
digest would be a confident lie. Buddy gets `list_projects`, `get_session`, and
friends, and the system prompt tells it to look before planning. *Cost:* an 8B
may not look; escalation ladder is prompt, then a small tail snapshot, then
something structural.


---

## harness — control plane

Lifecycle, log, runner, api.

**D13 · One global `BIGSERIAL` for event ordering.** Several things append
(loop, API, lifecycle) and a parked session has no loop to funnel through, so
per-session numbering would need a lock per write. *Cost:* a session's numbers
have gaps; harmless, since every read is "after N."

**D14 · A task IS a session — one id.** The old `session_id` column existed only
as a mem0 bucket key. *Cost:* pick one word; recommend renaming the table to
`sessions`.

**D16 · On restart, `running` fails; `idle` survives.** No auto-resume: a
session caught mid-run becomes `failed{interrupted}` and a human decides whether
to restart. `idle` conversations are untouched, so deploys never kill open
chats. Any session reopened later gets a synthetic `interrupted` tool result if
its log ends on an unclosed call — otherwise that conversation is permanently
unloadable, since the message list is rebuilt from the log on every load.
Restart is a human act, not a sweep: the operator hits restart and the session
takes the `terminal -> running` transition, which is the same mechanism that lets
you keep talking to a finished session. One affordance, two uses.
*Cost:* a long run that dies at 95% is re-run from scratch.

**D17 · Two log tables, not one, and not files.** `session_events` is the
transcript: user-visible, audit-grade, never pruned, and a failed append halts
the run because nothing may execute off the record. `system_events` is
operational: retries, lease churn, timing, DB hiccups — batched best-effort
writes that never block, pruned at 30 days, and a failed write is just a lost
diagnostic. The split is by AUDIENCE (does a human need to see this in a
transcript), not by location. stderr survives for exactly one case: the database
being unreachable, which cannot be written to the database. A small `logging`
module owns the JSON formatter, the session-id context var, and secret
redaction. *Cost:* two sinks to reason about; correlate via `session_id`.

**D22 · `idle` is a real state, and every state must be visible.** After a
session replies and before you type again it is not `running`, `completed`,
`awaiting_approval`, or `pending` — it is alive and waiting for a human. That is
`idle` (neutral grey in the grid), and an attended session may sit there
forever; terminals come from the unattended run. Guard rail against the state
graph creeping back: **a state must change what a person sees.** `pending` is
queued, `idle` is yours to continue, `running` is working, `awaiting_approval`
needs you, the terminals are done three ways. A proposed state that changes
nothing visible means control flow is being smuggled into the status column.
*Cost:* seven states instead of six.


---

## model_module

One client, one retry layer.

**D25 · Qwen3-8B on SGLang, with both parsers set at launch.** One authoritative
model config; `run.sh` is the only place it is declared. `--tool-call-parser
qwen25` (Qwen3 dense models use the Hermes-style `<tool_call>` format the qwen25
detector reads; `qwen3_coder` is for the Coder variants and `qwen3` is not a tool
parser at all) plus `--reasoning-parser qwen3`, because Qwen3 is a hybrid
reasoning model and un-split `<think>` output would stream into `content` events
as if it were the reply. `run.sh` is authoritative; `config.yaml` mirrors it so
the client can read it, and a mismatch between the two is a bug, which is what
Task 0a(i) verifies. The stale `"tgi"` model name and the `computer_agent.llm`
block are deleted (`computer_agent.sandbox` is promoted, not deleted, until Task 8). *Cost:* reasoning is STREAMED, not
dropped: it re-yields as a `reasoning` event alongside `content`, because Qwen3
reasons for hundreds of tokens before answering and a blank screen through the
slowest part of the turn is the opacity this redesign exists to remove. That makes
the audit table bigger, which is the honest price. Task 0a still measures whether
thinking earns its latency on our schemas; there is also a known Qwen3 failure where reasoning and tool calling
together leak tool-call XML into content, which the same task has to rule out.


---

## tool_module — hands

Tools, sandbox, browser, MCP.

**D6 · One agent. ComputerAgent deleted, computer_module dissolved** into
`tool_module/sandbox/`. It was our loop duplicated, with the same bugs.
*Cost:* none found.

**D7 · browser_use kept, leashed behind one tool.** Rebuilding DOM grounding is
negative payoff. *Cost:* a third-party loop inside a tool call; we bound it with
per-call budgets and surface its steps as `status` events.

**D19 · Folder per tool, Claude Code's shape.** Each tool owns its schema,
description (its own file, as a function), `validate` (preconditions before
work), `call`, and `render`. Registry auto-discovers. MCP tools are namespaced
`mcp_*` and are the only ones deferred. *Cost:* more files; worth it.

**D24 · Connections are identified by url, and the connection id is stored, not
derived.** `_user_conn_id()` builds Smithery's connection id by formula from a
config key, which makes a cosmetic-looking rename in `config.yaml` disconnect
every user and strand the old connection in Smithery's namespace. Anything
derived from mutable inputs is a drift surface, so we remove the derivation: the
id is generated once at connect time and persisted, and lookup is by `mcp_url`.
Config keys stay as in-process labels for logs and the tool registry, rebuilt
from config each startup, referenced by nothing durable. **We mint it, Smithery accepts it** (the id is a path segment we choose), and the
row is written FIRST: INSERT `status='pending'`, then PUT, then UPDATE from the
response. Minting after the PUT would strand the old connection holding a live
OAuth grant on any crash between the two, which is the same drift moved to
crash-time. *Cost:* one more stored column, a cheap pending row when a connect is
abandoned, and the id is no longer human-readable in Smithery's dashboard unless
we slugify the host when minting.

**D18 · Stateful hands are leased: sandbox and browser.** Both are persistent
per user (browser keeps a profile with logins), so both are held under a lease
for the whole task, released on park. MCP is stateless and runs free. *Cost:*
sandbox work is effectively serial per user; browser profiles become sensitive
at-rest state.
**Superseded for the sandbox, 2026-08-18 (spec Task 8.6b).** Once D27 made the
sandbox disk a cache, the box stopped being shared and stopped being stateful:
one box per session, capped per user by `sandbox.max_concurrent_per_user`, with
the `project:{id}` claims doing the serializing the lease used to. The browser is
still leased, and the rule the lease came from — serialize iff shared AND
stateful — is what removed the sandbox from the list.


---

## memory_module

**D8 · Long-term memory removed.** Explicit writes made mem0's auto-extraction
dead weight; the transcript is memory within a session. *Cost:* no
personalization until it returns; standing rules come back first.

**Amended 2026-08-18 (Task 8.8).** The explicit-write half is built, in the
store rather than a module of its own: `save_memory`, `search_memory`,
`read_memory` and `update_memory` over `memory_files`, searched with Postgres
FTS, with `MEMORY.md` injected into the system prompt at fold. What D8 deleted
stays deleted — no auto-extraction, no embeddings, no retrieval on the hot path.
The model decides what to keep and when to look.


---

## db

**D12 · asyncpg from day one.** The harness is new code, so there is nothing to
migrate. *Cost:* `$1` placeholders, and `statement_cache_size=0` if we point at
Supabase's transaction pooler.


---

## frontend

**D10 · Push, not poll.** Store-shape equals wire-shape: the event we save is
the event we stream. No `setInterval` anywhere. *Cost:* reconnect logic, which
`EventSource` gives us free.

**D23 · The right panel is a pinned TODO plus one canvas, not a corner pane.**
TODO is always visible (five lines, always relevant); below it the canvas shows
working files OR the live browser, one at a time. No memory tab — memory is
removed (D8), and a tab for it would spec a deleted feature.
The browser is announced by a `status` event and made *available* — it never pops
over the conversation. Choosing the tab is the difference between watching and
being interrupted, and a full-height panel is actually legible where a 360px
floating thumbnail is not. Frames stay ephemeral (never replayed from the log).
*Cost:* more layout work than an overlay; canvas open/closed state is per-user UI
state we have to remember.

**D21 · Cookies + EventSource for browser auth.** httpOnly, Secure,
SameSite=Lax session cookie set by our backend after verifying Supabase's JWT
once. The browser sends it automatically, so `EventSource` works with no
stream-token machinery and we get `Last-Event-ID` reconnect free; XSS cannot
read an httpOnly cookie. The old bearer-only design existed because wildcard
CORS (a demo convenience) forbids credentials — in production the frontend is
same-origin, so that reason is gone. *Cost:* CSRF surface, handled by SameSite
plus an origin check on mutations; non-browser clients (CLI, mobile) would need
bearer alongside, which is a small addition when it happens.


---

## infrastructure / runtime

**D11 · Sessions are asyncio coroutines, not threads.** The GIL makes threads
pointless here and the work is I/O-bound. The rule is never block the event
loop. *Cost:* one sync DB call anywhere freezes every user, so the rule is
absolute.
