# Implementation Notes — Single-Loop Redesign

> Split out of `single_loop_redesign_spec_01.md` on 2026-08-21 (owner) to keep
> the live spec readable. Same standing as before the move: operational facts
> and hard-won wiring detail, appended as they are learned. The plan lives in
> `single_loop_redesign_spec_01.md`; the background in `_00`; contracts is law.

Newest last. Keep entries short. An entry earns its place only if a later
session would otherwise re-derive it — a measurement and the decision it forced,
a deviation from the plan and why. `git log` is the changelog; this is not.

When work contradicts this document, **amend the document in the same commit.**

---

**2026-08-11 — Task 0a(i): config was lying about the served model.** `run.sh`
launched Qwen2.5-7B-Instruct with no parser flags; the live container serves
**Qwen/Qwen3-8B** with `--tool-call-parser qwen25 --reasoning-parser qwen3`
(verified via `docker ps` + `/get_server_info`). `llm.model_name` was the
placeholder `"tgi"`; context window 32768 → 40960 (native, no
`--context-length` override). `computer_agent.sandbox` was **promoted** to a
top-level `sandbox:` block rather than deleted — `computer_module/sandbox.py`
reads it at runtime and Task 8 has not relocated it yet. Commit `5f3705f`.

**2026-08-11 — Task 0a(ii)+(iii): native tool calling measured. It passes.**
22 realistic prompts, real SGLang endpoint, our actual tool schemas, default
temperature:

| | |
|---|---|
| malformed calls | **0 / 22** |
| tool-call XML leaking into `content` | 0 |
| transport/API errors | 0 |
| spurious calls (tool used when none needed) | 0 |
| answered in prose instead of calling a tool | 2 |
| latency | median 8.3s · mean 13.8s · **max 44.2s** |

**Decision: no temperature drop for tool turns, no constrained decoding, no
xgrammar.** Threshold was ~5% malformed; measured zero. The feared
`--reasoning-parser` × `--tool-call-parser` interaction did not appear.

Caveats worth carrying forward:

- **Bound to Qwen3-8B with those two parser flags.** Change either and
  re-measure before trusting this.
- Task 2's single repair retry stays regardless — it is cheap, and 22 prompts
  is not a proof.
- The 2 prose answers were the model declining to call a tool, not bad output.
  A Task 2 prompting concern, not a decoding one.
- Max 44.2s is the number for Task 1: under `timeout=90`, but not by the margin
  the median implies.

**2026-08-13 — Task 0c: migration 0 written, and the spec was wrong about it.**
`db/migrations/0000_migration_0.sql` is built from `schema.md` + contracts, not
from 0c's own table list — that list named `tasks`/`task_approvals` (renamed to
`sessions`/`approvals`) and omitted `sessions`, `system_events`,
`resource_leases`, `shared_connections`. Building from it would have shipped a
schema with no operational log and no lease table. 0c is now corrected; a sweep
of the other law docs for stale *table names* came back clean (that sweep did
not cover other stale values — `contracts.md` still had `context_window: 32768`,
found later by review and corrected to 40960).

Two details: the migration deletes the `schema_migrations` rows for
`0001`–`0007`, so a DB that ran the deleted chain does not report them applied;
and `repeat_tasks` has no FK to `users`, so it survives the `users` drop.

Verified by applying it to a throwaway DB — 12 tables, bookkeeping correct.
**Not yet applied to a real database:** it drops `users` and `tasks`, so it runs
at the Task 4 cutover.

**2026-08-13 — Task 1: a mid-stream failure is NOT retried by the client.**
The contract says the client retries within one `generate()` call, but a stream
that dies after emitting deltas cannot be replayed — the caller has already seen
the text, and retrying would duplicate it on screen and duplicate tool calls in
the transcript. So: retry only before the first delta; after that, raise
`retryable=True` and let the loop re-attempt the whole hop (Task 2). Still two
bounded layers, as contracts require. Pinned by
`test_midstream_failure_is_not_retried`.

Also settled here: `source` is `interactive | background` and changes exactly
one behaviour — `background` does not retry overload (429), so an unattended run
yields the GPU slot instead of queueing for it. Everything else retries
identically. New config keys under `llm:`: `timeout_s: 90` (0a measured max
44.2s), `max_attempts`, `retry_backoff_s`, `retry_backoff_max_s` — no magic
numbers in the client. `ModelError` gained `kind`; it defaults to `"unknown"`
only so the pre-redesign callers keep working until Tasks 7-8 delete them.

**2026-08-13 — review pass on Tasks 0-1; the teardown was the weak half.**
Migration 0 created the right schema but under-dropped. Added:
`computer_tasks` / `computer_task_events` (a DB that stopped at `0006` can never
run `0007` now the chain is deleted, so 0007's drops had to move here), mem0's
`memories` pgvector collection (memory_module pointed mem0 at *this* database),
and the orphaned `set_updated_at()`. Re-verified by seeding a realistic
pre-redesign DB — FKs, trigger, enum types, `waitlist` and `repeat_tasks` with
rows — then applying: zero leftovers, both preserved tables intact.

`user_sandboxes` is dropped rather than migrated (TEXT → UUID `user_id` cannot
carry across). That loses every live e2b handle, so sandboxes must be reaped
BEFORE the cutover or they keep billing to nobody. Now a WARNING in the
migration header.

Also corrected: `contracts.md` said `context_window: 32768` while config said
40960 — contracts contradicted its own rule that the value match the actual
SGLang launch, so contracts moves to 40960. And Task 1's config key was renamed
`max_attempts` → **`max_retries`** to match contracts, though the two halves of
contracts disagree on its meaning (the yaml block says "max_retries: 3", the
conformance test says "<=3 attempts"). Implemented as a cap on total attempts.
**Open question for the author: which did you mean?**

**2026-08-13 — sequencing settled: no flag, no bridge.** 0c runs after Tasks 1-3
and immediately before Task 4. The app is down in between. A flag was considered
and rejected: 0c drops the tables the old path reads, so the off position would
500 rather than roll back. Task 4's flag and Task 7's flag-removal step are gone
from the plan; Task 7 is now mechanical deletion of dead code.

**2026-08-13 — `append()` must hold `pg_advisory_xact_lock(session_id)`.**
BIGSERIAL assigns seq before commit, so an api transaction holding 100 can commit
after the loop's 101 and an SSE reader that already sent 101 never emits 100.
The lock makes commit order equal seq order within a session. D13 is untouched:
it argued against per-session counters, not per-session locking. The user-scoped
grid feed (G27) spans sessions and needs a hold-back window when it is built.
Pinned by a new conformance test, `test_append_ordering_under_concurrency`.

**2026-08-13 — schema.md amended while amendments are still free.** CHECK
constraints on `sessions.mode`, `sessions.status` and `session_events.kind`; a
partial unique on `approvals (session_id, tool_call_id) WHERE answered_at IS
NULL`; `ON DELETE CASCADE` on `resource_leases.session_id`; and the
`(user_id, created_at)` index the `new_sessions_per_hour` quota actually uses.
Verified against a live DB: bad status, double-park and bad kind all rejected.

Two known-wrong model declarations are recorded in the contracts violations
table rather than fixed: `app.py:67` (dies with Task 4) and `memory.py:58`
(dies with Task 7). Fixing scheduled corpses is wasted effort; leaving them
invisible is the actual risk.

**2026-08-13 — 0c applied, and only the real database found the last two holes.**
The migration had been verified against a seeded throwaway DB and reported clean.
Run against the actual one it failed on the first statement, and the second gap
would have passed silently:

- `public.memories` is a **VIEW** over `vecs.memories`. mem0 goes through the
  `vecs` extension, so the real vectors (500 rows) plus `memories_entities` and
  `mem0migrations` live in the `vecs` schema. `DROP TABLE memories` errored;
  dropping only the public name would have left every vector behind.
  `vecs.urop_benchmark*` is a different project's and is deliberately untouched.
- `user_oauth_tokens` — pre-Smithery OAuth storage, 3 rows of plaintext
  `access_token`/`refresh_token`, read by zero lines of code, dropped by no
  migration. It would have outlived the users keying it. Now dropped.

The lesson worth carrying: a seeded DB proves the migration handles what you
remembered to seed. Neither of these was in the seed.

Sandboxes were NOT reaped first. Four were alive (all paused since June); their
handles are `ij06czr3ca5fl784k5lbe`, `iic0cu1gr3aqn9vt7pvub`,
`ibwyx5gj6p8s9ieqohq88`, `i3u42o3ufy9ye7901yo1n` and the owner is killing them
in the e2b console. Task 14 (hardening) was NOT done first either; the risk was accepted
knowingly, per the card.

**2026-08-13 — Task 3d: a tool call never opens an OAuth flow.** The old
`call_tool` scanned every per-user server with a PUT when it did not recognise a
tool name, which is both the N-request warm path and a redirect the model cannot
complete. Now an unconnected server fails the call with `auth_required` carrying
the setup URL and `retryable=False`. Connecting is a human action through
`connect()`, never a side effect of the model reaching for something.

Also settled here: MCP specs are always `readonly=False`. A remote server does
not tell us whether a tool mutates, and guessing wrong makes the loop run writes
in parallel. And `Smithery.call` stores the blob itself when a result is over
`tools.result_view_cap_chars` — the loop view-caps for the screen but hands the
model whatever the envelope holds, and `ref` only ever comes from the envelope,
so this is where the Task 2 "oversized result loses its tail" gap actually closes
for MCP.

**2026-08-13 — Tasks 7 and 8 pulled forward; the repo is now the new shape.**
Both were scheduled late for one reason: the old path had to keep running. 0c
ended that, and everything still standing was rubble — 18 files across
`state_module`/`computer_module`/`base_module` already imported names deleted in
3d. Deferring further would have preserved dead weight and made every session
re-derive what was safe to touch.

Production 9.9k → **4,682 lines**, against a ~4.5k target. Tests 6,047 → 3,894;
suite green at 231. What survived the cull and why: `tool_module/browser_*`
(Task 9 leashes it), `tool_module/sandbox/` (moved, not integrated),
`harness_module/{jwt_utils,browser_routes}`, `config_module`, `db/`, `frontend/`,
`landing/`.

Two consequences to carry forward. **There is no HTTP server** until Task 4
writes one — `app.py` is gone rather than half-alive. And `harness_module/` is down
to two files, so Open Question 5 (keep the name or rename to `harness/`) costs
nothing to settle now and should be settled before Task 4 puts an api.py in it.

**2026-08-16 — five orphans folded into existing cards; no new task.** A
full-repo audit against contracts found five things contracts requires that no
card owned. Written up on the cards themselves (Task 4 ×3, Task 5, Task 9)
rather than as a new task, because each one belongs to work already scheduled and
a sixth card would have been a second place to look.

What the audit says about them as a group is the part worth keeping: **none was
a coding mistake.** Each is a seam between two cards where neither card's
done-when claimed it. The world tools fell between Task 3 (built the tool layer,
before `projects`/`sessions` existed) and Task 4 (builds the tables, was not
asked for the tools). The prompt fell between Task 2 (takes `messages` already
built) and Task 4 (builds them). The connections surface fell between Task 3
(kept the business logic) and Task 7 (deleted the routes). Browser registration
fell between Task 9 (leashes the tool) and Task 3 (deleted the API it registers
through). The ladder was scoped in prose and given an invariant, but prose is not
a card.

The lesson for the remaining cards: **a done-when that does not say "reachable"
does not make it reachable.** Task 8 said "register it in the manifest" and is
the only integration card that cannot be completed while leaving its code
unwired. Every other card should be read against that bar before it is started.

Two estimates moved with the scope: Task 4 2d → 4d, Task 5 3d → 4d.

All three of the decisions this note left open were settled on 2026-08-16.
`contracts.md` gained the auth and MCP connection endpoint rows (G36/G37). The
app port is whatever `config_module/config.yaml` says: the 1113 frontend fallback
and the README's 1114 are gone, and the Dockerfile's 1112 is left for the
container refactor. And the budget vocabulary collapsed rather than being mapped
— see the note below.

**2026-08-16 — three names for one question, now one.** `sessions.mode`
(`attended|unattended`), the `budgets:` keys (`interactive|worker`) and
`client.Source` (`interactive|background`) were three vocabularies for a single
predicate: is a human waiting? They arrived from different documents at
different times — mode from D5, the budget keys from the config block, `source`
from Task 1 — and nothing had ever consumed all three at once, so nobody
reconciled them. `runner.py` is the first code that would have had to, which is
why this surfaced at the Task 4 boundary and not earlier.

Resolved by deleting a vocabulary, not by writing the mapping down: the budget
keys are now `attended` / `unattended`, so `Budgets.load(mode)` takes what the
caller already has. `mode` could not move — it is a CHECK constraint in
migration 0, the lifecycle table's trigger column, and the name in
`max_unattended_sessions` — and the budget keys were two lines of yaml.

`client.Source` deliberately stays `interactive | background`. `model_module`
must not know what a session is, so that is a layer boundary rather than a third
vocabulary; `agent_module/loop.py:123` derives it from `mode` in one line. The
word `interactive` now appears in exactly three places, all of them that one
concept.

**2026-08-17 — the model is hosted OpenAI for now.** `llm.base_url` is
`https://api.openai.com/v1` and `model_name` is `gpt-4.1-mini`. Self-hosting
needs an NVIDIA GPU and there is not one on the machine doing the work; nothing
in the loop cares, because `client.py` is an `AsyncOpenAI` pointed at
`base_url`. `model_module/run.sh` still launches Qwen3-8B with the right parser
flags for when the box comes back.

Verified end to end through `run_turn`, not through a hand-rolled request: a
tool turn ran get_weather and finish_task and ended `done{completed}`, and a text
turn streamed 41 `content` events with a 1.66s TTFT and ended `done{turn_end}`.
Both tool calls arrived in ONE hop — OpenAI does parallel tool calling, and
`_batch_by_readonly` split them correctly since `finish_task` is not readonly.

Three things this exposed:

- **`llm.api_key` did not exist.** `client.py` falls back to `"-"`, which SGLang
  ignores and a real provider answers with a 401 that names no cause. It is now
  `${OPENAI_API_KEY}` in config, like every other secret.
- **gpt-5.x cannot be selected without a client change.** It rejects
  `max_tokens` in favour of `max_completion_tokens` and pins temperature. The
  4.1 family takes the parameters `client.py` already sends; gpt-4.1-nano and
  gpt-4o-mini were verified working too, if cost matters more than quality.
- **`context_window` is deliberately 128000, under the model's ~1M ceiling.**
  Self-hosted, contracts requires it to MATCH the SGLang launch flag. Hosted, the
  rule inverts to must-not-EXCEED: understating is free and bounds spend, while
  overstating buys a hard failure mid-run.

**Task 0a's measurement does not transfer.** 0/22 malformed calls was Qwen3-8B
with `--tool-call-parser qwen25 --reasoning-parser qwen3`, and its own note says
to re-measure if either changes. This changes both. The direction is favourable
and the repair retry stays regardless, so nothing is blocked — but do not cite
that number as evidence about this configuration. `reasoning` events simply stop
occurring: `reasoning_content` is SGLang's field, and the fold drops reasoning
anyway.

**2026-08-17 — Task 4: three deviations, and one of them was a latency bug.**

**The sink writes behind the loop, and the flush-window config key is gone.**
The first cut coalesced streamed `content` events on a timer before appending.
It did nothing, because the producer is serialized behind the consumer: the loop
yields a chunk, awaits the append, and only then produces the next one. So the
buffer never held more than one chunk, and every delta paid a full round trip —
about 150ms to the Supabase primary, which is thirty seconds of pure latency on
a 200-chunk reply. `Sink.emit` is now synchronous: it queues the event and a
writer task drains it, merging whatever piled up behind an in-flight append.
Coalescing became adaptive rather than tuned, so `harness.content_flush_ms` was
removed rather than defaulted. This is what contracts already called
write-behind persistence; the timer version was write-through wearing its name.

A consequence worth knowing: a failed append is now discovered one event late.
The writer records it and the drive loop raises on the next `emit`, so the run
still halts rather than continuing unrecorded — one event of lag is the price of
keeping Postgres off the token path.

**A Task 4 runner exists.** The card says "touch `api.py`", but chat transcripts
riding `session_events` means the second message of a conversation is rebuilt
from the log, and that is the fold. `runner.py` is here with the attended half
(fold · drive · translate · cancel · verify-on-wake); Task 5 adds leases,
wake-at-cursor, the approve path and the ladder. Its seams are marked in the
module docstring.

**Found by a test, and it was real:** a cancel landing during turn SETUP — the
wake repair, the fold, the manifest build, all network awaits — left the row
`running` with no task behind it, and `start()` refuses to touch a running
session, so only a restart cleared it. Every exit path now records a terminal,
with or without a sink.

**`jwt_utils` was rewritten around two secrets**, replacing `ARK_JWT_SECRET`.
`SUPABASE_JWT_SECRET` verifies a token somebody else issued and never signs.
`ARK_SESSION_SECRET` signs the cookie we issue, reachable only from
`POST /auth/session` after that verification returns. One secret for both would
have meant a token we verify and a token we mint being interchangeable, which is
the shape `demo-login` had. Supabase stamps `aud=authenticated` and PyJWT
refuses an audience the caller did not ask for, so it is named in config rather
than switched off.

**Approvals, attended:** `mcp_*` tools require approval by default (2026-08-16)
and Task 6 owns the park, so Task 4 had to answer them somehow. Attended
auto-approves, per G38 — the human is watching the stream and can cancel
mid-call — behind `approvals.attended_auto_approve` so it is a config edit, not
a code change. Unattended refuses and logs, rather than guessing yes.

**Config cleanup in the same commit:** `memory:`, `embedding:` and `state:` were
blocks describing deleted modules, and `app.system_prompt` was the pre-redesign
ARK prompt that `prompts.py` now owns. All four are gone.
`browser_routes.py:27` was the only live reader of any of them.

**2026-08-18 — the store landed (8.1-8.6b). What a fresh session needs to know.**

The agent's files now live in object storage we own, with the tree in Postgres
and the sandbox disk demoted to a cache (D27). All ten cards are done.

Memory came back with 8.8, in the store rather than a module of its own:
`memory_files` keyed by user, four tools, Postgres FTS, and `MEMORY.md` injected
into the system prompt at fold. mem0 stays deleted, and so does everything
automatic about it — no extraction, no embeddings, no per-turn retrieval. Whether
memory may ever be read from inside a sandbox is D30 and open; it does not mount
today.

Where things are: `harness_module/store.py` is blobs and trees and knows nothing
about e2b; `harness_module/workspace.py` fills and empties the sandbox cache;
`session_claims` rows say what a session may see; `runner._Sink._lease` claims a
slot in the user's sandbox pool plus `project:{id}` per write claim,
materializes, and flushes before it gives either back. The sandbox is not leased:
one box per session, `session_sandboxes` holding both its handle and its slot.

Five things a reader would otherwise re-derive:

- **The sandbox is asked nothing about itself that is not verified.** The first
  cut kept a manifest file in the sandbox recording what had been materialized,
  and used it to decide what to delete. That record is stale the moment a flush
  commits, so a file another session deleted was never removed from a warm
  sandbox and the next flush put it back in the tree — a deletion undone
  permanently. Both directions now hash the files on disk and compare against
  the tree. The manifest was deleted rather than fixed: a second source of truth
  about the same bytes is what created the hole.
- **A flush may only commit against a workspace that proves it was
  materialized.** Hashing the disk means an empty disk reads as "every file was
  deleted", so a box that died between materialize and flush had its emptiness
  committed: `commit_entries` replaced the project's tree with no rows, logged as
  a clean flush. `materialize` now seals the box with a nonce recorded against
  the session's slot and `flush` refuses without it. The proof is deliberately
  about the box and not its contents, so a session that really did delete
  everything still commits that.
- **A slot is capacity with an expiry, and the row comes before the box.** The
  `sandbox:{user}` lease it replaced expired on its own; a bare pool row did not,
  so a crashed process burned one of the user's boxes forever. Slots carry
  `expires_at`, renewed on every call into the box, reclaimed (and their boxes
  killed) by the next claimer and by a startup sweep. `get_or_create` refuses a
  session with no slot and kills a box whose handle it cannot record: a crash
  leaves a reclaimable row, never a box nothing knows about.
- **Blobs first, rows last, and `commit_entries` is why a flush is cheap.** A
  flush computes hashes in the sandbox, uploads only changed bytes, and writes
  every row from a hash. Both commit paths verify the blobs exist before any row
  moves, so the tree cannot come to point at bytes that are not there.
- **Supabase reports a missing object as HTTP 400** with a body saying 404. A
  mock built on the assumption of a clean 404 passed while every read of an
  absent blob raised. Live tests against the real bucket found it; the same
  pattern found the e2b column mismatch in Task 8a. For a vendor boundary, a
  fake tests the assumption, not the vendor.

Setup facts worth carrying: the store needs an `sb_secret_` key, not the
publishable one and not the legacy `service_role` JWT. `SUPABASE_URL` is derived
from `DB_URL` and need not be set. `.env` still carries `ARK_JWT_SECRET`, which
Task 4 replaced — the server will not start until `ARK_SESSION_SECRET` is set
(`SUPABASE_JWT_SECRET` is now optional, since tokens verify against the
project's published ES256 key). On a Python whose platform tag predates the
published wheels, `cryptography` must be installed with `--only-binary=:all:`
or pip tries to build it from Rust source.

**2026-08-17 — Task 4 reviewed, and the transcript was the weak half.** Twelve
findings, worked in three batches; the rest carded onto Tasks 5, 9 and LG-1
rather than fixed. What is worth carrying forward is the shape of the failures:
four of the five in batch 1 ended in a session that could never be loaded again,
and none of them was reachable by reading a single file. Each needed the log
format, the fold and the chat template held in mind at once.

- **A tool_result closed a call it did not answer.** `_OPEN_CALLS` bounded the
  call side to the current run but left the result side unbounded, and the loop
  minted `call_0_1` as the first synthetic id of EVERY turn (`seen_ids` is
  per-turn). So turn 2's call was matched by turn 1's result, the legitimate
  result was refused, the run died, and `close_dangling` could not see the call
  either. Fixed both halves: `r.seq > c.seq`, and a uuid suffix on synthetic ids.
- **A user message stranded a call.** `post_message` appended the user event
  before `_drive` ran the wake repair, so the log read `tool_call / user /
  tool_result` and the fold emitted `assistant(tool_calls) → user → tool`, which
  OpenAI rejects. Fixed at both ends: the endpoint closes dangling calls first,
  and the fold holds a user event that lands mid-call until the results close.
  The second half is what covers steering, which no write-side ordering can
  prevent. It is view-only, so replay stays deterministic.
- **The advisory lock keyed off the caller's spelling of the session id.**
  Postgres compares uuids case-insensitively and hashes text exactly, so an
  uppercased id in a URL took a different lock and serialized against nothing —
  silently un-doing the one guarantee that append exists to give.
- **The consent gate was unreachable.** `registry.dispatch` returned before
  `envelope.execute` for every `mcp_*` name, and `execute` is where
  `requires_approval` is read. Every MCP tool ran ungated and `_Sink._approve`
  was dead code, which made the Task 4 note about attended auto-approval false
  when it was written. The mcp branch now runs through `execute` like anything
  else.

Two bugs were found by writing the test rather than by reading. A cancel landing
during turn SETUP (fold, manifest, wake repair — all network awaits) left the
row `running` with no task, which `start` then refuses forever. And a `_finish`
interrupted after appending its `done` but before the transition would later be
completed by an abort carrying a DIFFERENT reason, so the transcript said
`turn_end` beside a status saying `failed`; the first reason to reach `_finish`
now owns the ending.

**2026-08-16 — the database moved. 0c re-applied to a new Supabase project.**
The target is now project **`sbtbbytesjobdpmqojlr`** (`db.sbtbbytesjobdpmqojlr.
supabase.co:5432`, PostgreSQL 17.6). The ark box database that 0c ran against on
2026-08-13 is no longer the backend's database, so every row count and table
inventory recorded in the 08-13 notes describes a machine we have moved off.

Migration 0 applied cleanly on the first run: all 12 contract tables created,
nothing missing. Verified afterwards — the three CHECK constraints
(`sessions_mode_check`, `sessions_status_check`, `session_events_kind_check`) and
both `approvals` indexes including `idx_approvals_one_open_per_call`, the partial
unique that makes a double-park impossible.

The whole teardown half was a no-op, because the project was clean: no `vecs`
schema, no mem0 `memories` view, no `user_oauth_tokens`, no old chain. The two
holes that only the real ark database exposed on 08-13 (see that note) could not
have been found here. Keep the teardown anyway — it is what makes the migration
safe to run against a database that HAS that history, and the drops are all
`IF EXISTS`.

**What did NOT come across, and this is the part to remember:**

- **The 8 waitlist signups.** Migration 0 preserves a `waitlist` table, it does
  not create one, so the rows do not travel. This project has its own `waitlist`
  with **3 rows** and they are unrelated to the ark box's 8. If those 8 are real
  signups, they are still on the old host and nobody is reading them.
- **`repeat_tasks`.** Absent here. It was out of scope anyway (watching was
  scrapped) but the "carried over untouched" line in the 0c card is now only half
  true.

Settled by running it, not by argument: the Vercel waitlist function
authenticates as **`waitlist_writer`**, not `anon`. The `waitlist_insert` RLS
policy and that role's INSERT grant were intact before and after the migration,
which is why live signups never noticed. So `003_waitlist_anon_insert.sql` on
`origin/vercel-landing` is NOT needed here, and should not be applied — the
role-based grant is the stricter of the two. G-note for whoever revisits: do not
"fix" this by granting anon.

Two operational facts about this database, both written into `.env.example`
because they cost an hour each to rediscover:

- **The password must be percent-encoded.** Supabase passwords routinely contain
  `@`, and a raw `@` does not error — the DSN silently reparses and everything
  after it becomes the host.
- **`db.<ref>.supabase.co` resolves AAAA-only.** Fine from a dev machine; a
  default Docker bridge and most CI runners are IPv4-only and fail with "Network
  is unreachable". The transaction pooler is the way out, and `db/pool.py`'s
  `statement_cache_size=0` is already exactly what the pooler requires.

**Consequence for Task 14 (hardening).** That card is P0 and blocks 0c and Task 4 because the
ark host's integrity was unestablished. The database half of that concern is now
moot — this is a fresh managed project, not the box in question. The host-side
items in `~/dev/vulnerabilities.md` stand on their own and are not addressed by
the move. Task 14 should be re-scoped to what is actually left rather than
carried as a blanket blocker on Task 4.

Side effect worth knowing: with a live database reachable from `.env`, the 33
`tests/test_smithery.py` cases that skip without one now run. The suite went from
222 passing with 33 skipped to 250 passing with none — so Task 3d had, until
now, no verified coverage on any machine without a database.

---

**2026-08-18 — Task 8.9(i): the FUSE probe. It works, so the lazy-mount rung is
open.** One sandbox on the `base` template (`scripts/fuse_probe.py`, re-runnable):

| | |
|---|---|
| kernel | Linux 6.1.158, Debian 12 (bookworm) |
| `/dev/fuse` | present, `crw-rw-rw-` |
| `fusermount` / `fusermount3` | both installed |
| rclone | absent, installs from apt |
| `rclone mount` | **mounted and read back through the mount** |
| mount table | `/tmp/src on /tmp/mnt type fuse.rclone` |
| running as | uid 1000, not root |

**Consequence.** The day eager materialize gets too slow — a project too large
to copy at every lease acquire — mounting the store instead of copying it is an
option we can take on e2b, without changing vendor or self-hosting the sandbox
layer. That was the fork this probe existed to close, and it closed the cheap
way. Nothing changes today: eager sync is faster for the project sizes we have,
and a mount would put a network dependency inside the box on the read path.

What the probe did NOT measure, on purpose: object-store latency through a FUSE
mount. It mounts rclone's local backend, because the question was whether
`/dev/fuse` works in the box and no store credential may enter a sandbox (D28).
Latency needs a real workload on a real bucket, and belongs to the card that
actually builds the mount.

**2026-08-18 — Task 8.9(ii): snapshots, and when to revisit any of this.**
`project_snapshots` + `snapshot_files` (migration 0007) hold a copy of a
project's tree rows; `store.snapshot_project` / `restore_snapshot` /
`prune_snapshots` are the calls, and `scripts/snapshot_store.py` is the timer
job. A snapshot copies rows and no bytes, which is only true because blobs are
immutable and nothing deletes them — **a blob GC, if one is ever written, has to
walk snapshots as well as trees, or restoring becomes a lie.**

The triggers that should send someone back to the storage design, written down
while they are still hypothetical:

- **materialize or flush taking seconds on a real project.** The first sign that
  eager sync has outgrown itself. The probe above says the answer is available.
- **A project too big to copy at every lease acquire.** The same trigger from the
  other side, and the fork between FUSE-on-e2b and self-hosting the sandbox layer.
- **Sandbox-hours crossing the cost of owned metal.** e2b is rented compute; at
  some volume it stops being the cheap answer, and the store being ours already
  is what makes that a migration rather than a rewrite.
- **A customer who forbids third-party VMs, even ephemeral ones.** Not a
  performance trigger but a contractual one, and it lands on the same fork.


## Arcade gateway — live test facts (2026-08-21, pre-11.10)

Tested against the real gateway (`https://api.arcade.dev/mcp/gw_3IErRNEhqI6uH9fdquWfocoxdcF`,
project "buddy", auth = API-key headers) from a scratch client, before any
harness code. What the wire actually does:

- **Transport is Streamable HTTP MCP with sessions.** A bare `tools/list`
  is refused (`400 missing Mcp-Session-Id header`). The client must
  `initialize` first (the response HEADER carries `Mcp-Session-Id`), send
  `notifications/initialized`, then include that header on every request.
  Server negotiated protocol `2025-11-25`. The 11.10 client owns this
  handshake — per user, since the session is minted under the caller's
  `Arcade-User-Id`.
- **`Accept: application/json, text/event-stream` is required**; responses
  came back as plain JSON in this test but the SSE accept must be offered.
- **Tools come back flat, prefixed by app** — 100 tools for the current
  selection: `Gmail_*` (30), `Github_*` (43), `Linear_*` (13),
  `GoogleSlides_*` (13), plus `Arcade_ListApps` (the gateway's own
  meta-tool: per-caller connected/not-connected state — likely useful for
  the settings panel). "Server" in our config = prefix group, not URL.
- **Per-user identity WORKS.** `Gmail_ListEmails` as `Arcade-User-Id:
  alice-test` and as `bob-test` returned two challenges with DIFFERENT
  `state` tokens — separate pending OAuth sessions per user. (The old
  miswiring returned the same state every time because no user was
  attached.) Grant-completion round trip still pending a human clicking a
  consent link.
- **The pre-consent challenge arrives as a SUCCESSFUL tool result** carrying
  `authorization_url` + `llm_instructions` ("show this link to the user").
  This is the as-wired evidence 11.10 deferred to: the choice between
  panel-first connect and an envelope intercept is now live, not
  hypothetical. Also: the gateway's own `initialize` instructions block
  explicitly says to treat its app list as untrusted data — same posture we
  need toward `llm_instructions`.
- The initialize instructions enumerate "available apps" per caller with
  connected state — the gateway knows per-user connection status, so the
  settings panel can render real per-service connect state from
  `Arcade_ListApps` instead of guessing.

Two build-time notes added 2026-08-21, pre-11.10: (1) **panel-first consent
initiation** — Arcade mints `authorization_url` in response to a tool call,
so the settings panel needs a way to obtain the link for its popup: likely
a cheap read-only probe call per service (capture the challenge, never show
it to the model), unless Arcade offers a dedicated auth-initiation call —
check before inventing. (2) **Stale rows** — `user_connections` /
`shared_connections` still hold rows keyed to the dead `*.run.tools` URLs;
the Smithery teardown should sweep them, not strand them.

**CORRECTION (2026-08-21, late): `tools/list` is PAGINATED and the earlier
"100 tools" reading was one page.** The gateway serves pages of 100 with a
`nextCursor`; following it yields the full roster, which matches the
dashboard exactly — 169 tools: Gmail 30, Github 43, Linear 46,
MicrosoftOutlookMail 30, Notion 10, GoogleCalendar 8, GoogleSearch 1,
Arcade 1 (meta). The 11.10 client MUST loop until no cursor comes back or
it silently ships 100/169 and Notion + Outlook "don't exist" — this is the
SSE-backlog LIMIT-1000 bug shape again; same law: page until the read
returns no cursor. Prefix for Outlook is `MicrosoftOutlookMail`. The
`Arcade` prefix (ListApps) is harness plumbing: never in the model's
manifest, used by the settings panel only. The dashboard's per-app counts
were right all along; probe with pagination before ever doubting it again.

**2026-08-21 — Task 11.10: the gateway's own app list cannot answer "is this
service connected".** The card had `Arcade_ListApps` supplying per-service
connect state for the settings panel. Measured, it reports PROVIDERS:
`arcade-github`, `arcade-google`, `arcade-linear`, `arcade-microsoft`,
`arcade-notion` — one `arcade-google` row covering Gmail, Google Calendar AND
Google Search. Meanwhile `authorize` for Gmail asks only for `gmail.readonly`
with `include_granted_scopes=true`. So connecting Gmail would have rendered
Google Calendar connected in the panel and let it be toggled into a session,
while every Calendar call still challenged. Name-matching cannot fix it; they
are genuinely the same app id.

**`POST /v1/tools/authorize` is the status read instead** — the same call that
mints the consent link. `completed` means this service's scopes are granted,
`pending` carries the url the popup opens; it invokes nothing, so asking is
free, and it is scope-aware, so it answers per service. One call per connector,
concurrent, on panel open. `ListApps` is used nowhere and is filtered out of the
model's manifest with the rest of the `Arcade` prefix.

Three more measured facts the client is built on. (1) `authorize` accepts BOTH
spellings of a tool name — `Gmail.ListEmails` and `Gmail_ListEmails` — with
byte-identical answers, so the client passes the gateway's own spelling through
rather than translating. (2) Revoking is per PROVIDER ACCOUNT
(`DELETE /v1/admin/user_connections/{id}`), so disconnecting Gmail disconnects
Google Calendar with it; there is no narrower revoke, so the panel names the
siblings and asks twice, and `DELETE /connections/{server}` answers with what
actually went rather than 204. (3) A pending authorization is idempotent: asking
twice returned the same `id` and the same `state`.

Also settled here: the durable key is the PREFIX, not the gateway url. The slug
is infrastructure and can be recreated, while the grants live Arcade-side keyed
by user id and survive that untouched — so a row's honest meaning is "user X
connected Gmail". `0019_arcade_connections.sql` renames `mcp_url` to `server` in
`user_connections` and `session_tools`, drops `connection_id`/`tools_cache`
(no connection object to name; the roster is not a per-user fact), drops
`shared_connections`, and deletes every row: all 548 `user_connections` rows and
all 4 `session_tools` rows were `*.run.tools`, so there was nothing to carry.

**Default-provider findings (2026-08-21, three live attempts — build on
custom providers, not this):** Arcade's default providers are dev-only:
their flow fronts an arcade.dev account picker (the "Arcade user
verifier", requiring Arcade project membership) BEFORE the provider
consent, and across three attempts (raw link; fresh link; fresh link with
Arcade sign-in completed) NO grant ever landed —
`/v1/admin/user_connections` stayed at total_count 0. Conclusion: the
production path (own OAuth app per provider + custom user verifier route)
is also the only path that works at all for non-Arcade users; do not spend
more time on the default providers. `authorize` remains idempotent while
pending (same `ar_` id and state re-returned). Google restricted-scope
note: unverified own-app = 100 test users max; verification review gates
GA.
