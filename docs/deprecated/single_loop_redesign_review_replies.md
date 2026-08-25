# Single-Loop Redesign — Draft Replies to Review Comments

Draft responses to the 15 unresolved comments on the Notion spec
*"Single-Loop Redesign — Native Tool Calling"*.

- **Spec:** https://app.notion.com/p/3920e72f16c6811fac12f572f3a8441e
- **Comments:** 15, all unresolved, left 2026-07-26 21:40–21:59 UTC
- **Drafted:** 2026-07-26
- **Status:** not posted to Notion

Every claim below is grounded in a read-only pass over the repo at `28e46b3`
(branch `landing`). File:line citations are inline.

---

## Task 2 — Core loop

> **Comment:** "What is the IO of the loop, how do we make streaming a first class
> primitive instead of an afterthought"

Today there are three shapes: `ComputerAgent.run(prompt, step_cap) -> dict`
(`computer_module/agent.py:96`), `Agent.step(messages, user_id) -> StateOutput | None`
(`agent_module/agent.py:348`), and `step_stream` yielding untyped dicts (`:445`).

Proposed:

```
async def run_turn(messages, tools, budgets, *, model, dispatch, session_id) -> AsyncIterator[LoopEvent]
```

`LoopEvent = ContentDelta | ToolStart | ToolProgress | ToolResult | NeedsInput | Done(reason, summary, outputs)`.

There is no return value — `run()`'s dict and `step()`'s `StateOutput` both collapse
into `Done`; a caller wanting the dict reduces the stream.

Streaming as substrate means three things:

1. The loop's only text source is the streaming completion. Text deltas yield on
   arrival; tool-call deltas off the *same* chunk accumulate in an index-keyed buffer
   and surface as `ToolStart` when the block closes. One iterator carries both because
   the wire already interleaves them — that is also the mechanism resilience contract
   #7 Phase 2 needs.
2. Tool progress originates *below* the loop via a sync callback
   (`computer_module/tools.py:205`), so it cannot be yielded from inside `dispatch`.
   `run_turn` needs an internal `asyncio.Queue` that the model stream and N concurrent
   tool tasks both feed. That queue is what makes concurrent readonly tools
   expressible at all.
3. Every consumer subscribes to the same stream — SSE handler, task runner with no
   client attached, session-log appender — fanned out through a per-session broker
   with bounded per-subscriber queues, so a disconnected browser cannot stall the model.

**Correction to the spec.** `make_llm_call(stream=True)` raises `NotImplementedError`
(`model_module/ArkModelNew.py:128`). The real `generate_stream` (`:199`) has **zero
production callers** — tests only. So Task 1's "streaming path has no retries" is
describing dead code, and Task 4 is wiring a streaming client in for the first time,
not fixing an existing one. Scope change, not a bug fix.

Two further mismatches worth folding into the task text:

- The computer pathway isn't fake-streamed, it isn't streamed at all — its transport
  is a 1s Postgres poll (`computer_module/computer_router.py:137`) fed by a
  *synchronous* DB write on the event loop (`runner.py:45`). Extracting `run_turn`
  does not make that surface live unless the transport changes with it.
- `onDelta` currently receives the *cumulative* reply, not a delta
  (`frontend/seed.jsx:341`), and no frontend consumer exists for
  `tool_start`/`tool_result` at all. Task 2 should say those events are recorded and
  subscribable but not rendered until Task 10.

---

## Task 5a — `finish_task` vs token matching

> **Comment:** "Should task completion require a tool call. What about other
> intermediary reasoning steps. cant we do token based matching?"

We already run that experiment in-repo. `_last_tool_was_error`
(`state_module/agent_executor/state_executor.py:38`) sniffs for content starting with
`"tool "` and containing `" -> "` — but tool results and model prose land in the *same*
`AIMessage` channel (`agent_module/agent.py:394`). A 7B writing ``tool `send` -> ok``
in prose is already indistinguishable from a real result. The same collision applies to
the ` ```ark-plan ` fence (`state_module/agent_buddy/state_plan.py:177`), regex-extracted
by the frontend. A sentinel inside a reasoning aside or a quoted plan fails *open* — it
grants completion.

The three options fail differently:

| Option | Failure mode |
|---|---|
| Absence of tool calls | A stuck model looks identical to a finished one. The current bug in a new costume. |
| Sentinel / token matching | Forgeable from prose; same channel; fails open. |
| Explicit `finish_task` | Arrives in a separate field of the response object. Prose cannot forge it. |

Cost is one hop. The failure it prevents is live today:

1. `agent_module/agent.py:378` sets `terminal_reason = TerminalReason.max_steps` on
   budget exhaustion.
2. `terminal_reason` is **read nowhere outside `agent.py`**.
3. `step()` returns `last_state_output` (`:443`), usually the tool state's
   `completion_signal="complete"` (`state_tool.py:106`), which carries no
   `all_steps_done`.
4. `base_module/task_runner.py:213` falls back to `completion_signal == "complete"` and
   calls `mark_task_completed` at `:216` with `step_idx < len(plan_steps)`.

Green task, unexecuted steps.

On "intermediary reasoning": nothing changes. Intermediate text is free, ungated, needs
no tool. Only the terminal claim is gated.

**Where token matching is defensible:** as a fail-closed *negative* guard, the way
`_TOOL_ERROR_SIGNALS` already blocks advance (`state_executor.py:251`). Never as a
positive completion claim.

**Test gap:** `tests/test_agent.py:262` asserts `terminal_reason == max_steps` at the
agent level, but nothing asserts the *task* is not marked completed on that path. That
is what `test_failed_task_never_reports_completed` closes.

---

## Task 5b — Step cursor and replanning

> **Comment:** "How do we define a step cursor? Are we not reinventing the state
> machine? We must now assume the model should not deviate from the plan, maybe a
> replanning phase is needed. Recursive."

They are different objects. What is being deleted is *named states with LLM-gated
transitions*: four states in `agent_executor/graph.yaml`, routers mapping signals to
state names, and `choose_transition` (`agent.py:428`) asking the model which state comes
next when no router matches. That is the part that hallucinates.

A cursor over an append-only log has no transitions and gates nothing. `task_events` is
already that log — `BIGSERIAL event_id`, append-only
(`db/migrations/0003_subagent_runtime.sql:36`). The cursor is a fold over it:
`1 + max(payload.step_idx)` across `kind='step_complete'`. Derived, discardable,
recomputable. It never decides anything.

Today `step_idx` is a plain int on the Agent (`agent.py:113`), zeroed at
`task_runner.py:175`. It advances in exactly one place — `state_executor.py:272`, on
`action=advance`, gated by the tool-error check. `state_tool.py:102` deliberately does
not advance; `state_approval.py:163` deliberately does not either (approval is
permission, not proof — confirmed in code). It is never persisted. Meanwhile the
transcript *is* durable (`conversation_context` keyed by `session_id`, pinned at
`task_runner.py:114`). So on restart `sweep_orphans` respawns with `step_idx=0` and
re-executes step 1 against a transcript that already contains its results.

**Task 5 is not adding state. It is persisting the one integer the log already implies.**

### The deviation concern is the real issue

The plan is frozen at `POST /tasks` (`base_module/tasks.py:191`) and three separate
prompts forbid replanning (`task_runner.py:50`, `state_executor.py:164`,
`graph.yaml:2-3`). Deviation routes to `ask_human` — which is why tasks stall in
`awaiting_approval`.

**Recommendation:** make the plan a **versioned artifact**, and replanning **one tool
call in the same loop** — `revise_plan(steps, reason)` appends a `plan_revised` event
and rewrites the artifact in place. The cursor becomes `(plan_version, step_idx)`; a
revision preserves the completed prefix and resets only from the divergence point.

**On "Recursive" — reject it.** A nested loop reintroduces exactly the budget conflation
Task 5 removes; one counter cannot cover both levels. One loop, one budget, replans
consume it like any other hop. Cap replans per task (3) and count them separately,
because runaway replanning is the new failure mode and it must terminate as `failed`,
not exhaust into a green.

**Constraint to flag:** `required_tools` is a hard allow-list fixed at spawn
(`task_runner.py:158`; `tool_module/scoped.py:50` raises `PermissionError`). A revised
plan introducing an unlisted tool hard-fails. Restrict revisions to the existing allowed
set; anything wider escalates to `ask_human`.

---

## Task 6a — Why 2s, why 24h

> **Comment:** "Why @2s and why is it up to 24hrs"

Neither number is derived from anything. Both are inline `or` fallbacks in a config
lookup — `config.get("approval.poll_interval_seconds") or 2.0` and `or 60 * 60 * 24`
(`state_module/agent_executor/state_approval.py:45-57`) — and there is no `approval.*`
key anywhere in `config_module/config.yaml`, so the fallbacks are what always run.
`86400` appears again as a bare literal in the computer agent's ask path
(`computer_module/agent.py:220`). They were picked to feel responsive and to feel long
enough. That is the whole derivation, and that is the point the comment is making.

**What they cost.** The loop issues two queries per tick — `get_approval` then `get_task`
(`state_approval.py:112,116`) — so one approval riding the full window is ~43,200 ticks
and ~86,400 queries. Nothing backs off. There is no pool: `task_store._connect()` opens
a fresh psycopg2 connection per call (`base_module/task_store.py:19`), so that is
~86,400 connect+auth cycles, and concurrent approvals multiply it linearly against
`max_connections`.

**One precision note:** the sleep is `await asyncio.sleep`, so the loop is not pinned for
the full 2s. The queries are the problem — blocking psycopg2 with no executor, so every
tick stalls the entire event loop for two round trips, not just this task.

The fix is not a better interval or a longer deadline. Under Task 6 both numbers stop
existing: nothing waits, so there is no interval to tune, and a parked task is just a
`task_approvals` row plus a task in `awaiting_approval` with no process attached —
leaving it parked costs nothing, so there is nothing to time out.

---

## Task 6b — Reminder notifications

> **Comment:** "Why are we firing reminder notifications"

Fair challenge.

**For:** with the timeout gone, a parked task waits forever by default. Today a
forgotten approval at least dies loudly at 24h — it logs `approval_timeout`, returns an
error output, and because `step_idx` never advanced, `all_steps_done` is false and the
task lands in `failed` (`task_runner.py:219`). Remove the timeout and nothing surfaces a
stuck approval on its own.

**Against, and this wins:** the pull surface already exists. `list_pending_approvals`
(`task_store.py:328`) backs `GET /tasks/approvals/pending` (`tasks.py:417`), which powers
the desk's Pending Approvals panel. Tasks 10–11 generalize exactly that into the
attention list. A push at 1h is the same species of magic number as the 2s poll —
nothing makes 1h right, and the first question is why not 15m, or a second at 4h. It is
also the weakest possible safety net: `send_dm` silently no-ops for any user who has not
linked Slack (`tool_module/slack_notify.py:42`), so the guarantee it appears to offer is
not one.

**Recommendation:** cut the reminder from Task 6. Keep the single DM at request time —
`state_approval.py:96` already sends it, and that one is justified because it is tied to
a real event. Let the attention surface carry staleness, and have it render age so an
approval sitting a day is visibly the top item. If we still want a nudge after 10–11
ship, add it then against real response-time data instead of a guess.

---

## Task 8a — Browser as a first-class primitive

> **Comment:** "How do we make browser first class primitives"

First-class means interface parity, not inlining. The browser stays a delegated inner
loop wrapping `browser_use`; what makes it first-class is that four seams look identical
to a Smithery tool's:

1. **One tool schema in the manifest.** Already half-true: `register_browser_tool` calls
   the same `register_local_tool` path as any tool (`tool_module/browser_tool.py:511`),
   surfacing under the synthetic server `"local"` in `list_all_tools()`
   (`tool_module/smithery.py:493`). What is *not* true is uniform reach — see the blocker
   below.
2. **One result envelope.** Today `BrowserToolError` propagates out of `_handler`
   (`browser_tool.py:503`); a 180s wall-clock abort (`:483`) should arrive as
   `{ok: false, error_kind: timeout}` the model can pivot on, not as a task failure.
3. **One event stream.** Right now the screencast goes to a separate broker and its own
   SSE route (`browser_tool.py:226`, `base_module/browser_routes.py:37`), disjoint from
   task events.
4. **Budgets passed per call**, not read from `BROWSER_USE_*` env at call time
   (`browser_tool.py:415`).

The tension the comment circles is real: a tool that runs its own 25-step, 180-second
LLM loop is not honestly "one tool call." The event stream is what resolves it — the
inner loop's steps stream into the caller's log as progress events, so the tool stays
*atomic in the transcript* (one call, one result, no interleaved reasoning) while being
*observable mid-flight* in Looking Glass. Atomic to the model, streaming to the human.

**Blocker to state in the task:** the computer agent is spawned with `tool_manager=None`
(`computer_module/computer_router.py:62`), so a computer task today has **no MCP tools
and no browser at all**. Task 8's acceptance test is net-new capability, not a
tightening of existing behavior.

---

## Task 8b — "subagent manifest" vs "agent manifest"

> **Comment:** "Are we considering this the subagent manifest or the agent manifest?"

Neither — and this is our editorial defect, not a question for you to resolve. The
Proposed Approach already retires the term (spec line 111: *"The word 'subagent' is
retired: there are only sessions, all peers running the same loop; capability differs by
role, not tier"*), and the manifest table is keyed by **session role** (Interactive /
Worker / Browser inner loop). The correct phrase is "the Worker session manifest," or
generically "the role manifest." Task 8 is stale text that predates that decision.

Exact edits:

| Line | Current | Change to |
|---|---|---|
| 216 (title) | "Unify computer into the subagent manifest" | "…into the worker session manifest" |
| 218 (Problem) | "any subagent reaches for mid-task" | "any worker session reaches for mid-task" |
| 219 (Done when) | "the unified subagent manifest" | "the Worker role manifest" |
| 220 (Touch point) | "subagent manifest config" | "session role manifest config" |

The same pass should fix **line 65**, which is worse: the heading "One subagent, three
capability families" and "A spawned subagent's manifest" sit in the same section that
retires the word, and its claim that the manifest "includes all three families"
contradicts the by-role table where Interactive gets no sandbox or browser.
`tool_module/scoped.py:2` also says "subagent" in its docstring — fold that into Task 8's
diff.

---

## Task 8c — What lazy provisioning implies

> **Comment:** "What does lazy provisioning imply?"

Six consequences, in rough order of how much design they cost:

1. **Mechanically it is nearly free.** Every sandbox operation already routes through
   `get_or_create` (`computer_module/sandbox.py:162,179,183,188`), so `agent.py:108` is a
   redundant eager warm-up — deleting that line makes provisioning lazy by construction.
   All the work is in the consequences below.
2. **Latency moves inside a tool result.** Measured e2b numbers: create 0.21s, resume
   0.36s (`docs/specs/COMPUTER_SPEC.md:399`). So the first `run_command` gets ~0.2–0.4s
   slower and nothing else changes. Caveat: that is the bare `base` template; a custom
   template or Task 11's project-file mount is what actually sets this budget.
3. **Provisioning errors become tool errors, not task failures.** `dispatch()` already
   swallows everything into `"ERROR running {name}: {e}"` (`computer_module/tools.py:203`),
   so this happens by default — but the text is written for a human. Needs
   `error_kind: upstream_error, retryable: true` so an e2b quota or outage does not read
   to the model as a malformed command it should "fix" by rewriting the shell string.
4. **Lifetime/teardown becomes conditional.** `runner.py:123` pauses the sandbox at task
   end; with lazy provisioning that must no-op when nothing was ever booted. Worse, a
   sandbox booted at hop 12 of a long browser-heavy task can be reaped by the 300s
   `timeout_seconds` (`config_module/config.yaml:59`) mid-run — `idle_timeout_seconds: 900`
   is documented but unimplemented (`docs/specs/UNSAFE_DECISIONS.md:143`).
5. **Task 11 mounting has to move.** If project files mount "at provision" and provision
   is lazy, the mount hook must live *inside* `get_or_create`, not at task start —
   otherwise the first `read_file` hits an empty box.
6. **Idempotency is safe in-process, not across processes.** `get_or_create` holds a
   per-user `asyncio.Lock` (`sandbox.py:109`), so two concurrent readonly calls collapse
   to one boot. Across two workers there is no lock: both call `Sandbox.create`, the
   `ON CONFLICT (user_id)` upsert (`sandbox.py:65`) picks a winner, and the loser's
   container leaks. Resilience contract 7 (concurrent readonly execution) is what makes
   this reachable.

---

## Task 8d — "We dont need an if we need to be concrete"

> **Comment:** "We dont need an if we need to be concrete"
> (on *"endpoints kept as thin shims if the frontend needs them"*)

Agreed, and the code settles it. The frontend calls exactly three computer routes:
`POST /computer/tasks` (`frontend/seed.jsx:247`), `GET /computer/files` (`:274`),
`GET /computer/file` (`:281`). It calls none of the others — task list, detail, events
and status all come from the shared `/tasks` and `/tasks/{id}/events`
(`frontend/app.jsx:131`, `seed.jsx:209`), because computer tasks are already
`agent_kind='computer'` rows in `tasks` (`computer_module/store.py:31,77`).

The concrete call:

- **Delete outright:** `GET /computer/tasks` (`computer_router.py:67`),
  `GET /computer/tasks/{id}` (`:74`), `GET /computer/tasks/{id}/events` (`:100`),
  `GET /computer/tasks/{id}/stream` (`:123`). Their only consumer is
  `computer_module/test_endpoints_live.py`, a manual smoke script — delete it too.
- **Keep, renamed and moved out of `computer_module`:** `GET /computer/files` (`:170`)
  and `GET /computer/file` (`:185`). These are sandbox filesystem reads, not task
  machinery, and Task 11's project-files tab needs them. They become `/sandbox/files`
  and `/sandbox/file`, frontend updated in the same PR.
- **Delete `POST /computer/tasks`; dispatch goes to `POST /tasks`.** Keeping it *is* the
  upfront "is this a computer task?" typing Task 8 exists to remove. This also deletes
  the frontend's `target === "computer"` branch (`app.jsx:250`,
  `frontend/components.jsx:87`) and `state_module/agent_buddy/state_computer_plan.py`.

Net: **no shims.** Two routes survive under a truthful name; five endpoints and one
state file die.

---

## Task 9a — Unifying the browser tool; is browserless used

> **Comment:** "How do we unify the browser tool with the main agent? Is browserless
> being used?"

**Yes, browserless — definitively, and it is the only path.** There is no local Chromium
launch anywhere in the tool. `run_browser_task` reads `BROWSERLESS_URL` and fails fast if
unset (`tool_module/browser_tool.py:393`), appends `?stealth=true` to the WS handshake
(`_augment_cdp_url`, `:301`), and builds browser_use's `Browser` with
`{"cdp_url": ..., "is_local": False}` (`:437`). Every custom action reuses that same CDP
socket rather than dialing browserless a second time, deliberately — a second connection
lands on a different browser instance (`tool_module/browser_actions.py:13,98`).

**It is up right now:** container `arkos-browserless-1`,
`ghcr.io/browserless/chromium:latest`, up 4 weeks on `0.0.0.0:3000`; `/json/version`
reports `Chrome/147.0.7727.15`, `ws://0.0.0.0:3000/`.

`.env:69-70` set `BROWSERLESS_URL` and `SGLANG_URL`, so the earlier missing-env failure
is fixed on this box — but **neither name exists in `.env.example`**, so a fresh clone
reproduces it. Fold that into Task 9's config move.

**On unification: three seams, two of which already exist.**

1. *One tool schema* — `register_local_tool` emits the same
   `{name, description, inputSchema}` shape MCP tools use and packs into `list_all_tools`
   under a synthetic `"local"` server (`smithery.py:325,493`); the model cannot tell
   `browser_task` from a remote tool.
2. *One result envelope* — success already returns
   `{"content":[{"type":"text",...}]}` (`browser_tool.py:508`), but failure *raises*
   `BrowserToolError`, so a timeout arrives as an untyped generic tool failure. Task 9
   should make failure return the envelope too.
3. *One event stream* — not done; see Task 9b.

**Caveat the spec should state outright:** unification here means a uniform *interface*,
not a merged loop. Re-implementing DOM grounding — accessibility-tree extraction, element
indexing, per-step screenshotting, iframe and shadow-DOM handling — is a multi-month
build with a permanent maintenance tail against a browser that ships every six weeks, to
replace a dependency that already does it. That is an unbounded liability bought for
marginal architectural tidiness. Keep the child loop; own its edges.

**Number to fix: "~40 env vars" is wrong. The real count is 27** — 18 read via
`os.environ.get` directly, 9 more via `_bool_env` (`browser_tool.py:136`), of which 23
are `BROWSER_USE_*`/`BROWSER_STREAM_*`. The module docstring documents only 22
(`BROWSER_USE_CUSTOM_TOOLS`, read at `:442`, is undocumented). Say "27 env vars → ~6
config keys"; an inflated number is the easiest thing for a reviewer to knock down.

---

## Task 9b — How tight is the integration

> **Comment:** "how tight is this interation" [integration]

Today it is **loose: a shell-out with a timeout.** Task 9 makes it a *supervised
shell-out*, not a supervised child loop. Worth saying that plainly in the spec rather
than letting "unify" imply more.

**The contract surface as specced.** Inward: task description, step and wall-clock
budgets, allowed_domains. Outward: progress events and one result envelope. Everything
else stays behind the boundary on purpose — browser_use's per-step action selection, its
planning passes, its message compaction, its loop detector. We do not want to arbitrate
those; that is what we are paying the dependency for.

**Where today diverges from that text.** The inward direction is not per-call at all.
`max_steps` / `max_seconds` / `allowed_domains` are read from process env *inside*
`run_browser_task` (`browser_tool.py:415,436`), and the tool's `inputSchema` accepts only
`task` (`:527`). Per-call budgets need a schema change — that is the actual content of
the acceptance test.

**Outward, nothing crosses during the run.** The whole call is one
`await asyncio.wait_for(agent.run(max_steps=...), timeout=max_seconds)` (`:483`)
returning `history.final_result()` (`:499`). The only live signal is JPEG frames pushed
into an in-process, **user-keyed** broker (`tool_module/browser_stream.py`) drained by a
separate SSE endpoint (`/v1/browser/stream`, `base_module/browser_routes.py:37`) that the
frontend consumes (`frontend/components.jsx:206`). That is not the task event stream —
that is `log_event` into `task_events` (`base_module/task_store.py:33`), and
`browser_task` writes nothing there between `tool_call` and `tool_result`. **Two streams,
keyed differently (user_id vs task_id); Task 9 has to bridge them.**

**Cancellation is the real tightness gap.** There is no per-call interrupt and no handle
on the running child. `task_runner.cancel` (`base_module/task_runner.py:233`) cancels the
entire subagent; `CancelledError` does propagate past the `except Exception` and the
`finally` tears down screencast and browser (`browser_tool.py:488`), so it is not leaky —
but it is all-or-nothing, executor-path-only, and no partial result is recoverable. The
buddy chat path cannot cancel at all. If Task 9 ships budgets plus events only, a 180s
task is still 180s of unblockable dead air. Either add a cancel token as one more thing
crossing inward, or say explicitly that mid-flight interrupt is out of scope.

---

## Task 10 — Looking Glass scope and contract

> **Comment:** "This can likely be its own spec but the scaffolding needs to exist first
> for easy integration. Contract needs to be established"

Agreed, and the split has a clean seam: **this spec owns the contract, the Looking Glass
spec owns the surface.**

**Stays here (scaffolding, all backend):**

1. **The typed event union.** Today `task_events.kind` is a bare `TEXT` column
   (`0003_subagent_runtime.sql:39`) and the loop only ever writes five values —
   `cancelled`, `completed`, `error`, `failed`, `resumed` (`task_runner.py:69-275`). None
   of the things Looking Glass renders (hop boundaries, tool calls, tool results, text
   deltas, `needs_input`, suggestions) are events yet. Define them as a Pydantic
   discriminated union here; the UI spec consumes it.
2. **The per-session SSE subscribe endpoint.** A version already exists but is namespaced
   wrong: `computer_router.py:123` (`GET /computer/tasks/{task_id}/stream`) already tails
   `task_events` for any task since migration 0007 unified the tables. Promote it to a
   session-scoped route and fix its transport gaps.
3. **The suggestion-injection endpoint and its semantics.** `POST .../suggest` appends a
   `suggestion` row; the loop drains pending suggestions at the *top of the next hop* and
   materializes them as a user message — never mid-tool-call, never mid-stream. The ack
   the client trusts is the echoed event, not the 200.
4. **Session-log-as-durable-truth.** Every renderable thing is a `task_events` row first;
   the stream is a projection, never a side channel. This explicitly rules out the
   in-memory broker pattern (`browser_routes.py:44`), which loses everything on
   disconnect.
5. **Sequence numbering.** `task_events.event_id` is already `BIGSERIAL` and monotonic
   per task (`0003:37`), and `list_events(after_id=…)` already exists
   (`task_store.py:60`). Name it as the wire `seq` here so Task 12's gap-free replay has
   something to resume from.

**Moves to the Looking Glass spec:** the grid, the bubbles, the four status colors, the
detail-view layout, the tabs, the nav change. All UI.

---

## Task 11 — Fold Projects into the Looking Glass spec

> **Comment:** "Integrate into looking glass spec above"

The overlap is real, and the ordering is inverted as written.

Task 10's landing level *is* Task 11's feature: spec line 128 defines Looking Glass's
landing as "a grid of project bubbles… each with a status dot… a projection of the
project's lifecycle rollup." That grid cannot be built without a `projects` table, and
there is no such table — migrations 0001–0008 give us `tasks`, `task_events`,
`task_approvals`, `users`, `repeat_tasks`, `user_sandboxes`, `waitlist`, and nothing else
(`computer_tasks` / `computer_task_events` were dropped in `0007`). The existing mock
proves it: `landing/lookingglass.jsx:5` hardcodes a four-item `LG_PROJECTS` array because
there is no endpoint to call.

Yet Task 10 lists blockers "Tasks 2, 4, 6" (line 246) while Task 11 lists Task 10 as a
blocker (line 256). **That is a cycle.** Task 11 depends on Task 10 for the session
window; Task 10 depends on Task 11 for the grid.

So yes — fold Task 11's product half into the Looking Glass spec and leave here only what
the loop genuinely needs:

- **`project_id` on tasks + the `projects` table.** One migration. The loop needs a
  container key to write against.
- **The attention query as a projection over lifecycle rows.** One SQL query over
  `task_approvals` — already indexed for exactly this (`0003:75`, partial index on
  `(user_id, status) WHERE status='pending'`) — plus `needs_input` events. Three
  surfaces, one query, one resolve path.
- **Project-scoped memory keying.** `memory_module/memory.py:135` passes only `user_id`
  into `mem0.add`; project scope is a keying change in the memory module.
- **Project files mounted at sandbox provision.** `user_sandboxes` is keyed per-user
  (`0005:8`), so a per-project mount is a genuine backend change, not UI.

Everything else — bubbles, colors, Command Center layout, tab nesting, IA — is surface.

---

## Task 12 — Stack and contracts

> **Comment:** "We need to establish the stack for frontend and backend and decide the
> contracts"

Calls, not options.

**Frontend: Vite + React 18 production build + TypeScript.** TS specifically because the
ask here is contracts, and generated types are the only mechanism that makes a contract
fail at compile time rather than at runtime. This is greenfield, not a migration — there
is no `package.json`, `vite.config`, or `tsconfig` anywhere in the repo, so
`frontend/index.html:15-17` (CDN React *development* UMD build + `@babel/standalone`,
four `type="text/babel"` script tags) gets deleted wholesale. Keep the existing
`StaticFiles` mount (`base_module/app.py:52`) and point it at `frontend/dist` — no new
deploy surface.

**Backend: unchanged.** FastAPI 0.115.0, Starlette 0.38.6, Pydantic 2.12.5, uvicorn
0.30.6, Python 3.12. One addition: use `sse-starlette` (already installed, 2.2.1, unused)
instead of hand-rolled generators — we currently have three, each formatting SSE
differently (`app.py:631`, `computer_router.py:137`, `browser_routes.py:44`).

**Contracts:**

1. **Envelope.** One Pydantic discriminated union — common header
   (`seq`, `session_id`, `ts`, `kind`) + typed payload — emitted by both the chat and
   session streams. Today chat emits OpenAI `chat.completion.chunk` with a smuggled
   non-standard `ark_status` field inside `delta` (`app.py:613`, consumed at
   `seed.jsx:342`), while task events emit `{event_id, kind, content, payload}`
   (`tasks.py:110`). Two shapes for one concept. Keep an OpenAI-shaped adapter route only
   for third-party clients.
2. **Transport.** `id: <seq>` on every event, seq = `task_events.event_id`.
   `Last-Event-ID` replays `WHERE event_id > $1` (`list_events` already takes `after_id`,
   `task_store.py:60`). `: heartbeat` every 15s — `browser_routes.py:28` does this, the
   other two streams do not. `retry:` hint plus client backoff capped ~30s. **The stream
   must not close on a terminal event**; `computer_router.py:149` returns on
   `completed`/`failed`, which is why clients still poll. **Decide auth now:**
   `EventSource` cannot set headers (`components.jsx:206`), and today the stream
   authenticates off a query-param `user_id` with a config fallback
   (`browser_routes.py:31`) — that is an authz hole, not just a style issue. Short-lived
   stream token or cookie.
3. **REST surface.** `respond` / `suggest` / `cancel` each return 202 plus the seq at
   which the effect will appear. No mutation returns state. One write path (REST), one
   read path (SSE). This kills the current `refreshAll()`-after-every-action pattern
   (`app.jsx:186,236,257`).
4. **Reconcile at the reducer.** Client tags each mutation with an `action_id`; server
   echoes it in the resulting event's payload; client drops the optimistic entry when it
   sees that event. Not in components.

Pin all four as Pydantic models, emit OpenAPI in CI, generate TS types into the frontend,
fail the build on drift. Hand-kept sync is precisely what produced today's three
divergent SSE shapes.

---

## Task 13 — Is the multi-worker approach right

> **Comment:** "Is this the best architecture choice?"

The evidence shifts the answer.

**The hazard is real but it is not uvicorn.** Nothing launches with `--workers`;
`base_module/app.py:677` is a plain `uvicorn.run`, and prod's deploy forces
`reload: false`. But `~/dev/arkos/.env` and `~/srv/arkos/.env` both point at
`localhost:54322/postgres` — dev (:1121) and prod (:9000) are **two processes over one
`tasks` table**. Each runs `sweep_orphans()` at startup (`app.py:295`), selecting every
`running`/`awaiting_approval` row and spawning it with no claim
(`task_runner.py:259-277`). Two workers already exist; the only thing hiding it is that
prod is currently crashed.

**Options evaluated:**

- *(d) Keep single-worker as an enforced invariant* — dead. Task 6's
  `POST /approvals/{id}/respond` re-spawns a task at its cursor in whichever process
  served the request, so the claim must be per-task, not per-process.
- *(c) A real queue (Redis/Celery)* — buys nothing. There is no concurrency config
  anywhere in the repo; `max_iter` (`agent_module/agent.py:23`) is the only bound; load is
  small. Adds a daemon to a box already juggling a deploy cron, a tmux prod pane, and
  SGLang.
- *(b) LISTEN/NOTIFY* — the natural pairing with leasing later, but dispatch today is not
  a poll at all, so this would be net-new surface, not a replacement.
- *(a) Postgres row-lease* — yes, **with two edits.**

**Edit 1: cut `SKIP LOCKED`.** Dispatch is by known `task_id`
(`base_module/tasks.py:219,473`) — a single-row conditional UPDATE with RETURNING is
already atomic and tells you when you lost. `SKIP LOCKED` earns its keep on contended
multi-row scans; the only multi-row site is the startup sweep, at tens of rows. Cut the
concept.

**Edit 2: cut the per-worker semaphore.** The bottleneck is the single SGLang instance on
:30000 — one Qwen3-8B scheduler shared by both checkouts, `computer_agent`, and
open-webui. N per worker × W workers bounds nothing real. Bound at the Task 1 model
client, or let SGLang's scheduler queue.

**Lease expiry is what to actually design around.** Task 5 sets a 30-min wall-clock cap.
If TTL is under that and the task is genuinely alive — `browser_task` holds 180s, MCP
calls longer — the lease lapses, a second process claims, resumes from cursor, and
duplicates side effects: precisely Task 13's failure mode, caused by Task 13. Heartbeat
at hop boundaries (the loop already checkpoints there to persist the cursor), TTL a small
multiple of the longest single hop, plus a fencing epoch so a revived old owner's writes
are rejected.

**This couples to Task 5 harder than "Blockers: Task 5" implies.** A resumed orphan is
only safe if the tool result and the cursor advance commit in one transaction. Crash
between the tool call and the cursor write and any claimer replays it — correctly leased,
still duplicated. The stated acceptance test (2 workers, 10 tasks, tool-call uniqueness)
never exercises that. Add a kill-between-call-and-cursor case.

---

# Cross-cutting findings

Three items that are not questions.

## 1. Live P0: orphan respawn replays completed work

`sweep_orphans` respawns at `step_idx = 0` (`base_module/task_runner.py:175`) and flips
status to `running`. The whole plan replays from step 1 **with side effects**, and the
original `pending` approval row is never resolved or cancelled — orphaned in the tray
forever. `state_approval.py:5-7` docstrings claim the state "can be rehydrated from the
DB row"; there is no rehydration path.

Combined with the shared dev/prod database above, **a dev restart right now would replay
prod's in-flight tasks from step 1.** This is independent of the redesign.

## 2. Spec numbers a reviewer will knock down

| Claim | Reality |
|---|---|
| "~40 env vars" (Task 9) | 27 — 18 direct + 9 via `_bool_env` |
| "Streaming is fake" (Problem) | Streaming was never implemented; `generate_stream` has zero production callers |
| Task 8 acceptance test | Describes net-new capability — the computer agent has `tool_manager=None` today |

## 3. Two structural doc defects

- **Terminology contradiction:** line 65 ("One subagent, three capability families") sits
  inside the section that retires the word at line 111, and contradicts the by-role
  manifest table.
- **Dependency cycle:** Task 10 (line 246) and Task 11 (line 256) each list the other as
  a blocker.

---

## Provenance

Findings produced by seven parallel read-only investigation passes over the repo at
`28e46b3`. No files were modified and nothing was posted to Notion. Live service checks
(`docker ps`, `curl localhost:3000/json/version`, `curl localhost:30000/v1/models`) were
read-only.
