# Feature Spec: Task Surfacing — Issues & Fixes

**Sources**

- Backend surfacing map: `base_module/tasks.py`, `base_module/task_store.py`, `computer_module/store.py`, `computer_module/runner.py`, `computer_module/computer_router.py`, `state_module/agent_executor/state_approval.py`, `db/migrations/0001,0003,0006,0007`
- Frontend map: `frontend/app.jsx`, `frontend/seed.jsx`, `frontend/views.jsx`, `frontend/components.jsx` (the **only** served UI — `arkos-webui` is not used)
- Companion specs: `HARNESS_SPEC.md`, `MEMORY_SPEC.md`, `ROLLBACK_SPEC.md`

**Status:** Not started | **Author:** | **Last updated:** 2026-06-15

---

# Consolidated Punch List (added 2026-06-15)

A single ranked view of every confirmed defect behind "tasks finish but don't show
up" and "things get missed along the way." It merges the two surfacing mechanisms
this spec already documents with three execution-layer bugs found by reading the
current tree. Line references verified against the working copy on 2026-06-15.

Whole thing is roughly a week of focused work, not a rewrite. The design (state
graph, schema, contracts) is intact; every item below is glue at one of three
seams: the frontend fetch layer, the identity boundary, or the post-unification
caller updates that didn't land.

| # | Sev | Seam | Defect | Evidence |
|---|-----|------|--------|----------|
| 1 | P0 | frontend | List never asks for finished work, so DONE never appears and a completed/failed task drops out on the next poll. Also kills the command log: events are fetched only for tasks still in the live buckets, so a finished computer task's shell history vanishes with it. | `app.jsx:111` (fetches only `running` + `awaiting_approval`), `app.jsx:130` (events only for live `rawTasks`) |
| 2 | P0 | identity | Reads scope by JWT `sub`; some writes scope by the `X-User-ID` header; `_user_uuid()` derives a third UUID keyspace. If the header ever differs from `sub`, a row is written under one identity and listed under another. "Stores but doesn't surface." | `task_store.py:23`, `app.py:238,299,344,371,505`, `seed.jsx` POST headers |
| 3 | P1 | execution | Executor subagent can never see tools. `ScopedToolManager.list_all_tools(self)` takes no `user_id`, but the executor calls it with one → `TypeError`, caught, logged as "could not list tools", tool list becomes "(no tools available)". Every MCP step then degrades to ask_human. Breaks the original MCP flow outright. | `tool_module/scoped.py:37` vs `state_module/agent_executor/state_executor.py:83` |
| 4 | P1 | execution | Executor error path crashes. On any error in a non-terminal state, `agent.py` sets `current_state = flow.get_state("agent_reply")` — a buddy-only state absent from the executor graph → `KeyError` → task marked failed. Combined with #1 the failure is invisible. | `agent_module/agent.py:398,491`; executor graph has only `executor/use_tool/ask_human/executor_done` |
| 5 | P1 | frontend | Approvals and plans fail silent. Optimistic approval removal isn't rolled back on a failed respond (zombie reappears next session). A malformed `ark-plan` fence is regex-parsed away with no affordance, leaving a plan with no approve button. | `app.jsx:164-173`, `seed.jsx:50-58` (`parsePlan`) |
| 6 | P2 | backend | Three writers touch `tasks.status`, and status is written separately from its event — a poll can see a status with no event, or a terminal status with stale `context_payload`. | `task_store.py:81-137`, `computer_module/store.py:94-109` |
| 7 | P2 | ops | Migrations are manual. `db/migrate.py` is not run at startup or in compose. If 0003/0007 aren't applied on the deployed DB, every unified query 500s and #1 hides it. | no startup/compose invocation of `db/migrate.py` |

**Map to the detailed tasks below:** #1 = Task 1, #2 = Task 2, #5 = Task 3, #6 =
Task 4. **#3, #4, #7 are new** and not yet broken out into Tasks — they are
execution/ops bugs, not surfacing, but they produce failed or stuck rows that #1
then renders invisible, which is why the system "feels" broken rather than "shows
an error."

**Direct answers to the three questions that prompted this:**

- *Why don't finished tasks surface to DONE?* Item #1. `refreshAll` only ever
  requests `status=running` and `status=awaiting_approval`. There is no fetch for
  `completed`/`failed` anywhere, so a finished task is simply never asked for and
  falls out of the list on the next 6s poll.
- *Is the plan JSON schema-constrained, or scraped?* Both, at different ends, and
  that split is the brittleness. The backend generates the plan under a real JSON
  schema (`state_plan.py` passes `WorkshopOutput.model_json_schema()` to the LLM),
  then re-serializes the validated object into a ` ```ark-plan ` fenced block
  inside the chat text. The frontend recovers it by **regex + `JSON.parse`**
  (`seed.jsx:50`), not constrained decode. So generation is structured but the
  channel back to JS is an unstructured text scrape; any fence drift makes
  `parsePlan` silently return `plan:null` (item #5).
- *Are the computer agent's commands surfaced?* At the data layer, yes — every
  sandbox tool call emits before it runs (`tools.dispatch` → `{kind:"shell",
  tool:"run_command", args}` at `tools.py:204`), which flows through the runner's
  `emit` into the shared `task_events` table and is readable at
  `/tasks/{id}/events`. The gap is purely #1: the UI fetches those events only for
  tasks still in the live buckets, so you can watch commands stream during a run
  but the whole log disappears the instant the task finishes, and a fast-failing
  task never shows them at all.

---

# Concurrency & Memory Fixes (DONE 2026-06-15)

Separate from the surfacing work above. These came out of a "the prompt is global
shared-only, I thought it was per-user" audit of the chat hot path. Three globals
that should be per-user, two of them real races, one a confirmed memory bleed. All
three are now fixed in the tree.

## CM1 -- Per-request global system-prompt rebuild (cross-user race) — FIXED

**Was:** `_refresh_system_prompt()` rewrote the module globals `_system_prompt` /
`_available_tools` on every chat request (`app.py:580`) and on every
connect/disconnect/oauth. It did `_available_tools = await
tool_manager.list_all_tools()` with an `await` in the middle, so two concurrent
users rewrote the same globals while a third read them — the exact shape of the
original U9 "last request's user leaks into everyone's prompt." Benign only because
`list_all_tools()` with no `user_id` returns the static shared set.

**Fix:** Deleted `_refresh_system_prompt` and all four call sites. The prompt is
already built per-user in `_make_agent` from live `tool_manager` state (shared tools
loaded once at startup + this caller's connected per-user tools). `_available_tools`
is now written once at startup and treated as read-only shared state. Removes the
race and an extra `list_all_tools()` round-trip per message.

**Touch point:** `base_module/app.py` (`_refresh_system_prompt` removed; calls at the
old 398/425/476/580 removed).

## CM2 -- Shared State instances could bleed across users (latent) — FIXED

**Was:** One `StateHandler` (`flow`) is shared across every concurrent request, so
every agent's `current_state` points at the *same* `State` object. Safe only because
states are stateless today (all `self.x =` writes are in `__init__`; per-request data
lives on the Agent). Nothing enforced that invariant — one `self.x = ...` added to a
`run()` would silently bleed one user's data into every concurrent user.

**Fix:** `State` now freezes after construction (`_freeze()` + a `__setattr__` guard),
and `StateHandler` calls `_freeze()` on each state it builds. Any attribute write at
runtime now raises immediately instead of bleeding. Verified: construction still works,
post-freeze writes raise.

**Touch point:** `state_module/core/state.py`, `state_module/core/state_handler.py`.

## CM3 -- Short-term memory bleed across conversations (confirmed) — FIXED

**Was:** `retrieve_short_memory` filtered `WHERE user_id = %s` only, never
`session_id`, even though `add_memory` writes a `session_id`. So it returned the
user's last N turns across *every* conversation, and unrelated chats bled into each
other. Compounded by one sticky `Memory` object cached per user (one session id for
the whole server lifetime).

**Fix:** Three parts.
- `retrieve_short_memory` now scopes to `(user_id, session_id)` (`memory_module/
  memory.py`).
- The chat endpoint reads an `X-Session-ID` header and threads it into a
  per-`(user, session)` Memory via `_get_or_create_memory(user_id, session_id)` /
  `_make_agent(user_id, session_id)` (`base_module/app.py`).
- The frontend generates a per-conversation session id, sends it on every chat
  request, and resets it on sign-out (`frontend/seed.jsx`).

The executor already built `Memory(session_id=...)` per task, so both paths now agree.

**Note / follow-up:** with short-term scoped per session, a fresh browser session
starts with empty DB recall and relies on the request's own full history (which the
frontend already sends). Durable cross-session recall is the long-term (mem0) path,
which is **off by default** in `config.yaml` (`memory.use_long_term: false`). If
cross-session "remembers last week" is wanted, that's a separate long-term-memory
task, not a regression of this fix.

---

# Problem

The database stores task/approval/event state correctly, but the UI surfaces it
brittlely — finished tasks disappear, approvals reappear after being resolved,
and some rows never surface at all. The cause is structural: task state lives in
**four sources of truth** (`tasks.status`, `task_events`, `task_approvals`,
`conversation_context`) with **no single stream the UI subscribes to**. The
frontend reconstructs the picture by polling several endpoints on a 6-second
timer, so any gap between "DB is correct" and "reconstructed view is correct
right now" shows as flakiness. Two concrete mechanisms produce the user-visible
"stores but doesn't surface":

1. **The UI never asks for finished work.** `refreshAll` polls only
   `running` and `awaiting_approval` (`app.jsx:111`). A task that completes is
   `completed`/`failed` — in neither bucket — so it drops out of the list on the
   next poll. The DONE state, and the computer-task badge count (`views.jsx:170`),
   are structurally unobservable.

2. **Two user-id keyspaces, read with the wrong one.** Tasks/approvals are scoped
   by `_user_uuid(sub)` → a derived **UUID** (`task_store.py:23`); Memory/sandbox/
   `conversation_context` use the **raw `sub`**. Frontend GET reads send
   **Bearer only**; POST writes send **Bearer + `X-User-ID`** (`seed.jsx`); and
   backend endpoints are **mixed** — some scope by the JWT `sub` (`CurrentUser`),
   others by the `X-User-ID` header (`app.py:238,299,344,371,505`). If
   `X-User-ID` ever differs from the JWT `sub`, a row is written under one
   identity and listed under another.

Secondary brittleness: optimistic approval removal that doesn't roll back on
failure (`app.jsx:166`), silent `ark-plan` fence parse failures (`seed.jsx:50`),
and a backend status machine with three writers and a status/event ordering
window.

**Success looks like:** a finished task stays visible with its DONE state; a row
created in the UI always appears in that same UI; approvals fail loud, not
silent; and task status is written by one path, atomically with its event.

---

# Technical Background

**The four sources of truth and who writes them:**
- `tasks.status` — written by `set_task_status`, `mark_task_completed/failed`
  (`task_store.py:81-137`) and the `set_computer_status` wrapper
  (`computer_module/store.py:94-109`). Three writers.
- `task_events` — append-only progress, written separately from status.
- `task_approvals` — created by `create_approval`, resolved by `resolve_approval`;
  polled every 2s by `state_approval.py` and `computer_module/runner.py`.
- `conversation_context` — the chat "Done." message, injected under the **raw
  sub** by `runner.py` Memory.add_memory.

**The keyspace boundary** (`task_store.py:23`): `_user_uuid()` converts the JWT
`sub` to a deterministic UUID. This is the single tasks/approvals boundary; the
sandbox and Memory keyspaces use the raw `sub`. The split is internally
consistent *only if every endpoint derives identity the same way* — which today
it does not (some use `CurrentUser`/JWT, some use the `X-User-ID` header).

**Unification status:** migrations `0006→0007` merged `computer_tasks` into the
`tasks` table (`agent_kind='computer'`), and events/approvals are now shared. The
projection `_project()` (`store.py:38`) reshapes a unified row back into the
computer-task shape by reading `context_payload`, which is only written on
terminal states — so mid-flight reads can see stale summary/outputs.

**Decisive diagnostic** (splits Mechanism 1 from 2; run while a "missing" task is
on screen):
```sql
select task_id, status, agent_kind, user_id from tasks order by created_at desc limit 10;
```
Row present with `status=completed` → Mechanism 1. `user_id` not matching
`_user_uuid(your_sub)`, or two `user_id`s for "you" → Mechanism 2.

**Not in scope:** the `arkos-webui` Svelte UI (not served — ignore/delete).
Lower-priority backend hardening (SSE terminal-event race
`computer_router.py:146`, orphan double-execution `task_runner.py:250`) is
deferred to Open Questions — real but not the everyday flakiness.

---

# Proposed Approach

Fix the two "doesn't surface" mechanisms first (highest visible impact), then make
approvals fail loud, then collapse the backend status machine to one atomic
writer. Nothing here changes the state-graph design; the edits live in the
frontend fetch layer, the identity-resolution seam, and the status-write helpers.

What stays the same: the `tasks`/`task_events`/`task_approvals` schema, the router
state graph, and the SSE endpoint that already exists for computer tasks.

---

# Implementation Plan

## Task 1: Surface finished tasks in the list

**Problem:** The list polls only `running` + `awaiting_approval`, so a completed
or failed task vanishes on the next poll; DONE never shows and the computer badge
undercounts.

**Decision (2026-06-15):** the finished-task window is **the last 15 minutes**.
Long enough to see a task you just ran land in DONE; short enough that the list
never grows unbounded. The bound is applied server-side in the `/tasks` query for
terminal statuses (`completed`/`failed`), keyed off the task's terminal timestamp,
not client-side.

**Done when:**
- `refreshAll` also fetches `completed` and `failed`, bounded to **terminal within
  the last 15 minutes**, and merges them into the list.
- The bound is enforced in the backend list query (e.g.
  `status IN ('completed','failed') AND finished_at >= now() - interval '15 minutes'`),
  so the client never pulls the full history.
- The task row renders its terminal state (`done`/`stop`) and persists across
  polls within the window; the computer-task badge counts terminal states correctly.

**Touch point:** `frontend/app.jsx:111-138`, `frontend/seed.jsx` (`api.tasks`),
`frontend/views.jsx:62-73,170`.

**Priority:** P0 | **Effort:** ~0.5 day | **Blockers:** none

**Out of scope:** Replacing polling with SSE (Open Question 1).

**Acceptance test:** `test_completed_task_stays_visible` (below).

---

## Task 2: One identity source across reads and writes

**Problem:** Reads scope by JWT `sub`, some writes by the `X-User-ID` header; if
they differ, a row is stored under one user_id and listed under another
("stores but doesn't surface").

**Done when:**
- Every task/approval/computer endpoint derives identity from one source — the
  verified JWT `sub` via `CurrentUser`. The `X-User-ID` header is no longer used
  for scoping (removed from `app.py:238,299,344,371,505` and from the frontend
  POST headers, or explicitly ignored server-side).
- Reads and writes resolve to the same `user_id`; the diagnostic query shows a
  single `user_id` for one logged-in session.

**Touch point:** `base_module/app.py` (X-User-ID lines), `computer_module/
computer_router.py`, `frontend/seed.jsx` (auth headers), `task_store.py:23`
(`_user_uuid` stays the one boundary).

**Priority:** P0 | **Effort:** ~1 day | **Blockers:** none

**Decision (2026-06-15):** we are also collapsing the keyspaces — see **Task 5**.
Task 2 (single-source *resolution*) still ships first as the fast fix; Task 5 then
removes the dual keyspace entirely so there is one `user_id` everywhere.

**Out of scope:** the keyspace consolidation itself (now its own Task 5). Task 2 is
just making every endpoint resolve identity from the verified JWT `sub`.

**Acceptance test:** `test_task_created_in_ui_appears_in_same_ui` (below).

---

## Task 3: Approvals fail loud, not silent

**Problem:** Optimistic approval removal isn't rolled back on failure (resolved in
UI, still `pending` in DB → reappears next session); a malformed `ark-plan` fence
is parsed away silently, leaving a plan with no approve control.

**Done when:**
- `respondApproval` throws on non-2xx; the approval is removed from the UI only
  after server confirmation (or restored if the call fails).
- `parsePlan` failure surfaces an inline "couldn't parse plan" affordance instead
  of silently dropping the block.

**Touch point:** `frontend/app.jsx:164-173,227-243`, `frontend/seed.jsx:50-59`
(`parsePlan`), `seed.jsx` (`respondApproval`).

**Priority:** P1 | **Effort:** ~1 day | **Blockers:** none

**Out of scope:** Backend approval lifecycle changes (Task 4).

**Acceptance test:** `test_failed_respond_restores_approval`,
`test_malformed_ark_plan_surfaces_error` (below).

---

## Task 4: One atomic status writer (backend)

**Problem:** Three writers touch `tasks.status` (`set_task_status`,
`set_computer_status`, `mark_task_completed/failed`), and status is written
separately from its event — a poll can see a status with no matching event, or a
terminal status with stale `context_payload`.

**Done when:**
- A single `update_task_status(task_id, status, *, summary=None, error=None,
  outputs=None, event=None)` writes status, `context_payload`, and the
  corresponding `task_events` row in **one transaction**.
- Both the executor and computer paths route through it; the
  `set_computer_status` wrapper either delegates to it or is removed.

**Touch point:** `base_module/task_store.py:81-137`, `computer_module/store.py:
94-109`, emit sites in `computer_module/runner.py` and
`state_module/agent_executor/state_approval.py`.

**Priority:** P1 | **Effort:** ~1.5 days | **Blockers:** none

**Out of scope:** SSE terminal-event race and orphan resume (Open Question 3).

**Acceptance test:** `test_status_and_event_are_atomic`,
`test_single_status_writer` (below).

---

## Task 5: Consolidate identity onto one UUID keyspace

**Problem:** Two keyspaces exist — tasks/approvals use `_user_uuid(sub)` (a derived
UUID), while Memory, the sandbox registry, and `conversation_context` use the raw
`sub`. Task 2 makes *resolution* single-source but leaves two stored shapes, which
keeps the door open for future drift and makes cross-joins (a task's chat message,
a user's sandbox) awkward.

**Decision (2026-06-15):** unify on the **UUID**. The derived `_user_uuid(sub)`
becomes the one identity for every per-user keyspace; raw `sub` is used only at the
JWT boundary to compute it, never as a storage key.

**Done when:**
- Memory (`MEMORY_SPEC.md`), the sandbox registry (`user_sandboxes`), and
  `conversation_context` key off the same UUID that tasks/approvals already use.
- `_user_uuid()` is the single conversion point, called once at the request edge;
  no module stores or queries by raw `sub`.
- A one-time backfill migration maps existing raw-`sub` rows to their UUID (or, for
  dev data with no lasting value, is documented as dropped — mirror the 0007 note).
- The diagnostic query shows exactly one `user_id` per user across `tasks`,
  Memory, and sandbox tables.

**Touch point:** `task_store.py:23` (`_user_uuid` stays the one boundary, now used
everywhere), `memory_module/*`, `computer_module` sandbox registry, `conversation_
context` writers in `runner.py`, plus a backfill migration under `db/migrations/`.

**Priority:** P1 | **Effort:** ~2 days | **Blockers:** Task 2 (resolution must be
single-source before the stored keyspace is collapsed)

**Out of scope:** changing what Memory stores; this only changes the *key*.

**Acceptance test:** `test_one_user_id_across_keyspaces` (below).

---

## Task 6: Replace 6s polling with SSE for all tasks

**Problem:** The list is rebuilt by polling several endpoints on a 6s timer, so any
gap between "DB is correct" and "the reconstructed view is correct right now" shows
as flakiness, and the finished-task window (Task 1) is a workaround for the poll
model rather than a fix.

**Decision (2026-06-15):** **yes**, extend the SSE pattern already used for
computer tasks (`/computer/tasks/{id}/stream`) to all tasks — but as a
**fast-follow after Tasks 1–4 land.** Identity (Task 2/5) and the single atomic
status writer (Task 4) must be clean first; SSE over a racy multi-writer backend
would just stream the inconsistency faster.

**Done when:**
- A single stream (e.g. `/tasks/stream` or `/tasks/{id}/stream` generalised from
  the computer endpoint) pushes status transitions, events, and approval
  state for the authenticated user.
- The frontend subscribes once on login and drops the `setInterval(refreshAll,
  6000)` loop; the 15-minute finished window becomes the initial backfill on
  connect, after which terminal transitions arrive live.
- Reconnect/backfill handles a dropped connection without losing or duplicating
  terminal events (the SSE terminal-event race in Open Question 3 is closed as
  part of this).

**Touch point:** `computer_module/computer_router.py:146` (existing stream to
generalise), `base_module/tasks.py` (new stream endpoint), `frontend/app.jsx:98-138`
(replace polling with a subscription), `frontend/seed.jsx` (EventSource client).

**Priority:** P2 (fast-follow) | **Effort:** ~2–3 days | **Blockers:** Tasks 1–4

**Out of scope:** none beyond the above; this subsumes Open Questions 1 and 3.

**Acceptance test:** `test_terminal_transition_streams_without_poll` (below).

---

## Task 7: One JSON-repair chokepoint per language (not per call site)

**Problem:** LLM JSON is parsed in several places and each one fails its own way —
the frontend `parsePlan` silently drops a malformed `ark-plan` fence, and the five
executor/buddy states call `model_validate_json` directly, so a single stray token
from the model breaks a parse with no recovery.

**Why repair is needed even with constrained generation:** schema-constrained
decoding does **not** guarantee valid JSON at the parse site. The output still
breaks when the response is truncated at the token cap mid-object, a stop sequence
fires early, the model wraps the JSON in a ```` ```json ```` fence, or a code path
calls the model without the grammar actually attached. `json-repair` closes brackets,
strips fences, removes trailing commas, and quotes stray keys — it is the safety net
for exactly the case where the constraint did not hold.

**Decision (2026-06-15):** add a JSON-repair pass at the **one function each parse
stems from**, not at every parse site. Two chokepoints, one repair library each.

**Done when:**
- **Python:** one helper — `parse_llm_json(content: str, model: type[BaseModel])`
  in a core util (or on the `ArkModelLink` wrapper) — runs the **`json_repair`**
  package (`pip install json-repair`; `from json_repair import repair_json`) to
  fix the raw string, then `model_validate_json` on the repaired output. The five
  sites switch to calling it and stop calling `model_validate_json` directly:
  `state_ai.py:121`, `state_plan.py:79`, `state_computer_plan.py:75`,
  `state_executor.py:124`, `state_tool.py:86`. No repair logic is duplicated in any
  state.
- **JS:** one helper — `safeJsonParse(str)` in `seed.jsx` — runs the **`jsonrepair`**
  package (`npm i jsonrepair`; `import { jsonrepair } from 'jsonrepair'`) before
  `JSON.parse`. The two parse sites route through it: `parsePlan` (`seed.jsx:55`)
  and the SSE payload parse (`seed.jsx:285`). `parsePlan` keeps the fail-loud
  affordance from Task 3 only for the case `jsonrepair` still can't salvage.
- Both libraries are added to `requirements.txt` / `package.json` respectively.

**Touch point:** one new Python util + one new JS util; the seven call sites above
become one-line swaps to the helper.

**Priority:** P1 | **Effort:** ~0.5 day | **Blockers:** none (complements Task 3)

**Out of scope:** the trusted `json.loads` over our own DB columns
(`tasks.py:146,311,405,413`, `store.py:43`, `task_runner.py:93`) — that data is
JSON we wrote, not model output, so it needs correctness from the writer, not
repair on read. Do **not** wrap these in the repair helper.

**Acceptance test:** `test_parse_llm_json_repairs_then_validates` (below).

---

## Task 8: Collapse routers into the graph (single-traversal routing)

**Problem:** Routing requires three artifacts kept in sync by hand — the signal
string a state emits in `structured_data["route"]`, the router function that
translates it to a state name, and the graph YAML's `transition.next` list. All
three must be updated together for any routing change. When they drift, failures
are silent: we added `action=advance` to the executor, missed the router and the
graph, and multi-step tasks silently ended after the first step.

**Background — how LangGraph does it:** after a node runs, one routing function
reads the updated state and returns the next node name directly. No intermediate
signal object, no translation layer, no separate router file. Node updates state →
routing function reads state → returns destination. Single traversal, one source
of truth.

ARKOS currently does two traversals: state emits a signal → router translates it
→ graph validates the result. The router `routers.py` is a third file that
duplicates information already in the YAML, and `StateHandler` must load both. The
only reason the pattern exists is the CLAUDE.md contract that states must not
hardcode their successor names — that decoupling is worth preserving, but the
implementation is more complex than necessary.

**Done when:**
- The graph YAML supports declaring signal→state mappings inline on each
  transition, e.g.:

  ```yaml
  executor:
    transition:
      edges:
        tool:     use_tool
        ask:      ask_human
        continue: executor
        _default: executor_done
  ```

- `StateHandler` auto-generates a router from the YAML for any state that uses
  `edges:` — no `routers.py` entry needed for simple signal→state maps.
- States still emit semantic signals (`"plan"`, `"tool"`, `"continue"`) — the
  decoupling from successor names is preserved.
- `routers.py` is kept only for states whose routing logic is too complex to
  express as a flat signal map (e.g. inspecting `output.content` or combining
  multiple fields). Simple cases need no Python at all.
- A routing change (new signal, new successor) requires editing exactly one file:
  `graph.yaml`.

**Touch point:** `state_module/core/state_handler.py` (auto-router generation),
`state_module/agent_executor/graph.yaml` + `routers.py` (migrate to inline edges),
`state_module/agent_buddy/graph.yaml` + `routers.py` (migrate simple cases).

**Priority:** P2 | **Effort:** ~1–2 days | **Blockers:** none

**Out of scope:** replacing the Python override path for complex routers (those
stay as functions); changing the signal protocol states use.

**Acceptance test:** `test_yaml_edge_map_routes_without_router_function`,
`test_routing_change_requires_only_graph_edit` (below).

---

# Tests

## Test 1: test_completed_task_stays_visible

**What it verifies:** A task transitioned to `completed` is still returned by the
list the UI fetches and renders with a terminal state across two consecutive
polls.

**Why this matters:** Disappearing finished tasks is the #1 visible symptom; the
test pins that completion is observable, not just stored.

---

## Test 2: test_task_created_in_ui_appears_in_same_ui

**What it verifies:** A task created through the dispatch path (POST) is returned
by the list path (GET) for the same authenticated session — i.e. write-identity
and read-identity resolve to the same `user_id`.

**Why this matters:** This is the "stores but doesn't surface" identity bug; the
test guarantees one identity end-to-end.

---

## Test 3: test_failed_respond_restores_approval

**What it verifies:** When the approval-respond call fails, the approval is
restored in the UI state (not left optimistically removed) and remains
resolvable.

**Why this matters:** Silent optimistic removal produces zombie approvals that
reappear; the test pins fail-loud behavior.

---

## Test 4: test_malformed_ark_plan_surfaces_error

**What it verifies:** A buddy reply with a malformed `ark-plan` fence renders a
visible parse-error affordance rather than silently dropping the block.

**Why this matters:** A plan with no approve button looks like the system did
nothing; surfacing the failure is the difference between "broken" and "told me
why."

---

## Test 5: test_status_and_event_are_atomic

**What it verifies:** A status transition and its corresponding event are written
in one transaction — a reader never observes the new status without its event (or
vice versa).

**Why this matters:** The status/event ordering window is a core source of
"flaky for a second" surfacing; atomicity closes it.

---

## Test 6: test_single_status_writer

**What it verifies:** Both executor and computer terminal transitions go through
the one `update_task_status` helper, which sets status + `context_payload`
together (summary/outputs never null on a terminal row).

**Why this matters:** Multiple writers + payload-only-on-terminal causes stale or
inconsistent rows; one writer makes the terminal row self-consistent.

---

## Test 7: test_one_user_id_across_keyspaces

**What it verifies:** After Task 5, a single user resolves to exactly one `user_id`
across `tasks`, Memory, and the sandbox registry — no row is keyed by raw `sub`.

**Why this matters:** Pins the keyspace consolidation so the dual-identity bug
cannot silently reappear in a new module.

---

## Test 8: test_terminal_transition_streams_without_poll

**What it verifies:** With the SSE subscription (Task 6) and polling removed, a task
transitioning to `completed`/`failed` pushes a terminal event to the subscribed
client without a poll, exactly once across a reconnect.

**Why this matters:** Confirms SSE actually closes the poll-window flakiness and
the terminal-event race, not just relocates them.

---

## Test 9: test_parse_llm_json_repairs_then_validates

**What it verifies:** `parse_llm_json` repairs a malformed-but-recoverable model
output (trailing comma, unquoted key, code-fence wrapper) and returns a valid
model; an unrecoverable input raises one well-defined error, not a silent drop.

**Why this matters:** Pins that repair lives in the one helper and that every state
inherits it by calling that helper, not by re-implementing parsing.

---

## Test 10: test_yaml_edge_map_routes_without_router_function

**What it verifies:** A state whose graph YAML uses `edges:` (signal→state map)
is routed correctly by `StateHandler` without any entry in `routers.py` — the
auto-generated router picks the right successor for each signal, and falls back
to `_default` for unknown signals.

**Why this matters:** Pins that the graph is the single source of truth for simple
routing. If a router function is accidentally left in place alongside an `edges:`
map, the test should confirm the YAML wins.

---

## Test 11: test_routing_change_requires_only_graph_edit

**What it verifies:** Adding a new signal→state mapping to `graph.yaml` (no
`routers.py` change, no state code change) causes the harness to route that signal
to the new successor correctly.

**Why this matters:** The whole point of the task is that a routing change touches
one file. This test pins that the promise is real and won't silently break when
`StateHandler` is extended.

---

# Open Questions

1. ~~Replace the 6s task polling with SSE, extended to all tasks?~~ **RESOLVED
   (2026-06-15): yes — see Task 6.** Done as a fast-follow after Tasks 1–4 so SSE
   streams a clean, single-writer, single-identity backend rather than the current
   race.
2. ~~Unify the UUID vs raw-sub keyspaces, or is single-source resolution enough?~~
   **RESOLVED (2026-06-15): unify onto the UUID — see Task 5.** Task 2 ships the
   resolution fix first; Task 5 then collapses the stored keyspace so there is one
   `user_id` everywhere (Memory included).
3. Backend hardening pass for the SSE terminal-event race (`computer_router.py:146`)
   and orphan double-execution on restart (`task_runner.py:250`). **The SSE race is
   now folded into Task 6.** Orphan double-execution still stands alone — real but
   not everyday flakiness; bundle separately.
4. ~~Should the finished-task list be time-bounded?~~ **RESOLVED (2026-06-15):
   last 15 minutes, enforced server-side — see Task 1.**

---

# Implementation Notes

*Add entries here as work lands.*

- `arkos-webui` (Svelte) is **not served** — only `/frontend` is. It has no auth
  and no task handling; ignore it in all surfacing work, or delete it to remove
  the temptation to "use both UIs."
- Run the decisive diagnostic query (Technical Background) before starting: it
  tells you whether the first missing task you see is Mechanism 1 (Task 1) or
  Mechanism 2 (Task 2), so you fix the one that's actually biting first.
- **Tool state vs conversation memory (2026-06-17).** These are different things.
  Conversation memory (`conversation_context`, mem0) is persistent Postgres/vector
  state keyed by `user_id` — fully shared between buddy and executor subagents.
  Tool state (`tool_manager._user_tools`) is a transient Python dict in the
  FastAPI process, populated lazily only when a user sends a chat message through
  `/v1/chat/completions`. It is lost on every server restart and was never written
  by the task runner (which bypasses the chat path entirely). Patched in
  `task_runner.run_task()` by pre-loading per-user OAuth tools before spawning the
  executor (commit `42b954e`). The durable fix is to persist connection status (not
  tokens — those stay in Smithery) in the DB so any code path can load them without
  requiring a prior chat turn. Track as a follow-on to Task 5 (identity/keyspace
  consolidation).
