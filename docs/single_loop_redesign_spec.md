# Single-Loop Redesign — Native Tool Calling

**Sources**

- `agent_module/agent.py` (step / step_stream loop review)
- `tool_module/smithery.py`, `model_module/ArkModelNew.py`, `memory_module/memory.py` (infra review)
- `base_module/task_runner.py`, `state_module/agent_executor/*` (executor loop review)
- `computer_module/agent.py`, `computer_module/model.py`, `computer_module/runner.py` (computer pathway audit)
- `tool_module/browser_tool.py`, `browser_actions.py`, `browser_stream.py` (browser pathway audit)
- Anthropic, "Scaling Managed Agents: Decoupling the brain from the hands" (Apr 2026) — brain/hands/session interfaces
- Harness studies: Claude Code source (`src/query.ts`, `toolExecution.ts`, `withRetry.ts`, AgentTool) and OpenClaw (`pi-embedded-runner`, model-fallback, session guards) — resilience contracts
- vLLM tool-calling docs (`--enable-auto-tool-choice --tool-call-parser hermes`)

**Status:** Not started | **Author:** John Wallace | **Last updated:** 2026-07-03

---

# Problem

One "use a tool" turn today costs 3 to 5 sequential LLM calls (reply, plan, tool choice, tool args, advance decision), each preceded by a Postgres read and a mem0 vector search, each producing structured JSON that is parsed, repaired, and routed through a state graph. On a self-hosted 7B model this is the difference between a turn feeling instant and a turn taking tens of seconds.

Stacked on top: three nested retry loops (OpenAI SDK default 600s timeout with 2 hidden retries, inside generate_response's 3 attempts, inside the agent loop's 11 iterations) give a worst-case turn latency of ~16.5 hours and 99 requests against a struggling model server. Streaming is fake: the full completion is awaited, then replayed one character per SSE envelope. The executor's iteration budget conflates transitions with retries, so runaway tasks burn their budget and then report **completed**. The approval state polls Postgres synchronously every 2s for up to 24 hours, then fails the task.

Cost of current behavior: multi-second to multi-minute perceived latency, tasks that hang for ages then fail, tools reported missing after restart, ~9,900 lines of production Python to maintain.

Success looks like: first token on screen in under a second, one LLM call per agent hop, tool calls executing in 1 HTTP round trip when warm, tasks that resume instead of re-executing after restart, and a production codebase around 4,500 lines.

---

# Technical Background

**Native tool calling.** OpenAI-compatible servers (including vLLM serving Qwen 2.5 with `--enable-auto-tool-choice --tool-call-parser hermes`) accept a `tools=[...]` array and return `tool_calls` (name + JSON args) inside the same completion that streams text. The model's chat template does, in one call, what ARKOS currently does in three (choose tool, fill args, decide next state). Routing information already exists in the response shape: tool calls present means execute and loop; absent means the turn is done.

**Why the state graph exists.** The FSM + enum-constrained selection was built to keep a 7B model on rails: every decision is a closed-choice structured output. The insight of this redesign is that the rails cost 5x latency on every turn to prevent a failure mode (malformed tool call) that is rare and cheaply recoverable with validate-and-retry.

**The four loops today.**

1. *LLM retry stack* — SDK retries x generate_response retries x agent-loop retry signal. Too loose: multiplicative, unbounded in practice.
2. *Agent step loop* — `retry_count` counts successful transitions and error retries against one `max_iter` budget. Simultaneously too tight (a 3-step executor plan with pagination exhausts it doing legitimate work) and too loose (a failing tool loops executor↔tool until the budget dies, then the runner marks the task green via the `completion_signal="complete"` fallback).
3. *Approval poll* — 2s synchronous psycopg2 poll on the event loop, 24h timeout, hard-fails on expiry.
4. *Browser-use internal loop* — `browser_tool.py` delegates to the third-party `browser_use` Agent, which runs its own multi-step loop (25 steps, 180s wall clock, its own LLM calls, compaction, loop detection) configured via ~40 env vars. From the main agent's perspective it is one long blocking tool call.

**The computer agent already proves the design.** `ComputerAgent.run()` (`computer_module/agent.py:96`) is precisely the proposed loop: in-memory message list, one native tool-calling completion per hop, dispatch `tool_calls`, append results, repeat until the model stops calling tools, step cap. It shipped and works, which retires the biggest risk of this redesign. It also shares the infra bugs this spec fixes elsewhere: `AsyncOpenAI` rebuilt per property access with no timeout (`model.py:46`), zero retries (one transient `ModelError` fails the whole task, `agent.py:144-151`), no streaming, sequential tool execution, an 86,400s blocking `ask` wait (`agent.py:220`), and no transcript persistence (a restart loses the run).

**What is load-bearing and stays:** Smithery per-user OAuth (connection upsert, setup URLs, AuthRequiredError), the tasks/approvals Postgres tables, Pydantic validation at module boundaries, `logging_module.emit_log`, the FastAPI surface.

---

# Proposed Approach

Replace the state graph with a single agent loop. Context is an in-memory message list for the duration of a turn. One streaming completion with `tools=[...]`. If the response contains tool calls, execute them (concurrently when independent), append results as tool messages, and call again. No tool calls means the turn is over. The loop is one function, ~200 lines including budgets and error handling.

**States collapse into prompts and tools.** "Ask the user" is the model ending its turn with a question. Planning is plain text, or a `submit_plan` tool when the structured artifact is needed. Approval is a `request_approval` tool that parks the task; resume is event-driven when the answer arrives (no polling). Routing code (graph.yaml, routers.py, choose_transition, StateOutput, StateHandler discovery) is deleted, not ported.

**One retry policy.** A single cached `AsyncOpenAI(timeout=90, max_retries=0)` client. `generate_response` keeps its 3 attempts with backoff and becomes the only retry layer. The loop itself gets two separate counters: `hops` (LLM calls per turn, cap ~15 for executor, ~6 for chat) and per-tool `attempts` (cap 3, then surface the failure to the model as a tool result and let it decide).

**Real streaming.** The reply completion streams token deltas straight to the SSE response from the first token. Tool-call deltas are accumulated silently and executed at message end. Accumulated text is written to memory once per assistant message, write-behind.

**Persistence becomes write-behind, not read-through.** The turn never re-reads Postgres mid-loop. mem0 retrieval happens once per turn and is injected as one system message. Executor tasks persist their transcript and step cursor after each hop so a restart resumes from the cursor instead of re-executing from step 0.

**Tool layer slims down.** Smithery tool specs pass through to the model as OpenAI tool schemas unchanged. One long-lived `aiohttp.ClientSession`. Connection freshness tracked with a TTL; the pre-call upsert happens only on cold or 401, cutting a warm per-user tool call from 3+ round trips to 1.

**Computer pathway: promote, don't rewrite.** `ComputerAgent.run()` is the reference implementation. Task 2 extracts it into `agent_module/loop.py`, parameterized by toolset, system prompt, event sink, and budgets. `computer_module/model.py` is deleted in favor of the shared client from Task 1 (which brings timeout, retries, and streaming for free). The computer agent's `ask` moves onto the same event-driven approval mechanics as Task 6, replacing the 24h in-process wait, and its transcript persists per hop like executor tasks so runs survive restarts.

**One subagent, three capability families.** There is no "computer task" vs "research task" typing decided upfront. A spawned subagent's manifest includes all three families as first-class primitives it knows it has: MCP services (Smithery), the sandbox (run_command, read/edit/write_file, list_dir, grep, glob), and the browser (`browser_task`). If the subagent decides mid-task it needs to run code, it calls a sandbox tool and the sandbox is provisioned lazily at that moment; if it decides it needs the web, it calls `browser_task`. The model discovers the need; the harness provisions on demand; nothing is decided at spawn time. Browser stays a delegated inner loop behind a single tool (its DOM machinery is too complex to inline, see Task 9), but from the subagent's perspective it is just another primitive in the manifest. This deletes the separate computer task pathway (`computer_plan` state, `computer_router`, dedicated runner/store) in favor of one task path.

**Browser pathway: keep the delegation, put it on a leash.** Re-implementing DOM interaction, element grounding, and page-state management would be a large project for negative payoff; `browser_use` stays as the specialist inner loop, and the main loop sees it as one tool. What changes: (1) the ~40-env-var surface collapses into a `browser` section of config.yaml with ~6 knobs that matter (max_steps, wall_clock, allowed_domains, vision, stealth, model endpoint), the rest pinned to the defaults already chosen in the docstring; (2) the tool emits progress events into the same event stream as other tools (the CDP screencast broker already exists, so the UI can show a live pane instead of a silent 3-minute hang); (3) its LLM client is built from the shared client factory so timeout and retry policy are uniform; (4) its budgets are passed per call by the main loop, not read from process-wide env vars, so the executor can grant a scrape 10 minutes while chat grants 90 seconds.

**Harness boundary: brain, hands, session.** Following the decoupling in Anthropic's Managed Agents writeup (Apr 2026), the system is three interfaces, opinionated about shape, not implementation:

- *Session* — one append-only event log in Postgres, the only durable truth. Every hop appends events (user msg, assistant msg, tool call, tool result, lifecycle transition). Unifies today's three logs (`conversation_context`, task events, `computer_task_events`). The session is NOT the context window: per model call the harness slices the log and transforms it (short window, mem0 injection, cache-friendly ordering). Transformations can change freely per model; only the log's durability is guaranteed.
- *Brain* — `run_turn` plus the lifecycle control plane, stateless by design: nothing in the harness needs to survive a crash; resume is `wake(task_id)` → replay events from cursor. The ONE state machine that survives this redesign lives here: task lifecycle (`pending → running → awaiting_approval → completed | failed | cancelled`) as an enum plus an allowed-transitions map, code-enforced, never LLM-gated. Task status is a projection of the session log. This honors the original CLAUDE.md contract (deterministic transitions) where it actually belongs.
- *Hands* — every side effect behind `execute(name, input) → string`: Smithery MCP tools, sandbox tools, browser tool. `computer_module/tools.py:dispatch()` already has this exact shape. Hands are cattle: a dead sandbox is a tool error fed back to the model, and a fresh one is provisioned on demand.

Coupling rules: the harness is the only writer of lifecycle state; the loop is the only caller of the model; the loop never touches Postgres; the harness never calls the model. They meet at exactly two points: the typed event stream (loop yields, harness translates into session appends and lifecycle transitions) and budgets (harness passes in, loop enforces). Credentials live in neither the brain nor the sandbox: Smithery already implements the vault+proxy pattern (the harness never holds OAuth tokens); keep that boundary.

No open-source orchestration framework. The state graph being deleted was itself a stale-assumption harness in the article's sense (enum-constrained tool choice, LLM transition gates: assumptions about what a 7B cannot do, welded into the architecture). The replacement stays thin precisely so those assumptions can be deleted as models improve. The whole harness is a few hundred lines; the lifecycle FSM is ~20.

**Resilience contracts.** Validated against harness studies of Claude Code and OpenClaw (both implement variants of all nine):

1. *Errors are model input, not control flow.* `execute()` never raises. Every tool call yields exactly one result envelope `{ok, content, error_kind: none|invalid_args|not_found|auth_required|timeout|upstream_error|interrupted, retryable}` where failure content is written FOR the model ("required parameter `page_id` missing", "Notion not connected, give the user this link: ...", "timed out, operation may have completed, verify before retrying"). The loop appends it and continues; the model acknowledges and pivots. Only cancellation propagates: user intent outranks the model. `AuthRequiredError` collapses into `error_kind: auth_required` with the setup URL in content; no dedicated control-flow path.
2. *Typed reasons on every loop decision.* One enum for continues (`next_hop`, `tool_error_pivot`, `model_retry`, `context_recovery`), one for terminals (`completed`, `needs_input`, `max_hops`, `wall_clock`, `model_error`, `cancelled`), both appended to the session log per hop. Tests assert on reasons, not transcript contents. This is the observability that replaces watching state transitions.
3. *Classified, source-tagged retries, one layer.* The model client is the only retry layer. Errors classify to `timeout|connect|rate_limit|server_error` (retry with backoff+jitter, cap 3) or `bad_request|auth` (fail immediately). Every call carries a source tag: `foreground` (user watching) retries fully; `background` (executor hops, mem0 extraction, summaries) fails fast on overload so retries never amplify load on the single vLLM endpoint. Multi-endpoint failover slots in here later as a profile list; classification is the seam.
4. *Circuit breakers on every retry path.* Per run: 3 consecutive model failures → terminal `model_error`; max 2 context-recovery attempts (sticky, never re-arms); per-tool attempt cap 3, after which the failure stands as a permanent tool result and the model must route around it. All breakers are counters in the loop state with explicit reset rules.
5. *Transcript invariant, enforced in the appender.* Every `tool_call` event is closed by exactly one `tool_result` event before any later assistant or terminal event. Abort, crash, and wall-clock paths synthesize `{ok: false, error_kind: interrupted}` results for dangling calls before the terminal event is written. Oversized results are stored in full as a blob; the event carries a head+tail preview plus a reference. The view is capped; the truth is not. Resume folds events into an always-valid message list with zero reader-side repair.
6. *Context recovery ladder, deterministic first.* On overflow: (a) drop old tool results from a whitelist of re-derivable tools (read_file, grep, list ops) from the view; (b) summarize hops older than the last N under a summary contract that preserves task status, counts, and opaque identifiers; (c) terminal `context_overflow` with a remediation hint. Every step is a view transform recorded as an event; the log is never rewritten; recorded decisions replay deterministically on resume so the prompt cache stays warm.
7. *Concurrency by declaration.* Tool specs carry `readonly: bool`. Consecutive readonly calls in one assistant message execute concurrently; mutating calls serialize. Phase 2 (after real streaming): begin executing each tool call the moment its block finishes streaming.
8. *Observation and suggestion (no join).* Every session exposes a subscribable event stream and registers a handle `{queue_message, abort}`. A human or another surface may subscribe to observe and append a suggestion message, injected as a user event before the next hop; the model decides what to do with it, never a direct action. Sessions do not return values: completion writes terminal status + summary to the session's row and log, and surfacing to a user is a UI/notification read, not an injection into another session's reasoning. Two-way takeover (operator executing tools) is deferred.
9. *Constraints at the tool boundary only.* The allowed tool list is computed per run from (agent kind, user, connected services, surface). Approval-required is a per-tool flag checked at execute time; denial returns to the model as an ordinary error result it can react to. Phased availability (plan tool before write-effect tools) is a harness policy over the event log, ~10 lines, never conversation state.

Known cost: pivot quality is model-bound. A 7B pivots less cleverly than a frontier model, so LLM-actionable error text matters more here, not less. Steering and announce-back are new surface; sequence them after the core loop lands.

**Tool surface: manifests, deferral, persist-and-preview.** (Patterns confirmed in both Claude Code and OpenClaw.)

*Manifests.* The allowed tool list is computed per run; separation between agents is the manifest, not separate loop code. A spawned agent's manifest replaces, never inherits, its parent's (no capability leakage):

| Session role | Manifest |
|---|---|
| Interactive (buddy) | Smithery MCP + `create_session` + `read_result` + task-table reads. No sandbox/browser inline: heavy work is forked to a worker session so the chat stream stays responsive. |
| Worker (full generalist) | Smithery MCP + sandbox tools (lazy-provisioned) + `browser_task` + `request_approval` + `finish_task` + `read_result` |
| Browser inner loop | Not a session: one tool wrapping `browser_use`, callable from any manifest that includes it |

Manifest is by role, not by parent/child (there is no parent). Sessions are peers; the only cross-session invariant is that grants and approvals are never shared between sessions.

*Tool-list deferral.* Full JSON schemas for many MCP tools can eat half of an 8k window. If a user's connected services push the manifest past a schema budget, MCP tools are listed name-plus-one-line only, and a `load_tools(query)` tool fetches full schemas on demand (Claude Code's deferred-tools pattern). The validation error for calling an unloaded tool contains the recovery instruction ("load the tool first, then retry"): self-healing, no harness intervention.

*Persist-and-preview for oversized results.* Neither reference harness paginates results in the harness; both cap the view, keep the truth, and let the model page. ARKOS version: any tool result over the per-result view cap (tuned to `tool_result_budget`, ~4k chars) is stored in full as a blob in the session store; the tool_result event carries a head+tail preview plus a `ref`. A `read_result(ref, offset, limit)` tool in every manifest lets the model page through the full content on demand. Truncation text always states what was cut and how to get it ("200,113 chars total, showing first/last 2,000; use read_result to page"). Read-style sandbox tools already take offset/limit and are exempt from persistence (persisting what the model re-reads is circular). Per-hop aggregate budget: if one hop's parallel results together exceed the view budget, the largest are demoted to preview+ref first.

**Sessions, Looking Glass, and the task UX.**

*No subagents; peer sessions.* A session is one run of the loop with its own log, budget, and lifecycle row. Buddy is the interactive session bound to a chat surface. It can create other sessions via `create_session(goal)`, but this is fork-and-forget, not spawn-and-await: created sessions run to terminal independently and never return a value into the creator's reasoning. Coordination is through the shared task table and session logs, pulled when asked (buddy reads task status as a normal tool call), never pushed back as a return. A `created_by` pointer is kept for provenance only, with no runtime coupling. The word "subagent" is retired: there are only sessions, all peers running the same loop; capability differs by role, not tier.

*Why Looking Glass is in scope, not north-star.* Because sessions do not return, the human stays in the loop by observing and suggesting, not by receiving output. That makes a live window into each session load-bearing: the no-return model is only livable with it. The current behavior (`task_runner.py`: run blind to terminal, then `mark_task_completed` with a summary string) is exactly the UX this replaces.

*Looking Glass v1 is one-way plus suggestions:*
- *Observe.* Any session's typed event stream is subscribable and rendered live (content, tool_start/result, status, done). Watching a task uses the same UI component as the chat window; the only difference is the model drives autonomously instead of turn-taking.
- *Suggest.* A human can send a message into any session, exactly like typing in chat. It is appended as a user event and injected at the next hop boundary; the model decides what to do with it. A suggestion, never a direct action.
- *Deferred (two-way).* Human tool takeover ("execute on its behalf") is out of v1. Because the human never writes to the sandbox, browser, or state directly, v1 has a single actor per resource, so the computer/browser/state races do not arise. Takeover is what introduces them and is phased to a later pass with hop-boundary arbitration.

*UX change.* The task tray of status rows becomes live session windows you attach to, watch, and suggest into. `needs_input` stops being a dead-ended approval-tray item and becomes an in-window interactive pause: the session asks, you answer in its window, it continues. Buddy chat and task-watching converge on one component pointed at different sessions.

**Product surface: Projects, Command Center, Looking Glass.** The session/table model needs a shell that puts the human back in the driver's seat.

*Projects.* A project is the durable container; its base unit is a task (a session). A project owns working files and project memory, both shared across its tasks. Grouping related work under a project makes long-horizon work legible instead of a flat list of runs. This also answers the earlier separate-filesystem-from-computer question: files live at the project (durable, user-visible), and the sandbox is ephemeral compute a task mounts them into. Each new task spawns its own project by default; it joins an existing project only when explicitly pointed at one. So the project grid is also the task list.

*Command Center — attention as a flow, surfaced at every scope.* An approval or `needs_input` is not owned by any one surface: it is a lifecycle state on a task, and a task belongs to a project, so it surfaces simultaneously wherever that task is visible — inline in the task's Looking Glass window, in its project's attention list, and in the global Command Center. All three are projections over the same lifecycle rows, filtered by task, by `project_id`, and by user respectively; nothing is duplicated. It is resolvable from any of them, and resolving it anywhere writes the same respond event and wakes the task at its cursor. The Command Center is simply the widest projection (triage across all projects in one place); the project attention list is the same query scoped to one project; the in-window pause is the same scoped to one task.

*Looking Glass.* Two levels. The landing is a grid of project bubbles (rounded cards), each with a status dot in the top-right that is a projection of the project's lifecycle rollup: green = working, ochre = needs attention (an approval or needs_input is waiting), red = failed, gray = done. Clicking a bubble opens the detail view: the streaming session UI where you observe live, suggest (a message injected at the next hop), and upload files, with working-files and project-memory tabs alongside the stream. Looking Glass subsumes the old "tasks" nav item, which is removed (the desk keeps its active-tasks zone). The project grid doubles as the task list, so a separate tasks view is redundant.

*Three storage scopes, distinct.* (a) Session log — per task, the event transcript, durable truth for that run. (b) Project files — durable, user-visible, uploadable; mounted into a task's sandbox at provision; the filesystem separated from the computer. (c) Memory — project memory (shared across a project's tasks) plus user long-term memory (cross-project), both distinct from per-session short-term context.

**Not in scope:** swapping the model (the new loop makes this trivial later, but this spec targets Qwen 2.5 on vLLM), frontend changes beyond consuming the same SSE shape, replacing `browser_use` with a bespoke browser agent, multi-worker task leasing (tracked as a follow-up task card).

**Deletion inventory (the LOC case):**

| Module | Now | After | Notes |
|---|---|---|---|
| state_module | 1,708 | ~0 | graphs, routers, discovery, both agent packages |
| agent_module | 535 | ~250 | loop.py (generalized from ComputerAgent.run) replaces step/step_stream/choose_transition |
| base_module | 2,564 | ~1,700 | event-driven approvals, resume-from-cursor, no orphan re-execution |
| tool_module | 1,930 | ~1,450 | drop discovery scan, scoped wrapper, back-compat shims; browser tool stays, config surface shrinks |
| model_module | 442 | ~150 | one client, one retry policy, llm_json mostly dies |
| computer_module | 1,757 | ~650 | sandbox manager + toolset stay; agent/model/router/runner/store fold into the unified task path |
| tests | 5,037 | ~2,200 | tests of deleted machinery go with it |
| **Total (py)** | **~14.9k** | **~6.8k** | production ~9.9k → ~4.6k |

Migration is replace-then-delete: `loop.py` lands beside the old code behind a config flag; old modules are deleted one PR each only after the new path carries chat and one real executor task end to end.

---

# Implementation Plan

## Task 1: Model client rewrite

**Problem:** Client rebuilt per access, no timeout (600s SDK default), hidden SDK retries, retry sleep after final attempt, streaming path has no retries.
**Done when:** One cached `AsyncOpenAI(timeout=90, max_retries=0)` per ArkModelLink; `generate_response` is the only retry layer (3 attempts, backoff, no trailing sleep); `generate` and `generate_stream` share the client and accept `tools=[...]`; worst-case single call ≤ ~5 min.
**Touch point:** `model_module/ArkModelNew.py`
**Priority:** P0 | **Effort:** 1 day | **Blockers:** none
**Out of scope:** the loop itself; any caller changes beyond the constructor.
**Acceptance test:** mock a hung endpoint; a call fails in ≤ 3 x timeout + backoff, exactly 3 requests observed, no event-loop stall.

## Task 2: Core loop (`agent_module/loop.py`), generalized from ComputerAgent

**Problem:** 3-5 LLM calls and 2+ DB reads per hop, routed through a state graph that exists to compensate for not using native tool calling. The correct loop already exists in `computer_module/agent.py:96` but is welded to the sandbox toolset.
**Done when:** `run_turn(messages, tools, budgets) -> AsyncIterator[event]` (extracted from `ComputerAgent.run`, parameterized by toolset, system prompt, event sink, budgets) executes the call → tool_calls → append → call cycle; streams text deltas; validates tool args against the tool's inputSchema with one repair retry; separate `hops` and per-tool `attempts` counters; yields typed events (`content`, `tool_start`, `tool_result`, `done` with terminal reason); `ComputerAgent` becomes a thin configuration of it.
**Touch point:** new file `agent_module/loop.py`; `computer_module/agent.py` shrinks to config + sandbox glue
**Priority:** P0 | **Effort:** 2-3 days | **Blockers:** Task 1
**Out of scope:** memory, persistence, FastAPI wiring (Tasks 4-5).
**Acceptance test:** with a mocked model emitting two tool calls then a text reply, the loop makes exactly 3 LLM calls, executes tools concurrently, and terminates with reason `completed`; existing computer-agent live tests still pass on the extracted loop.

## Task 3: Tool layer slim-down

**Problem:** New TLS session per call, upsert PUT + tools/list before every per-user call, sequential 30s discovery scans, broken async wrappers in scoped.py.
**Done when:** One `ClientSession` owned by SmitheryManager (created at startup, closed at shutdown); per-connection freshness TTL (~10 min) with re-upsert only on cold/401; tool specs exposed as OpenAI tool schemas via `get_openai_tools(user_id)`; `tools/call` timeout raised to 120s (PUT/list stay 30s); scoped.py deleted; per-user connections persisted to a `user_connections` table and rehydrated on startup.
**Touch point:** `tool_module/smithery.py`, new migration, delete `tool_module/scoped.py`
**Priority:** P0 | **Effort:** 2 days | **Blockers:** none
**Out of scope:** changing the Smithery API contract; new auth patterns.
**Acceptance test:** warm per-user tool call performs exactly 1 HTTP request; restart + first request performs 1 DB read and 0 Smithery PUTs for already-connected servers.

## Task 4: Chat on the new loop, behind a flag

**Problem:** step_stream fakes streaming (full completion, then per-char SSE envelopes); per-request mem0 init when X-Session-ID is missing; unbounded memory cache; no SSE error path.
**Done when:** `agent.loop_v2: true` routes `/v1/chat/completions` through `run_turn`; first token reaches the client before generation completes (measured); one shared Mem0 instance at startup, retrieval once per turn; `_memory_cache` bounded (LRU 500); SSE emits an error chunk + `[DONE]` on mid-stream failure; memory writes are write-behind.
**Touch point:** `base_module/app.py`, `memory_module/memory.py`
**Priority:** P0 | **Effort:** 2 days | **Blockers:** Tasks 1-3
**Out of scope:** frontend changes; deleting old step_stream (Task 7).
**Acceptance test:** integration test asserts first SSE content chunk arrives before the mocked model finishes emitting, and a forced mid-stream exception yields an error chunk, not a truncated stream.

## Task 5: Executor on the new loop

**Problem:** Conflated iteration budget, false-green completion on max_iter, tool exceptions killing tasks before the recovery path runs, restart re-executes completed steps, PATCH-cancel overwritten to failed.
**Done when:** Executor runs `run_turn` with a checklist system prompt; transcript + step cursor persisted after each hop; restart resumes from cursor; task completion requires an explicit `finish_task` tool call (never inferred from a completion signal); terminal reason recorded on the task row; budgets: hops cap, per-tool attempts cap, wall-clock cap (default 30 min); cancel wins all races.
**Touch point:** `base_module/task_runner.py`, `base_module/task_store.py`, `base_module/tasks.py`
**Priority:** P1 | **Effort:** 3 days | **Blockers:** Task 4
**Out of scope:** multi-worker leasing (Task 8); approval mechanics (Task 6).
**Acceptance test:** kill the process mid-task, restart, task resumes at its cursor with no duplicate tool executions (assert via tool-call log); a task that exhausts its hop budget is marked `failed` with reason `max_hops`.

## Task 6: Event-driven approvals

**Problem:** 2s synchronous DB poll on the event loop for up to 24h, then hard failure; respawn orphans pending approvals.
**Done when:** `request_approval` tool creates the approval row, persists the transcript, sets `awaiting_approval`, and returns from the loop (no polling); `POST /approvals/{id}/respond` appends the answer to the transcript and re-spawns the task at its cursor; no timeout fails the task, a reminder notification fires at 1h; zero DB polls while waiting.
**Touch point:** `base_module/tasks.py`, `base_module/task_store.py`, executor tool registry
**Priority:** P1 | **Effort:** 2 days | **Blockers:** Task 5
**Out of scope:** notification channels beyond the existing Slack DM.
**Acceptance test:** task waits on approval with zero queries issued during the wait (assert via query counter); respond resumes it within one dispatch cycle; process restart while waiting preserves the pending approval.

## Task 7: Delete the old machinery

**Problem:** Two parallel implementations double the maintenance surface; the point of the redesign is thousands of lines removed.
**Done when:** state_module (both agent packages + core), old step/step_stream/choose_transition, llm_json repair paths, and their tests are deleted; flag removed; loop_v2 is the only path; CLAUDE.md architecture contracts rewritten for the loop model; total production LOC ≤ ~5k.
**Touch point:** `state_module/`, `agent_module/agent.py`, `model_module/llm_json.py`, `tests/`, `CLAUDE.md`
**Priority:** P1 | **Effort:** 2 days (one PR per module, per the 300-line PR rule this is several PRs) | **Blockers:** Tasks 4-6 stable for one week of real use
**Out of scope:** browser tool internals (Task 9); sandbox lifecycle.
**Acceptance test:** full test suite green; `grep -r "StateHandler\|StateOutput\|graph.yaml" --include="*.py"` returns nothing outside tests of deleted code (which are also gone).

## Task 8: Unify computer into the subagent manifest

**Problem:** The computer agent is a separate pathway (`computer_plan` state, `computer_router`, dedicated runner/store/model client) that forces an upfront "is this a computer task?" decision. Sandbox and browser should be first-class primitives any subagent reaches for mid-task. `computer_module/model.py` duplicates the model client with the same bugs Task 1 fixes; `ask` blocks in-process for up to 24h (`agent.py:220`); no transcript persistence; MCP schemas hand-rolled (`agent.py:63-85`).
**Done when:** The sandbox toolset (run_command, read/edit/write_file, list_dir, grep, glob, todo_write) joins the unified subagent manifest behind `execute()`; sandbox provisioned lazily on first sandbox-tool call (not before the first model call as in `agent.py:108`), so a task that never runs code never boots a container; `computer_module/model.py` deleted in favor of the Task 1 client; the separate computer task pathway (`computer_router`, dedicated runner/store) folds into the unified tasks path, endpoints kept as thin shims if the frontend needs them; `ask_user` uses Task 6 approvals; transcript + cursor persisted like any task; verified e2b sandboxes carry no credentials in env (tokens stay behind the Smithery proxy).
**Touch point:** `computer_module/*`, `base_module/tasks.py`, subagent manifest config
**Priority:** P1 | **Effort:** 3 days | **Blockers:** Tasks 1, 2, 6
**Out of scope:** sandbox lifecycle internals (SandboxManager is fine); the sandbox toolset behaviors themselves.
**Acceptance test:** one spawned task calls a Notion MCP tool, then writes and runs code in the sandbox, then calls `browser_task`, in a single run with no re-spawn and no upfront task typing; the sandbox boots only at the first sandbox-tool call; kill the process mid-task, restart, it resumes at its cursor.

---

## Task 9: Browser tool on a leash

**Problem:** The browser pathway is a fourth loop with its own LLM client, ~40 env vars of process-wide config, no progress events to the main event stream (a silent 3-minute blocking call from the user's perspective), and budgets that callers cannot vary per task.
**Done when:** Config collapses to a `browser:` section in config.yaml (max_steps, wall_clock, allowed_domains, vision, stealth, model endpoint; everything else pinned to current defaults); budgets accepted per call from the invoking loop; progress events (step, url, action) emitted into the same event stream as other tools alongside the existing CDP screencast; LLM client built from the shared factory with uniform timeout/retry policy.
**Touch point:** `tool_module/browser_tool.py`, `config_module/config.yaml`
**Priority:** P2 | **Effort:** 2 days | **Blockers:** Task 1
**Out of scope:** replacing `browser_use`; vision model swap; changes to `browser_actions.py` custom actions.
**Acceptance test:** a browser task invoked with `max_seconds=60` is killed at 60s regardless of env; progress events appear in the task event log while the browser runs; zero `BROWSER_USE_*` env vars read at call time.

---

## Task 10: Looking Glass v1 — observe + suggest

**Problem:** A running task is invisible: it runs blind to terminal, then reports a summary string on a row (`task_runner.py` `mark_task_completed`). For long-horizon autonomous work with no return-to-parent, the human's only way to stay in the loop is to watch and suggest, and neither exists today.
**Done when:** Any session's typed event stream is subscribable over SSE and rendered live in a session window (same component as chat); a human can append a suggestion message to any session, injected as a user event at the next hop; `needs_input` renders as an in-window pause answered inline (not a separate approval tray); the task tray becomes a list of attachable session windows. Strictly one-way plus suggestions: no human tool execution, no cursor/screen control.
**Touch point:** `base_module/app.py` (per-session SSE subscribe + suggestion endpoint), frontend session-window component
**Priority:** P1 | **Effort:** 3 days | **Blockers:** Tasks 2, 4 (typed event stream + real streaming), Task 6 (event-driven pause)
**Live updates:** the window subscribes via `EventSource` and applies events as deltas (push, not poll); this is the mechanism Task 12 rolls out across every surface, replacing the 6s `setInterval`.
**Out of scope:** two-way takeover (operator executing tools, cursor/screen control) — deferred to a later pass with hop-boundary resource arbitration; the computer/browser/state races only arise there.
**Acceptance test:** open a running task, see its tool calls and text stream live; type a suggestion mid-run and observe it injected as a user event at the next hop; a session that hits `needs_input` is answered in-window and continues; no code path lets the human execute a tool in the session.

---

## Task 11: Projects, Command Center, and project storage

**Problem:** Work is a flat list of tasks with no durable container, no shared files, no scoped memory; the human has no organizing home and no single attention queue.
**Done when:** a `projects` table with tasks carrying `project_id`; project files stored durably (user-visible, uploadable) and mounted into a task's sandbox at provision (the fs/compute split); a project memory scope (mem0 keyed by `project_id`) alongside user long-term memory; attention items (`awaiting_approval` + needs-input) surfaced as one query at three scopes — inline in the task window, in the project's attention list, and in the global Command Center — resolvable from any surface and writing the same wake event; Looking Glass windows nested under their project with working-files and project-memory tabs.
**Touch point:** new migration (`projects`, `project_files`), `base_module` tasks/endpoints, `memory_module` (project scope), frontend IA
**Priority:** P2 | **Effort:** 4 days | **Blockers:** Tasks 6 (approvals), 8 (sandbox mount), 10 (Looking Glass)
**Out of scope:** multi-user project sharing/permissions; project templates.
**Acceptance test:** create a project, upload a file, start a task that reads the uploaded file from its sandbox; the task's approval appears in both the project's attention list and the global Command Center, and resolving it from either surface resumes the task at its cursor.

---

## Task 12: Frontend modernization — build step, push, motion

**Problem:** The UI is prototype-grade and feels dated: it loads React's *development* build from a CDN and runs Babel *in the browser* (`@babel/standalone`), transpiling every JSX file on each page load, so it paints blank then pops in. State is 6s `setInterval` polling (`app.jsx:105`) that snaps between ticks; there are no optimistic updates and no motion, so actions wait for a round-trip plus the next poll and nothing animates. That combination, not any one bug, is why it feels like a 90s page next to modern apps.
**Done when:** a real build (Vite): production React, minified, code-split, zero runtime Babel, sub-1s first paint; all state push-based via one `EventSource` per surface (Looking Glass / project / Command Center) with `id: <seq>` + `Last-Event-ID` gap-free replay on reconnect, and `setInterval` polling deleted; actions are optimistic (apply locally, reconcile on the returned lifecycle event); enter/exit and layout transitions on lists (tasks, approvals, messages) plus skeleton loading instead of blank states; streamed text renders smoothly.
**Touch point:** `frontend/*` (rebuild), `index.html`, build tooling
**Priority:** P1 for perceived quality | **Effort:** 4-5 days | **Blockers:** Task 4 (real streaming), Task 10 (per-session SSE)
**Out of scope:** a full design-system overhaul beyond motion + skeletons; native apps.
**Acceptance test:** cold load paints the shell in under ~1s with no blank-then-pop; a task state change animates in with no poll; clicking cancel updates the row instantly and reconciles on the event; killing and restoring the network resumes the stream from the last `seq` with no missed transitions.

---

## Task 13: Multi-worker safety (follow-up)

**Problem:** `_RUNNING` is per-process and sweep_orphans has no claim, so >1 worker duplicates every task's side effects.
**Done when:** Task dispatch claims via `UPDATE ... WHERE locked_by IS NULL` (or `FOR UPDATE SKIP LOCKED`); a semaphore bounds concurrent tasks per worker; stale leases expire.
**Touch point:** `base_module/task_runner.py`, migration
**Priority:** P2 | **Effort:** 1-2 days | **Blockers:** Task 5
**Out of scope:** distributed queues (Postgres is enough at current scale).
**Acceptance test:** two workers + 10 tasks: each task executes exactly once (assert via tool-call log uniqueness).

---

# Tests

## Test 1: `test_loop_single_call_per_hop`

**What it verifies:** One user turn with one tool use makes exactly 2 LLM calls (tool call + final reply), not 4-5.
**Why this matters:** The call count IS the latency thesis of the redesign; regression here silently reintroduces the old cost.

## Test 2: `test_streaming_first_token_before_completion`

**What it verifies:** First SSE content chunk is emitted before the mocked model finishes generating.
**Why this matters:** Guards against any future "buffer then replay" regression; perceived speed lives or dies here.

## Test 3: `test_retry_budget_bounded`

**What it verifies:** With a permanently failing model endpoint, a turn fails in ≤ 3 attempts and ≤ ~5 minutes wall clock, with exactly 3 HTTP requests.
**Why this matters:** The old stack allowed 99 requests / 16.5 hours; this pins the new single-retry-layer contract.

## Test 4: `test_executor_resume_no_duplicate_side_effects`

**What it verifies:** Process death mid-task, then restart: completed tool calls are not re-executed; the task resumes at its cursor.
**Why this matters:** Duplicate side effects (double emails, double writes) are the worst externally visible failure a task runner can have.

## Test 5: `test_failed_task_never_reports_completed`

**What it verifies:** Exhausting the hop budget, a tool exception storm, or cancel mid-run each end with the correct terminal status; `completed` requires an explicit `finish_task` call.
**Why this matters:** The current fallback marks runaway tasks green, which destroys trust in the task list.

## Test 6: `test_malformed_tool_call_recovery`

**What it verifies:** A model emitting invalid tool args gets the validation error fed back and succeeds on the retry; two consecutive failures surface as a tool error result, not a crash.
**Why this matters:** This is the safety net that replaces the entire enum-constrained state graph; it must demonstrably work on the 7B model.

---

# Open Questions

1. Qwen 2.5 7B tool-calling reliability under the hermes parser at temperature 0.7: what is the malformed-call rate on our real tool schemas? Measure before Task 7 deletes the fallback machinery. If above ~5%, consider temperature 0.2 for tool-bearing turns or constrained decoding via vLLM guided_json.
2. ~~computer_module (1,757 lines): is it a bespoke loop that collapses into `run_turn` with a different tool set, or does it have real unique machinery?~~ Resolved 2026-07-03: audited. It IS the target loop already (native tool calling, message list, step cap). Task 2 generalizes it; Task 8 puts it on shared infra. Unique machinery worth keeping: sandbox toolset with read-before-edit enforcement, tool descriptions as scaffolding.
6. Browser tool inner-loop model: browser_use currently runs text-only (vision off) against Qwen. Is grounding accuracy acceptable, or does the browser pathway justify a small VL model before anything else in this spec touches it?
7. Session log unification: merging `conversation_context`, task events, and `computer_task_events` into one append-only log is the clean end state, but is it a prerequisite for Tasks 5/8 or a follow-up migration? Leaning follow-up: resume-from-cursor only needs the task transcript, not the unified log.
8. Memory scoping (Task 11): is project memory a hard third tier (mem0 keyed by `project_id`) or user memory filtered by a project tag? And does a project's interactive session auto-load project memory + a working-file index into context on every turn, or only on request? The auto-load choice trades context budget against the session knowing what's in the project.
3. Parallel tool execution: safe default, or opt-in per tool? Write-effect tools (send message, create page) may need sequential ordering guarantees.
4. Does anything downstream consume `StateOutput.structured_data` besides the runner and SSE layer? Grep before Task 7.
5. Model swap: once the loop is OpenAI-native, is a hosted frontier model for the executor (keeping local 7B for chat) worth the cost? Out of scope here, but the redesign makes it a config change.

---

# Implementation Notes

*Add entries here as work lands.*
