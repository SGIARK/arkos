# Mega Spec: ARKOS Personal Computer

*One self-contained implementation spec. It subsumes the identity/registry tasks from `MULTIUSER_SPEC.md` (folded in as Phase 0), builds the per-user persistent computer + the agent that operates it, wires buddy's dispatch, and solves completion notification ("send a message back when it's done"). Written to be implemented end-to-end in one pass. `PLATFORM_SPEC.md` is the vision; this is the build.*

**Sources / current code (read these before implementing)**

- `base_module/jwt_utils.py` -- `get_current_user`, `CurrentUser`, `issue_token`, `decode_token` ALREADY EXIST. Phase 0 wires them in and gates the legacy fallback.
- `base_module/users.py` -- `/auth/demo-login` (issues a JWT, no password) and `/auth/me` already exist. Token issuance is solved.
- `base_module/app.py` -- per-request agent (`_make_agent`), per-user memory cache (`_get_or_create_memory`), chat endpoint with streaming (`step_stream` + `StreamingResponse`), `_system_prompt`/`_available_tools` globals, `tool_manager` singleton. Endpoints currently read `X-User-ID` directly.
- `tool_module/smithery.py` -- `SmitheryManager`, global `_tool_registry` (tool->server, NO user dimension), `_user_tools[user_id][server]`, `_pending[user_id][server]`, `call_tool(tool_name, arguments, user_id)`, `AuthRequiredError`.
- `base_module/task_runner.py` + `base_module/task_store.py` -- async background task pattern: `spawn(task_id)`, `log_event(task_id, kind, content, payload)`, `task_events` table, `mark_task_completed/failed`, `set_task_status`. The approval tray: `create_approval`, `get_approval`, `resolve_approval`, `list_pending_approvals`.
- `state_module/agent_executor/state_approval.py` -- the human-in-the-loop poll-an-approval pattern to mirror.
- `memory_module/memory.py` -- `Memory(user_id, session_id, db_url, use_long_term)`, `add_memory(message)` writes to `conversation_context (user_id, session_id, role, message)`. This is how completion posts a message into the user's chat.
- `model_module/ArkModelNew.py` -- `ArkModelLink` (the local model client; the computer-agent runs on this for the MVP), `AIMessage`/`UserMessage`/`SystemMessage`.
- `config_module/config.yaml` -- has an `openhands:` section (rename to `computer_agent:`), plus `llm.*` (local model) and `smithery.*`.
- e2b SDK (`e2b` / `e2b-code-interpreter`): `Sandbox.create()`, `sandbox.commands.run(cmd)`, `sandbox.files.write(path, data)`, `sandbox.files.read(path)`, `sandbox.files.list(path)`, pause/resume (`sandbox.pause()` -> id, `Sandbox.resume(id)` -- verify exact API against the installed version).

**Status:** Not started | **Author:** | **Last updated:** 2026-06-03

---

# Problem

ARKOS can chat (buddy, local Qwen) and make MCP tool calls (Smithery). It cannot do real computer work -- read/write/run files, iterate over a session, keep durable state -- and it cannot do *any* of it safely for multiple users, because identity is unverified and tool routing is global. Five concrete gaps:

1. **No per-user computer.** The only "execution" is the in-process executor iterating MCP calls. No filesystem, no shell, no persistence, no isolation.
2. **No capable worker + dispatch.** No agent that drives a filesystem/shell, and no path for buddy to hand a heavy task off to one and get out of the way.
3. **No completion notification.** An async task has no way to tell the user "I'm done, here's the result" -- the thing the user explicitly needs.
4. **Identity is forgeable (gates everything above).** Endpoints trust `X-User-ID`; any client can impersonate any user. Exposing a per-user filesystem + credentials on top of this leaks one user's computer to another.
5. **Tool routing is global (gates MCP-in-the-computer).** `_tool_registry` has no user dimension, so two users with the same tool name cross-route.

**Success:** an authenticated user tells buddy "write a python script that does X and run it." Buddy routes it to the computer-agent in *that user's* persistent sandbox. The agent writes the file, runs it, streams its commands to a Computer tab + activity view (with a "using the computer" indicator), and **when it finishes, a message appears in the chat: "Done -- here's what I built."** The file is still there next session. User B's task runs in a separate sandbox with separate tools and never sees A's anything.

---

# Technical Background

**The core primitive.** A per-user, persistent, isolated sandbox with an exec API -- durable filesystem + run-commands. Persistence via e2b **pause/resume** (hibernate with full state, wake in seconds); compute ephemeral, state durable. Isolation is e2b's job: one sandbox per user, filesystem never shared.

**The model is a swappable knob (MVP = local Qwen).** Because the sandbox is owned separately, the agent's model is swappable. For the MVP the computer-agent runs on the **local Qwen** (reuse `ArkModelLink`) -- proves the whole pipeline for free. Swapping to a frontier model later is changing `computer_agent.llm` to point at an Anthropic endpoint; no rewrite, because the agent only talks to a generic tool interface. Honest caveats on Qwen: real multi-step coding will be rough and self-termination is poor, so the **step cap is load-bearing**, and early demos prove "the pipeline works," not "it codes well."

**Headless, terminal-first.** Tools are `run_command` / `read_file` / `write_file` (+ MCP) -- a shell and files, not GUI computer-use. Covers ~90% of work, faster + auditable.

**One worker, MCP native (like Claude Code).** The computer-agent holds file/shell tools **and** the user's MCP tools (Smithery proxy) from the start. One agent, one context, for tasks needing both. MCP via Smithery needs no raw token (Smithery holds them write-only and proxies); only future `git push` needs a real token (deferred).

**Completion notification -- the mechanism (this is the part the user wasn't sure about).** An async task tells the user it's done through three durable writes plus one live push:
1. **Status flip** -- `computer_tasks.status = 'completed'` (or `'failed'`).
2. **Final event** -- a `task_events`-style row `kind='completed'` with the result summary + output paths.
3. **Chat message injection** -- write an assistant message into the user's *chat* `conversation_context` via `Memory.add_memory(AIMessage(...))`, addressed to the **chat session id captured at dispatch**. This is what makes "a message appears in the chat saying it's done" literally true -- the result lands in the same conversation the user reads.
4. **Live push (SSE)** -- a Server-Sent-Events stream the Computer tab / chat subscribes to, so progress and the completion land in real time without polling. Polling the events table is the fallback if no SSE client is connected.

So "send a message back when done" = (chat injection for durability + SSE for liveness). Same mechanism carries progress events while the task runs.

**Identity already mostly exists.** `jwt_utils.get_current_user` + `CurrentUser` are written; `/auth/demo-login` issues tokens; `/auth/me` reads them. The work is: (a) make every per-user endpoint *depend* on `CurrentUser` instead of reading `X-User-ID`, and (b) gate the legacy `X-User-ID` fallback inside `get_current_user` behind `ARK_DEMO_MODE`.

**Patterns to reuse (do not rebuild):** `_get_or_create_memory` (per-user lazy cache) -> per-user sandbox cache. `task_runner.spawn` + `task_events` -> the computer-task runner + progress. `create_approval`/`get_approval` -> the agent's structured asks. `StreamingResponse` (already used for chat) -> the SSE stream.

---

# Architecture & Data Flow

```
 User (web/mobile, holds JWT)
   │  Authorization: Bearer <token>   ── verified by CurrentUser ──► user_id
   ▼
 BUDDY (Qwen)  ── routes by task weight ──┐
   │  conversational → answer              │ trivial single MCP lookup → buddy-direct call (no computer)
   │                                       │
   └─ "needs computer" ───────────────────┘
                │ create computer_tasks row (user_id, chat_session_id, prompt, status=pending)
                │ spawn async runner;  reply to chat: "started, I'll let you know"
                ▼
        COMPUTER-AGENT RUNNER (async, local Qwen)
           loop: model → tool calls → execute → feed back → until done | step cap
           tools: run_command / read_file / write_file  +  MCP (Smithery, user-scoped)
                │  every tool use → task_event(kind-tagged)  → SSE + activity view
                │  needs human input → create_approval → tray → resolve → continue
                ▼
        ON DONE:  status=completed
                  + final task_event(kind=completed, result)
                  + Memory.add_memory(AIMessage(result)) into chat_session_id   ← "message back that it's done"
                  + SSE push
                  + pause sandbox (idle)
           ▲
           │ operates
        e2b PERSISTENT SANDBOX  (per user; FS persists, compute pauses)
           ▲ viewed by
        COMPUTER TAB (filesystem)   +   ACTIVITY VIEW (streamed commands + "using computer" icon)
```

Two new tables (`user_sandboxes`, `computer_tasks`), one SSE endpoint, two filesystem endpoints, the auth wiring, the registry scoping, the runner, the agent.

---

# Implementation Plan

Ordered by dependency. Each task is self-contained; `Depends on` defines sequence. No time estimates -- implement straight through.

## Phase 0 -- Identity & routing (the gate; folded from MULTIUSER_SPEC)

### Task 1: Verify identity on every per-user endpoint  ✅ DONE

*All per-user `app.py` endpoints depend on `CurrentUser`; `X-User-ID` fallback gated behind `ARK_DEMO_MODE` (on in dev `.env`); `/oauth/callback` keeps query-param identity as documented exception (UNSAFE_DECISIONS U1). Committed.*

**Problem:** Endpoints read `X-User-ID` (`app.py:238,299,344,371,483`); any client impersonates any user. The dependency exists but is unused, and its legacy fallback is ungated.

**Done when:**
- Every per-user endpoint in `app.py` takes `current: dict = CurrentUser` and uses `current["user_id"]` -- never `request.headers.get("X-User-ID")`. Covers: the chat/completions endpoint, `/services`, `/services/{service}/connect`, `/disconnect`, `/oauth/callback`, and all task + new computer endpoints.
- In `jwt_utils.get_current_user`, the `x_user_id` legacy branch is taken **only if** `os.environ.get("ARK_DEMO_MODE")` is truthy; otherwise a missing/invalid Bearer returns 401 (no fallback identity).
- A request with a valid token for A but `X-User-ID: B` operates as A (Bearer wins; the header is ignored when a token is present -- already true since the token branch returns first, but assert it in a test).

**Touch points:** `base_module/app.py` (all per-user endpoints), `base_module/jwt_utils.py` (gate the fallback).

**Depends on:** none.

**Out of scope:** roles/RBAC, refresh tokens, new login UX (`/auth/demo-login` already issues tokens).

**Acceptance test:** `test_missing_token_rejected_outside_demo`, `test_header_cannot_override_verified_identity`.

### Task 2: User-scope the tool registry  ✅ DONE

*`_tool_registry` is shared-only; new `_user_tool_registry[user_id][tool]`; `_resolve_server` (user-first then shared) on `call_tool`, buddy, and executor; `list_all_tools(user_id)` no longer unions across users; `_pending` pop + `reset`/disconnect fixed. 6 isolation tests pass. Committed.*

**Problem:** `_tool_registry: dict[tool->server]` (`smithery.py:257`) has no user dimension; users with colliding tool names cross-route (`:434` reads it for any user).

**Done when:**
- `_tool_registry` becomes `dict[user_id, dict[tool_name, server_name]]`, mirroring `_user_tools`. Writes (`:319,:373`) and reads/scans (`:434,:457`) are user-scoped.
- `call_tool(tool_name, arguments, user_id)` resolves the server from `self._tool_registry.get(user_id, {})` only; a tool absent from that user's map raises `AuthRequiredError` / a clear "not connected", never another user's server.
- Fix the `_pending.get(user_id, {}).pop(...)` no-op (`:375`) so pending entries actually clear; fix `reset()` (`:512`) to clear the nested structure.

**Touch points:** `tool_module/smithery.py`.

**Depends on:** none.

**Acceptance test:** `test_tool_registry_is_user_scoped`, `test_colliding_tool_names_do_not_cross_users`.

### Task 3: Concurrency-safe shared state + secrets fail-fast  ✅ DONE

*`assert_secure_secret()` refuses to boot on the default JWT secret outside demo mode; `memory.py` key override removed. Shared state found already race-free under single-threaded asyncio (reassign-not-mutate, no await in the memory factory) -- pinned with comments instead of needless locks. Committed.*

**Problem:** `_memory_cache`, `_system_prompt`, `_available_tools` mutated across concurrent requests without locks (`app.py:61,194`); JWT secret defaults to a public string; `memory.py:42` overwrites the LLM key.

**Done when:**
- `_get_or_create_memory` is guarded by an `asyncio.Lock` (keyed or global) so a user's `Memory` is created exactly once under concurrency.
- `_system_prompt` / `_available_tools` are rebuilt-then-assigned atomically (no in-place mutation a reader can observe half-built).
- Startup refuses to boot if `ARK_JWT_SECRET` is unset/equals `"ark-dev-secret-change-me"` and `ARK_DEMO_MODE` is not set.
- Remove `os.environ["OPENAI_API_KEY"] = "sk"` from `memory.py:42`.

**Touch points:** `base_module/app.py`, `base_module/jwt_utils.py` (or a startup check in `app.py`), `memory_module/memory.py`.

**Depends on:** Task 2 (registry shape).

**Acceptance test:** `test_concurrent_requests_get_isolated_memory`, `test_startup_fails_on_default_secret_in_prod`.

## Phase 1 -- The computer (sandbox layer)

### Task 4: SandboxManager -- per-user persistent sandbox lifecycle  ✅ DONE

*Built and verified against live e2b + Postgres. `computer_module/sandbox.py` (`SandboxManager`, module singleton `sandbox_manager`), migration `db/migrations/0005_user_sandboxes.sql` (note: 0005, not 0006 -- next free number in the repo). Resume uses `Sandbox.connect(sandbox_id)`; `pause()` returns None. Persistence + per-user isolation confirmed.*

**Problem:** No per-user execution environment with durable state.

**Done when:**
- New `compute_module/sandbox.py` with an async `SandboxManager` singleton:
  ```python
  class SandboxManager:
      async def get_or_create(self, user_id: str) -> "SandboxHandle": ...   # resume paused, or create
      async def exec(self, user_id: str, command: str, timeout: int = 120) -> dict:  # {stdout, stderr, exit_code}
      async def read_file(self, user_id: str, path: str) -> str: ...
      async def write_file(self, user_id: str, path: str, content: str) -> None: ...
      async def list_dir(self, user_id: str, path: str = "/home/user") -> list[dict]:  # [{name, is_dir, size}]
      async def pause(self, user_id: str) -> None: ...                       # hibernate, persist state
  ```
- Lifecycle uses the e2b SDK; a per-user lock prevents two concurrent `get_or_create` from creating duplicate sandboxes (mirror Task 3's memory lock).
- New table (migration `db/migrations/0006_user_sandboxes.sql`):
  ```sql
  CREATE TABLE IF NOT EXISTS user_sandboxes (
      user_id        TEXT         PRIMARY KEY,
      e2b_sandbox_id TEXT         NOT NULL,
      status         TEXT         NOT NULL DEFAULT 'active',   -- active | paused
      created_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
      last_active_at TIMESTAMPTZ  NOT NULL DEFAULT now()
  );
  ```
- `get_or_create`: look up the row; if `paused`, resume by `e2b_sandbox_id`; if missing, create a sandbox, insert the row; update `last_active_at`. On `pause`, hibernate and set `status='paused'`.
- Config `computer_agent.sandbox`: `api_key` (`${E2B_API_KEY}`), `template` (default), `idle_timeout_seconds`. Add `e2b` to `requirements.txt`.
- Isolation: one sandbox per user; never pass another user's id.

**Touch points:** new `compute_module/sandbox.py`, `db/migrations/0006_user_sandboxes.sql`, `config_module/config.yaml`, `requirements.txt`, `db/migrate.py` (auto-picks up the migration).

**Depends on:** none (but exercise after the one-off spike note below).

**Out of scope:** idle-sweep daemon (manual/last_active pause for now), quotas, volume-mount persistence (pause/resume only).

**Acceptance test:** `test_get_or_create_returns_same_sandbox_per_user`, `test_file_persists_across_pause_resume`, `test_sandboxes_are_isolated_between_users`.

> **Spike note (do first, throwaway):** before wiring, prove in `scripts/spike_sandbox.py` that a file written, sandbox paused, then resumed, is still readable, and that wake latency is acceptable. If e2b persistence/latency disappoints, revisit the approach before building Tasks 5-10.

## Phase 2 -- The worker (computer-agent)

### Task 5: ComputerAgent -- the agent loop over the sandbox  ✅ DONE → see `COMPUTER_AGENT_SPEC.md`

*Built and verified against live e2b + Qwen3-8B. `computer_module/prompt.py` (layered system prompt), `tools.py` (agent-computer interface with read-before-edit/grep/edit discipline), `model.py` (auto-detect native tool-calling vs constrained-JSON fallback; Qwen3 thinking disabled), `agent.py` (the loop, kind-tagged events, step cap). Verified: wrote fibonacci script, ran it, verified output, returned correct summary in 7.9s. SGLang lacks `--tool-call-parser qwen3` so the constrained-JSON path runs; native activates when flag is added.*

*Decision update: the worker is a **persistent, separate agent we own** (under `computer_module`) with **Claude Code's scaffolding/prompts borrowed as paradigms** (re-authored, licensing-clean) and the **model as a knob** (`computer_agent.llm`, point at a frontier model for real capability). The detailed design -- the layered system prompt, the agent-computer tool interface (grep/glob/edit discipline + plan/verify), the loop -- lives in `COMPUTER_AGENT_SPEC.md` (its Tasks A-E). The bullets below are the summary; build from that spec.*

**Problem:** Something must drive the sandbox: take a task, use tools to do it, verify, report progress, return a result.

**Done when:**
- New `compute_module/agent.py` with:
  ```python
  class ComputerAgent:
      def __init__(self, user_id: str, sandbox: SandboxManager,
                   tool_manager, llm: ArkModelLink, emit, ask): ...
      async def run(self, prompt: str, *, step_cap: int = 25) -> dict:
          # returns {"status": "completed"|"failed"|"needs_input", "summary": str, "outputs": [paths]}
  ```
- The loop: build the tool list, call the model with tools, parse tool calls, execute each against the sandbox or MCP, feed results back, repeat until the model emits a final answer or `step_cap` is hit. Reuse `HARNESS_SPEC` discipline: validate model output, classify model errors (`ModelError`), never let an exception escape -- a failed step becomes an error result, not a crash.
- **Native toolset** (each tool is a function the model can call):
  - `run_command(command)` -> `sandbox.exec(user_id, command)`
  - `read_file(path)` -> `sandbox.read_file(user_id, path)`
  - `write_file(path, content)` -> `sandbox.write_file(user_id, path, content)`
  - the user's **MCP tools** -> `tool_manager.call_tool(name, args, user_id=user_id)` (user-scoped from Task 2; no raw token needed)
- **Progress:** before/after each tool call, invoke `emit(event)` where `event` carries `kind` in `{"shell","file","mcp"}`, the command/path/tool, and a one-line reasoning string. `emit` writes a `task_event` (Task 7) and feeds the SSE stream (Task 8).
- **Asks:** when the model decides it needs the human, it calls a special `ask_user(prompt)` tool -> `ask(prompt)` which creates an approval (Task 8) and blocks until resolved, then returns the answer into the loop.
- **Model:** read from `computer_agent.llm`; for the MVP this points at the local endpoint and is driven via `ArkModelLink`. Kept a separate config key so the frontier swap is one line.
- **Step cap** halts runaway loops with a clear terminal reason -- essential on Qwen.

**Touch points:** new `compute_module/agent.py`, `config_module/config.yaml` (`computer_agent.llm`), reuse `model_module/ArkModelNew.py`, `tool_module/smithery.py`.

**Depends on:** Task 2 (user-scoped MCP), Task 4 (sandbox).

**Out of scope:** recursive sub-spawning (THREAD -- later), token-by-token reasoning stream.

**Acceptance test:** `test_computer_agent_writes_and_runs_file`, `test_computer_agent_uses_correct_users_sandbox`, `test_step_cap_bounds_loop`, `test_computer_agent_mcp_is_user_scoped`.

## Phase 3 -- Dispatch, notification, surface

### Task 6: Buddy dispatch + computer_tasks table

**Problem:** Buddy must classify task weight and, for real work, dispatch to the ComputerAgent asynchronously, capturing the chat session so completion can post back.

**Done when:**
- Buddy routing gains a `needs_computer` route signal (extend `state_ai` guidance: pick it when the request needs files/shell/run or is multi-step computer work; keep trivial single MCP lookups as buddy-direct, conversational as direct answer).
- New table (`db/migrations/0007_computer_tasks.sql`):
  ```sql
  CREATE TABLE IF NOT EXISTS computer_tasks (
      task_id         UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
      user_id         TEXT         NOT NULL,
      chat_session_id TEXT         NOT NULL,        -- where to post the "done" message
      prompt          TEXT         NOT NULL,
      status          TEXT         NOT NULL DEFAULT 'pending',  -- pending|running|awaiting_input|completed|failed
      summary         TEXT,
      error           TEXT,
      created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
      updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
  );
  CREATE INDEX IF NOT EXISTS idx_computer_tasks_user ON computer_tasks (user_id, created_at DESC);
  ```
- A dispatch path (in the buddy tool/route handler) that: inserts a `computer_tasks` row with the verified `user_id` and the **current chat session id** (from the agent's `Memory.session_id`), then spawns an async runner (`compute_module/runner.py`, mirroring `task_runner.spawn`) and returns immediately. Buddy tells the user: "Started -- I'll message you when it's done."
- New `compute_module/store.py` with `create_computer_task`, `set_computer_status`, `get_computer_task`, `list_computer_tasks(user_id)` (all user-scoped, mirroring `task_store.py`).

**Touch points:** `state_module/agent_buddy/` (route + dispatch), new `compute_module/runner.py` + `compute_module/store.py`, `db/migrations/0007_computer_tasks.sql`.

**Depends on:** Task 5.

**Acceptance test:** `test_buddy_routes_real_work_to_computer`, `test_trivial_lookup_does_not_wake_computer`, `test_dispatch_records_chat_session`.

### Task 7: Completion notification -- "send a message back when it's done"

**Problem:** When the async task finishes, the user must get a message in the chat saying so, with the result -- whether or not they're watching.

**Done when:**
- The runner (`compute_module/runner.py`) drives `ComputerAgent.run(prompt)`; on terminal state it performs **all** of:
  1. `set_computer_status(task_id, "completed"|"failed", summary=..., error=...)`.
  2. Emit a final `task_event` row (`kind='completed'` or `'failed'`) with the summary + output paths (events table per Task 8).
  3. **Inject a chat message:** construct a `Memory(user_id, session_id=chat_session_id, db_url, use_long_term=False)` and call `add_memory(AIMessage(content=<result message>))`. This writes an assistant turn into `conversation_context` for the user's chat session, so the result is in the conversation the user reads. The message is human-friendly: a one-line outcome + key outputs + any file paths.
  4. Push the completion onto the SSE stream (Task 8) for live delivery.
  5. `sandbox.pause(user_id)` to stop idle compute.
- On `failed`, the injected message is a clear, non-alarming failure ("I couldn't finish that -- <reason>. Want me to try again?"), never a stack trace.
- The same path is used for `awaiting_input` -> the agent's ask surfaces (Task 8) and the runner resumes on resolution.

**Touch points:** `compute_module/runner.py`, `memory_module/memory.py` (reuse `add_memory`), the events table (Task 8).

**Depends on:** Task 5, Task 6, Task 8 (events/SSE).

**Acceptance test:** `test_completion_injects_chat_message`, `test_failure_injects_friendly_message`, `test_completed_status_and_event_written`.

### Task 8: Surface -- progress events, SSE stream, approval asks

**Problem:** Progress, asks, and completion need to reach the UI live and durably, with the agent's own words.

**Done when:**
- A `computer_task_events` table (or reuse `task_events` with a `computer_task_id`) storing every emitted event: `(event_id, task_id, kind, content, payload jsonb, created_at)`. `kind` includes `shell|file|mcp|reasoning|ask|completed|failed`.
- `emit(event)` (passed into `ComputerAgent`) writes a row to this table.
- **SSE endpoint** `GET /computer/tasks/{task_id}/stream` (FastAPI `StreamingResponse`, `media_type="text/event-stream"`, `CurrentUser`-scoped to the task's owner): tails new event rows (poll the table every ~1s for `event_id > last_seen`, yield as `data: {json}\n\n`); closes on a `completed`/`failed` event. Polling `GET /computer/tasks/{task_id}/events?after=<id>` is the fallback.
- **Asks:** the `ask(prompt)` callback creates an approval via `create_approval(task_id, user_id, kind, prompt)`; it surfaces in the existing tray (`/tasks/approvals/pending`-style, extended to include computer tasks); the runner polls `get_approval` (like `state_approval`) and resumes when resolved. The ask text is the agent's own -- never re-generated by buddy/Qwen.
- Hard rule: subagent messages (progress, asks, completion) are passed through verbatim; the Qwen front never paraphrases the Claude/Qwen worker's output.

**Touch points:** new events table migration (fold into `0007` or a `0008`), `compute_module/runner.py`/`agent.py` (`emit`/`ask`), new endpoints in `base_module/app.py` (or `base_module/computer_router.py`), reuse `task_store` approval helpers.

**Depends on:** Task 6.

**Acceptance test:** `test_events_carry_kind_tag`, `test_sse_stream_is_owner_scoped`, `test_agent_ask_routes_through_tray_and_back`.

## Phase 4 -- UI

### Task 9: "Computer" tab -- filesystem viewer

**Problem:** The user should see their persistent computer, not just chat about it.

**Done when:**
- New **Computer** tab in the frontend (`arkos-webui` / `frontend`): a browsable file tree + read-only content view (download optional).
- Backed by `CurrentUser`-scoped endpoints, only ever touching the caller's sandbox:
  - `GET /computer/files?path=` -> `SandboxManager.list_dir(user_id, path)` -> `[{name, is_dir, size}]`
  - `GET /computer/file?path=` -> `SandboxManager.read_file(user_id, path)` -> content (size-capped; large files truncated with a notice)
- Viewing a paused sandbox wakes it (or reads persisted state) transparently.

**Touch points:** new endpoints in `base_module/app.py`/`computer_router.py`, `compute_module/sandbox.py` (`list_dir`), frontend Computer tab.

**Depends on:** Task 1 (auth), Task 4 (sandbox).

**Out of scope:** in-browser editing, upload, diffs (v2).

**Acceptance test:** `test_computer_files_endpoint_is_user_scoped`.

### Task 10: Activity stream + "using the computer" indicator

**Problem:** The subagent/activity section must show the computer-agent's streamed commands/thoughts, and a glanceable icon distinguishing computer-use from generic MCP.

**Done when:**
- The activity view subscribes to the SSE stream (Task 8) and renders progress events live (commands run, files touched, the per-step reasoning line).
- Each subagent in the UI shows a **kind indicator** -- a distinct icon for "using the computer" vs "MCP call" vs "chat" -- driven by the event `kind` tag from Task 5/8. The indicator updates live (e.g. computer icon + current command).

**Touch points:** frontend (activity/subagent components), the SSE endpoint (Task 8).

**Depends on:** Task 8.

**Acceptance test:** covered by `test_events_carry_kind_tag` (backend) + manual UI check.

## Phase 5 -- Config & cleanup

### Task 11: Config rename + identity gate doc

**Done when:**
- `config.yaml` `openhands:` renamed to `computer_agent:` with `sandbox` (e2b) and `llm` subsections; `computer_agent.llm` points at the local endpoint for the MVP (separate key, swap-later). OpenHands references removed.
- A boot-time assertion / README note that the computer endpoints (`/computer/*`, dispatch) must run under verified identity -- already enforced by Task 1's `CurrentUser`, but documented as the non-negotiable gate.

**Touch points:** `config_module/config.yaml`, `PLATFORM_SPEC.md` cross-reference.

**Depends on:** Task 1, Task 4, Task 5.

---

# Consolidated Data Model (new)

```sql
-- 0006_user_sandboxes.sql
user_sandboxes(user_id PK, e2b_sandbox_id, status, created_at, last_active_at)

-- 0007_computer_tasks.sql
computer_tasks(task_id PK, user_id, chat_session_id, prompt, status, summary, error, created_at, updated_at)
computer_task_events(event_id PK, task_id FK, kind, content, payload jsonb, created_at)
```

# Consolidated API surface (new, all `CurrentUser`-scoped)

```
POST /computer/tasks                 dispatch a computer task        -> {task_id, status}
GET  /computer/tasks                 list caller's computer tasks    -> [task]
GET  /computer/tasks/{id}            one task (owner only)           -> task
GET  /computer/tasks/{id}/events?after=<id>   poll progress events   -> [event]
GET  /computer/tasks/{id}/stream     SSE: live progress + completion (owner only)
GET  /computer/files?path=           list dir in caller's sandbox    -> [{name,is_dir,size}]
GET  /computer/file?path=            read file in caller's sandbox   -> content
```
(Approvals reuse the existing `/tasks/approvals/*` surface, extended to include computer tasks.)

---

# Tests (consolidated)

**Identity/registry (Phase 0):** `test_missing_token_rejected_outside_demo`, `test_header_cannot_override_verified_identity`, `test_tool_registry_is_user_scoped`, `test_colliding_tool_names_do_not_cross_users`, `test_concurrent_requests_get_isolated_memory`, `test_startup_fails_on_default_secret_in_prod`.

**Sandbox (Phase 1):** `test_get_or_create_returns_same_sandbox_per_user`, `test_file_persists_across_pause_resume`, `test_sandboxes_are_isolated_between_users`.

**Agent (Phase 2):** `test_computer_agent_writes_and_runs_file`, `test_computer_agent_uses_correct_users_sandbox`, `test_step_cap_bounds_loop`, `test_computer_agent_mcp_is_user_scoped`.

**Dispatch/notification/surface (Phase 3):** `test_buddy_routes_real_work_to_computer`, `test_trivial_lookup_does_not_wake_computer`, `test_dispatch_records_chat_session`, `test_completion_injects_chat_message`, `test_failure_injects_friendly_message`, `test_completed_status_and_event_written`, `test_events_carry_kind_tag`, `test_sse_stream_is_owner_scoped`, `test_agent_ask_routes_through_tray_and_back`.

**UI (Phase 4):** `test_computer_files_endpoint_is_user_scoped` (+ manual: tab renders files, indicator shows computer icon).

Each test names the property it pins; the highest-stakes are the cross-user isolation tests (`*_isolated_between_users`, `*_is_user_scoped`, `*_owner_scoped`) -- a failure there leaks one user's computer/files/tools to another, the worst failure this design has.

---

# Open Questions

1. ~~**e2b API specifics.**~~ **RESOLVED via the Task 0 spike (e2b 2.25.1, `computer_module/spike_sandbox.py`).** Confirmed API: `Sandbox.create(timeout=...)` -> sandbox with `.sandbox_id`; `sbx.commands.run(cmd)` -> `.stdout/.stderr/.exit_code`; `sbx.files.write(path, content)` / `sbx.files.read(path)`; **`sbx.pause()` returns `None`** (NOT the id -- the docstring is wrong); reconnect/resume via **`Sandbox.connect(sandbox_id)`** using the id captured at create; `sbx.kill()` to destroy. So `user_sandboxes` must store the `sandbox_id` from create and resume by it. **Measured latency: create 0.21s, resume 0.36s, persistence PASS** -- sub-second wake means pause-on-idle is effectively free; the "keep warm during session" optimization is unnecessary. `SandboxManager` remains the only place that touches the e2b SDK.
2. **Completion delivery when offline.** Chat injection + SSE covers connected clients; a user who has closed the app sees the message next open. True mobile/web **push notifications** are an optional follow-on (web-push subscription + a `/computer/tasks/{id}` completion hook) -- spec'd as out-of-scope for the first pass, but the chat-injection design means nothing is lost, only delayed.
3. **Routing threshold.** Where "buddy-direct trivial" ends and "wake the computer" begins. Start coarse (anything file/shell/multi-step -> computer); tune from real use.
4. **Token / cost caps.** Local Qwen has no API cost, so the MVP cap is the step cap only. When the model swaps to a frontier endpoint, add a per-task token budget and decide whether to always-gate dispatch behind approval.
5. **Idle/pause policy + quotas.** When does a warm sandbox pause (idle timeout via `last_active_at`)? Per-user storage quota? Needs a sweep eventually; manual/last_active pause for the first pass.
6. **`computer_tasks` vs `tasks`.** A separate table (chosen here) keeps the computer lifecycle clean; if it converges with `tasks` later, migrate then.

---

# Implementation Notes

- **Spike first.** Do the throwaway `scripts/spike_sandbox.py` (Task 4 note) before building -- prove e2b persistence + wake latency in isolation. Everything else assumes it works.
- **The gate is non-negotiable.** Phase 0 (Tasks 1-3) must land before any `/computer/*` endpoint is reachable. A dispatch under header-trust identity lands user A's task in user B's sandbox with B's tools -- the worst failure this design has. `CurrentUser` on every computer endpoint, `ARK_DEMO_MODE` off in any shared deployment.
- **Model is a knob.** `computer_agent.llm` points at the local Qwen now; the swap to a frontier model is one config line because the agent only talks to a generic tool interface. Keep it that way -- never couple the agent to model-specific behavior.
- **Reuse, don't rebuild.** Phase 3 is mostly wiring over `task_runner`/`task_events`/`create_approval`/`StreamingResponse`/`Memory.add_memory`. Resist writing new async/progress/approval/streaming machinery -- it all exists.
- **Completion = durable + live.** The "message back when done" is the chat injection (durable, the user always sees it) *plus* the SSE push (live if watching). Implement both; the injection is the one that must never be skipped.
- **One e2b touch point.** All sandbox SDK calls live in `compute_module/sandbox.py`. Nothing else imports `e2b`. This is what keeps the persistence flavor (pause/resume vs volume) swappable later.
