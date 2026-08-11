# Feature Spec: ARKOS Agent Harness — Resilience Layer

**Sources**

- `arkos-inspo/claude-code` — agent loop (`src/query.ts`), tool orchestration (`src/toolOrchestration.ts`, `StreamingToolExecutor.ts`), model retry/fallback (`withRetry.ts`), permission gate (`hooks/useCanUseTool.tsx`), typed `Terminal` outcomes
- Current code: `agent_module/agent.py` (`step`, `step_stream`, `call_llm`, `choose_transition`), `model_module/ArkModelNew.py` (`ArkModelLink`), `state_module/core/` (`StateOutput`, `StateHandler`), `state_module/agent_buddy/state_tool.py`, `state_module/agent_executor/state_approval.py`
- Companion: `arkos-inspo/specs/MEMORY_SPEC.md` (token budgeting overlaps Task 4 here)

**Status:** Not started | **Author:** | **Last updated:** 2026-05-31

---

# Problem

ARKOS and Claude Code share the same control-flow philosophy — *deterministic harness, the model only fills decisions* — and ARKOS enforces it strictly (the no-LLM `check_transition_ready` contract, constrained-decoding for every choice). That foundation is sound.

The gap is the **resilience layer** — everything that lets a harness survive a flaky model, a malformed output, or an overflowing context. Today ARKOS has almost none of it, and several failures are silent:

- **LLM errors masquerade as results (most dangerous).** `make_llm_call` catches every exception and *returns the string* `f"Error: ..."` (`ArkModelNew.py:137`); `call_llm` wraps it in `AIMessage(content=...)` (`agent.py:147`). A timeout or HTTP 500 becomes "the assistant said: Error: …", is written to memory, and flows downstream as a normal reply. No retry, no backoff, no fallback.
- **Unvalidated model output.** Structured outputs are parsed with raw `json.loads` across the hot path — `agent.py:175` (transition choice), `state_tool.py:49,67` (buddy tool select + args), `agent_executor/state_tool.py` and `state_executor.py` (subagent tool select), `state_ai.py:120` (reply schema), and `task_runner.py:91` (stored context payload). A malformed payload throws and kills the state. CLAUDE.md *itself* mandates `model_validate_json` + `ValidationError` handling — the hot path ignores its own contract.
- **Two error paths that disagree.** `step()` has no `try/except` around `state.run` (`agent.py:250`) — an exception propagates and aborts the turn. `step_stream()` wraps it and reroutes to `agent_reply` (`agent.py:318-328`). Same framework, two behaviors.
- **No context-overflow recovery.** When context exceeds the window the turn drops history or the API 413s; there is no compaction/truncation fallback (Claude Code does collapse → reactive-compact → truncate).
- **Fake streaming.** `step_stream` buffers the whole state output then replays it `for char in update.content: yield char` (`agent.py:336`). Real token streaming (`generate_stream`, `ArkModelNew.py:171`) exists but is never wired in.
- **Cost/latency: two LLM calls per tool use.** `_choose_tool` calls the model once to pick the tool, then again to fill args (`state_tool.py:48,66`) — 2× latency and 2× prefix-cache breakage per tool. *(Not fixed here — `ARCHITECTURE_SPEC.md` treats the 2-call pattern as a prior replaced by native tool-calling, not optimized; collapsing it to 1-call would be throwaway work and a dubious 7B win.)*
- **No tool-result budgeting.** `str(tool_result)` is stuffed into context unbounded (`state_tool.py:84`), a latent 413.
- **No network resilience on the tool path.** aiohttp calls to Smithery have **no timeout** (`smithery.py:135,165`) — a hung Smithery hangs the agent indefinitely. A network blip during tool discovery crashes the whole tool call because only `AuthRequiredError` is caught (`smithery.py:452`), and a 401 is raised as a generic `SmitheryError` (`:138,168`) instead of `AuthRequiredError`, breaking the connect-prompt flow.
- **Blocking I/O and leaked resources in async paths.** The async health endpoint uses blocking `requests.get()` (`app.py:436`) and a `psycopg2.connect()` not wrapped in `try/finally` (`app.py:457`); `task_store.py`/`tasks.py` open connections with no `connect_timeout` and don't explicitly close cursors → event-loop stalls and leaks under DB pressure.
- **Background-task lifecycle is fragile.** A task row is created `running` *before* `spawn()` succeeds (`tasks.py:207`); a failed spawn leaves an orphaned "running" row that re-resumes and spams logs on restart.
- **Config fails late, not fast.** `${VAR}` with a missing env raises `OSError` at load (no `:-default`); missing keys return `None` and surface as cryptic downstream errors instead of a clear startup failure (`config_module/loader.py`).
- **`print()` instead of structured logging**, pervasive (`agent.py`, `memory.py`, `smithery.py`, `state_ai.py:120`, `ArkModelNew.py:110`) despite the CLAUDE.md contract — a dead backend is invisible.

> **Scope note — multi-user correctness/security is a separate spec.** The audit also surfaced cross-user data exposure (a global tool registry that routes users to each other's servers) and header-trust identity (any client can impersonate any user). Those are **not** harness resilience and live in `arkos-inspo/specs/MULTIUSER_SPEC.md`. This spec covers only single-process robustness: errors, validation, recovery, network/resource hygiene.

**Success looks like:** a model error retries or surfaces (never becomes content); a malformed output becomes a clean `error` outcome (never a crash); one unified error path; context overflow degrades gracefully; a hung dependency times out instead of hanging the agent; and the loop owns named, inspectable exit reasons.

---

# Technical Background

**What ARKOS already gets right (preserve, do not regress):**
- Deterministic control flow — YAML state graph, signal-based routing (`routers.py`), `check_transition_ready` reads `completion_signal` only, never the LLM (CLAUDE.md contract #2). This is Claude Code's principle "the harness owns control, the model fills decisions."
- Constrained decoding for every decision — `create_next_state_class` / `create_tool_option_class` (`agent.py:107,82`) force the model to pick from an enum; it cannot hallucinate a state or tool name.
- Typed outcomes — `StateOutput.completion_signal ∈ {complete, incomplete, error, needs_input}` (`base_state.py`), analogous to Claude Code's typed `Terminal` reasons.
- A real human-in-the-loop gate for the subagent (`state_approval.py`).

**Claude Code's resilience model (the parts worth importing):**
- *Recoverable vs terminal errors are classified.* Prompt-too-long / max-tokens / image-size are withheld from the user and recovered (collapse → compact → escalate) before surfacing; auth/400/model errors return immediately (`query.ts:788-825,1062-1184`).
- *Retry with backoff in the model client.* 429/5xx/timeout retryable; 400/401/403 not (`withRetry.ts`).
- *Validate before trust.* Every tool input `safeParse`d; failures returned to the model as structured errors, not crashes (`toolOrchestration.ts:97`).
- *Named loop transitions.* Seven explicit continue/exit reasons stored on loop state — no magical retries (`query.ts:214-217`).
- *Tool results are budgeted*, truncated in place before the next API call (`query.ts:369-394`).

**Key constraint:** these are *additive* to ARKOS's loop, not a rewrite. The state graph, routing, and discovery stay exactly as they are; the changes live in the model client, the `json.loads` call sites, and the loop's error/exit handling.

**Mental model — failures enter at exactly three boundaries.** Every brittle moment in the harness is a failure entering at one of three points, and each task below installs a catch-and-classify at one of them:

```
   ┌─────────────────────────────────────────────────────┐
   │  agent.step() loop                                   │
   │                                                      │
   │   get_context ──► state.run ──► add_context          │
   │        ▲              │                              │
   │        │              ▼                              │
   │   (3) context    call_llm ──► (1) the model CALL     │
   │   overflow /          │           (network / API)    │
   │   tool-result         ▼                              │
   │   size         (2) the model OUTPUT                  │
   │                    (malformed JSON / wrong choice)   │
   └─────────────────────────────────────────────────────┘
```

- **(1) The call** — network/API failure. Today returns an error *string* (`ArkModelNew.py:137`). → Task 1.
- **(2) The output** — malformed/unparseable content. Today crashes via raw `json.loads`. → Task 2.
- **(3) Resources** — context overflow / unbounded tool results. Today unhandled / latent 413. → Task 4 (results) + `MEMORY_SPEC.md` Task 2 (context).

Claude Code puts a catch + recover at all three; ARKOS currently catches none cleanly. The goal: after these tasks, a failure can only leave the loop as a *typed outcome* (Task 3), never as a crash or a fake reply.

---

# Proposed Approach

Add a resilience layer around the existing loop in four moves, hardest-failure-first:

1. **Make the model client honest.** `ArkModelLink` raises a typed `ModelError` instead of returning an error string, and retries transient failures with backoff. The loop — not a string — decides what happens next.
2. **Validate every structured output.** Replace raw `json.loads` with Pydantic `model_validate_json`; a parse/validation failure becomes a clean `StateOutput(completion_signal="error")`, never a crash.
3. **Unify the error path and give the loop named outcomes.** One `try/except` discipline shared by `step` and `step_stream`; the loop returns a typed terminal reason (`completed | max_steps | model_error | needs_input`). Transient model errors retry the step; validation errors surface.
4. **Budget tool results + collapse the two-call tool pattern.** Truncate oversized tool output before it re-enters context; choose tool and fill args in a single constrained call.

What stays the same: the YAML graph, `StateHandler` discovery, routers, `check_transition_ready` contract, constrained-decoding for choices, and the subagent approval flow.

Explicitly **not in scope** (deferred to keep this high-impact):
- Real token streaming + user interruption/abort — Open Question 1 (own spec; `generate_stream` exists but unwired).
- Fallback-model switching (Claude Code `FallbackTriggeredError`) — Open Question 2; single local vLLM today.
- A generalized permission gate for the chat agent (today only the subagent has `state_approval`) — Open Question 3.
- Prompt-cache break detection/telemetry — folded informally into Task 4 (single-call tooling helps cache), not built out.

---

# Implementation Plan

## Task 1: Typed ModelError + retry/backoff in the model client

**Problem:** LLM failures return a string that the loop treats as a real assistant reply; no retry on transient errors.

**Done when:**
- `make_llm_call` no longer returns `f"Error: ..."`; on failure it raises a typed `ModelError` (new, in `model_module`) carrying the underlying cause and a `retryable: bool`.
- `ArkModelLink` retries `retryable` failures (timeout / 429 / 5xx) with exponential backoff (cap ~3 attempts); non-retryable (400/401/403) raise immediately.
- The bare `except Exception` returning a string is removed; `print` replaced with `emit_log`.

**Sketch (illustrative, not final):**

```python
# model_module/errors.py
class ModelError(Exception):
    def __init__(self, message: str, *, retryable: bool, cause: Exception | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.cause = cause

# ArkModelNew.py — replacing the except block in make_llm_call
except (APITimeoutError, RateLimitError, InternalServerError) as e:
    raise ModelError(str(e), retryable=True, cause=e) from e
except (BadRequestError, AuthenticationError) as e:
    raise ModelError(str(e), retryable=False, cause=e) from e

# generate_response — retry wrapper around the call
async def generate_response(self, messages, json_schema) -> str:
    delay = 0.5
    for attempt in range(3):
        try:
            return await self.make_llm_call(messages, json_schema=json_schema)
        except ModelError as e:
            if not e.retryable or attempt == 2:
                raise
            emit_log(...)                 # structured, not print
            await asyncio.sleep(delay)
            delay *= 2
```

**Touch point:** `model_module/ArkModelNew.py`, new `model_module/errors.py`, `logging_module`.

**Priority:** P0 | **Effort:** ~1 day | **Blockers:** none

**Out of scope:** Fallback-model switching (Open Question 2); streaming changes.

**Acceptance test:** `test_model_client_raises_on_failure`, `test_model_client_retries_transient` (below).

---

## Task 2: Validate every structured output

**Problem:** Raw `json.loads` on model output crashes the state on malformed JSON; violates the CLAUDE.md `model_validate_json` contract.

**Done when:**
- Every `json.loads` on model/stored output is replaced with a Pydantic `model_validate_json` against the requested schema — full inventory: `agent.py:175`, `agent_buddy/state_tool.py:49,67`, `agent_executor/state_tool.py` + `state_executor.py` (subagent tool select), `state_ai.py:120`, and `task_runner.py:91` (DB-stored context payload).
- A `ValidationError`/parse failure is caught and converted to `StateOutput(completion_signal="error", error_detail=...)` (or a bounded re-ask), never propagated raw.
- A hallucinated tool name is validated against the *current user's* available tools before `_tool_registry[...]` is indexed (`state_tool.py:52`), so a bad name yields an `error` outcome, not a `KeyError`.

**Sketch (illustrative, not final):** parse against the same Pydantic model the schema was built from (`create_next_state_class` / `create_tool_option_class` already produce it), folding failure into the return value rather than an exception:

```python
def parse_structured(content: str, model: type[BaseModel]) -> BaseModel | None:
    try:
        return model.model_validate_json(content)
    except ValidationError:
        emit_log(...)
        return None

# choose_transition, after the call
parsed = parse_structured(output.content, NextStates)
if parsed is None:
    return None          # "couldn't decide" — the loop (Task 3) handles it
return parsed.next_state.value
```

**Touch point:** `agent_module/agent.py` (`choose_transition`), `state_module/agent_buddy/state_tool.py`, any other `json.loads` on LLM output.

**Priority:** P0 | **Effort:** ~1 day | **Blockers:** none

**Out of scope:** Re-ask/repair loops beyond a single bounded retry.

**Acceptance test:** `test_malformed_structured_output_yields_error_outcome` (below).

---

## Task 3: One error path + named loop outcomes

**Problem:** `step` and `step_stream` handle `state.run` exceptions differently; `max_iter` is a blunt counter that just `break`s with no reason.

**Done when:**
- Both `step` and `step_stream` wrap `state.run` with one shared error-handling helper that maps exceptions → `completion_signal` (transient `ModelError` → retry the step up to a cap; validation/other → `error` outcome routed to `agent_reply`).
- The loop returns a typed terminal reason — `completed | max_steps | model_error | needs_input` — surfaced on/alongside `StateOutput` so callers (`app.py`, `task_runner.py`) can react instead of guessing.
- `max_iter` exit emits `max_steps` via `emit_log`, not a bare `print`/`break`.

**Sketch (illustrative, not final):** one helper both entrypoints call; the three boundaries all drain into one classified exit:

```python
# base_state.py
class TerminalReason(str, Enum):
    completed = "completed"
    max_steps = "max_steps"
    model_error = "model_error"
    needs_input = "needs_input"

# agent.py — the single place state.run is invoked
async def _run_state(self, context):
    try:
        return await self.current_state.run(context, self), None
    except ModelError as e:
        if e.retryable:
            return None, "retry"                      # loop re-runs the step, bounded
        return StateOutput(content="…", completion_signal="error",
                           error_detail=str(e)), TerminalReason.model_error
    except Exception as e:                            # validation, bugs
        return StateOutput(content="…", completion_signal="error",
                           error_detail=str(e)), None  # reroute to agent_reply

# max_iter exit now means something
if retry_count > self.max_iter:
    self.terminal_reason = TerminalReason.max_steps
    emit_log(...)
    break
```

**Touch point:** `agent_module/agent.py` (`step`, `step_stream`), `state_module/core/base_state.py` (terminal-reason enum), call sites in `base_module/`.

**Priority:** P0 | **Effort:** ~2 days | **Blockers:** Task 1, Task 2

**Out of scope:** Interruption/abort (Open Question 1).

**Acceptance test:** `test_step_and_stream_share_error_behavior`, `test_loop_returns_named_terminal_reason` (below).

---

## Task 4: Bound tool-result rendering + drop the dead stash

**Problem:** Buddy renders the full tool result into `content` unbounded (`state_tool.py:84`) — the actual 413 — and also stashes the full result in `structured_data["tool_result"]` (`:87`), which nothing reads. The executor caps with a blunt `[:400]` (`agent_executor/state_tool.py:87`), dropping data arbitrarily.

**Done when:**
- The dead `structured_data={"tool_result": ...}` stash in buddy `state_tool.py:87` is removed — grep-verified that nothing reads the `"tool_result"` key (only `structured_data["route"]` and the executor's *final* summary are consumed). No result is stored anywhere until a reader exists (retrieval is downstream, `ENVIRONMENT_SPEC.md`).
- The tool result's **`content` rendering is bounded** via the context-aware budgeter (Task 7) — replacing buddy's unbounded `str(tool_result)` and the executor's blunt `[:400]` with a structure-aware head+tail view sized to remaining context.

**Sketch (illustrative, not final):**

```python
# bound the rendering with the context-aware budgeter (Task 7), drop the dead stash
view = render_for_context(tool_result, budget=agent.tool_result_budget())  # head+tail, structure-aware
return StateOutput(content=view, completion_signal="complete",
                   structured_data={"route": "continue"})   # no tool_result key — nothing reads it
```

**Touch point:** `state_module/agent_buddy/state_tool.py`, `state_module/agent_executor/state_tool.py`.

**Priority:** P1 | **Effort:** ~1 day | **Blockers:** Task 7 (budgeter)

**Out of scope:**
- **Single-call tool selection** (nesting args under the tool-choice enum) — *removed from this task.* It's a tool-*selection* mechanic, not result handling; `ARCHITECTURE_SPEC.md`'s removal-trigger table already treats the current 2-call `select→fill` (`state_tool.py:48,66`) as a **prior replaced by native tool-calling**, not optimized in place — so 1-call is throwaway before that refactor. It's also a dubious 7B win: keeping constrained arg-decoding forces a `oneOf` over every tool's schema (one big union decode) vs. today's two small focused decodes.
- On-demand retrieval / `get_tool_result` / writable workspace (downstream, `ENVIRONMENT_SPEC.md`); parallel tool execution; streaming tool results.

**Acceptance test:** `test_buddy_tool_content_is_bounded`, `test_no_dead_structured_data` (below).

---

## Task 5: Network resilience on the tool path

**Problem:** Smithery calls can hang forever (no timeout); a network blip during tool discovery crashes the whole tool call; 401s are misclassified as generic errors, breaking the connect-prompt flow.

**Done when:**
- Every aiohttp call to Smithery (`smithery.py:135,165` and the JSON-RPC path) sets an explicit `ClientTimeout` (e.g. `total=30, connect=10`); a timeout raises a typed, classified error, not a hang.
- The tool-discovery loop catches `SmitheryError` as well as `AuthRequiredError` (`smithery.py:452`) and skips/logs the failing server instead of aborting the whole call.
- HTTP 401 from Smithery (`:138,168`) is raised as `AuthRequiredError` (needs-setup), distinct from transient `SmitheryError` (retry/skip) — same recoverable-vs-terminal split as Task 1.
- The bare `except Exception` that returns an empty dict on parse failure (`smithery.py:176`) raises a `SmitheryError` instead of silently yielding "success with no result".

**Sketch (illustrative, not final):**

```python
TIMEOUT = aiohttp.ClientTimeout(total=30, connect=10)
async with session.post(url, json=payload, timeout=TIMEOUT) as resp:
    if resp.status == 401:
        raise AuthRequiredError(service=server_name, ...)   # needs human setup
    if resp.status >= 400:
        raise SmitheryError(f"{method} {resp.status}", retryable=resp.status >= 500)

# discovery loop
try:
    await self._ensure_user_server(session, user_id, candidate)
except AuthRequiredError:
    continue                      # surface connect prompt
except SmitheryError as e:
    emit_log(...); continue       # skip this server, don't crash the call
```

**Touch point:** `tool_module/smithery.py`.

**Priority:** P1 | **Effort:** ~1 day | **Blockers:** none (composes with `MULTIUSER_SPEC.md` Task 1, same file)

**Out of scope:** Retry/backoff *policy* for Smithery (mirror Task 1's model-client backoff if needed); tool-list cache TTL.

**Acceptance test:** `test_smithery_timeout_raises_not_hangs`, `test_network_blip_skips_server_not_crash`, `test_401_classified_as_auth_required` (below).

---

## Task 6: Resource hygiene, fail-fast config, structured logging

**Problem:** Blocking I/O and leaked resources in async paths; config fails late and cryptically; `print()` hides failures.

**Done when:**
- The async health endpoint replaces blocking `requests.get()` with `httpx`/`aiohttp` and wraps its DB connection in `try/finally` (`app.py:436,457`); `task_store.py`/`tasks.py` pass `connect_timeout` and close cursors via context managers.
- A task row is inserted `pending` and promoted to `running` only after `spawn()` succeeds (`tasks.py:207`), so a failed spawn can't orphan a "running" row.
- Config validates required keys (`database.url`, `llm.base_url`, JWT secret, …) at startup and supports `${VAR:-default}`; a missing required key/env aborts boot with a clear message instead of surfacing as a downstream `None`.
- The `print()` calls on the hot path (`agent.py`, `memory.py`, `smithery.py`, `state_ai.py:120`, `ArkModelNew.py:110`) are routed through `emit_log`.

**Touch point:** `base_module/app.py`, `base_module/task_store.py`, `base_module/tasks.py`, `config_module/loader.py`, `logging_module`, and the `print()` sites above.

**Priority:** P2 | **Effort:** ~1–2 days | **Blockers:** none

**Out of scope:** Full migration to an async DB driver (just timeouts + correct close for now); CORS/rate-limit hardening; DB foreign-key/schema-drift cleanup (track separately).

**Acceptance test:** `test_required_config_missing_fails_fast`, `test_failed_spawn_does_not_orphan_running_task` (below).

---

## Task 7: Context-aware token budgeting

**Problem:** Oversized payloads are rendered into context with either no bound (buddy `str(tool_result)`) or a blunt fixed cap (executor `[:400]`). A fixed char cap is both too small to use available headroom *and* blind to how full the context already is — the budget must be computed against the live context, not hardcoded.

**Done when:**
- The `Agent` tracks current context size in tokens — `self.context_tokens`, recomputed in `get_context()` whenever the message list is assembled.
- A shared budget helper computes remaining room and sizes any oversized rendering to fit:
  `room = context_window − context_tokens − output_reserve − safety_margin`, with `safety_margin` ≈ 1–2k so we never render up to the hard edge.
- Config gains the real limit: **`llm.context_window`** (total tokens the model accepts), distinct from `llm.max_tokens` (the *output* reserve). Until `context_window` is set, conservatively treat `llm.max_tokens` as the whole budget (safe, wastes some room).
- A single token counter is used everywhere: **`tiktoken`** behind one function (`count_tokens`). Note it's an *approximation* — `tiktoken` is OpenAI's tokenizer, the served model (Qwen via TGI/vLLM) tokenizes differently, so counts can be off ~10–30%. Absorb that error in the safety margin: apply a fudge factor (`tokens * ~1.15`) and/or keep `safety_margin` at the high end (~2k). Exactness isn't the goal — staying off the 413 edge is.
- `agent.tool_result_budget()` (used by Task 4) returns `room` so a tool result renders to exactly the space left, not a magic number.

**Sketch (illustrative, not final):**

```python
# config.yaml
# llm:
#   context_window: 32768   # NEW — the real limit (total in+out)
#   max_tokens:     8192    # output reserve (already present)

# agent.py
async def get_context(self, ...):
    ctx = await self._assemble()                 # long-term + short-term
    self.context_tokens = count_tokens(ctx)      # one counter, used everywhere
    return ctx

def tool_result_budget(self) -> int:
    limit  = cfg("llm.context_window") or cfg("llm.max_tokens")
    reserve = cfg("llm.max_tokens") if cfg("llm.context_window") else 0
    return max(0, limit - self.context_tokens - reserve - SAFETY_MARGIN)  # ~1–2k

```
NOTE: ALL ENV vars like SAFETY MARGIN should be configurable in config.yaml

**Touch point:** `agent_module/agent.py` (`context_tokens`, `get_context`, `tool_result_budget`), `config_module/config.yaml`, shared token counter + budget helper.

**Priority:** P1 | **Effort:** ~2 days | **Blockers:** none. **Shared infra:** this counter + helper is the same one `MEMORY_SPEC.md` Task 2 (token-budgeted working memory) needs — build once here, reuse there.

**Out of scope:** *Acting* on an exhausted budget (compaction/summarization is the memory/compaction work) — this task only **measures** context and **bounds** a rendering; when `room` is near zero that's a signal for compaction, handled elsewhere.

**Acceptance test:** `test_context_tokens_tracked`, `test_budget_keeps_headroom_below_limit` (below).

---

# Tests

## Test 1: test_model_client_raises_on_failure

**What it verifies:** A simulated transport/HTTP error from the LLM endpoint makes `ArkModelLink` raise `ModelError`, not return a string beginning with "Error:".

**Why this matters:** This is the most dangerous current behavior — an error silently becoming assistant content that gets stored and reasoned over. The test pins that errors can never masquerade as results.

---

## Test 2: test_model_client_retries_transient

**What it verifies:** A retryable failure (timeout/429/5xx) is retried with backoff up to the cap and succeeds if a later attempt returns; a non-retryable (400/401) raises immediately with no retry.

**Why this matters:** Distinguishes recoverable from terminal failures — the load-bearing classification that lets the loop degrade gracefully instead of failing on the first transient blip.

---

## Test 3: test_malformed_structured_output_yields_error_outcome

**What it verifies:** When the model returns invalid JSON for a constrained schema, the state returns `StateOutput(completion_signal="error")` with `error_detail` set, and no exception propagates out of `run()`.

**Why this matters:** Constrained decoding is best-effort; a single malformed payload must not crash the turn. Enforces the framework's own `model_validate_json` contract in code, not just docs.

---

## Test 4: test_step_and_stream_share_error_behavior

**What it verifies:** Given a state whose `run()` raises, both `step` and `step_stream` produce the same outcome (reroute to `agent_reply` / `error` signal) — no divergence between the two paths.

**Why this matters:** Two code paths that handle errors differently is a latent correctness bug; the streaming and non-streaming entrypoints must agree on failure semantics.

---

## Test 5: test_loop_returns_named_terminal_reason

**What it verifies:** Hitting `max_iter` returns terminal reason `max_steps`; a clean finish returns `completed`; an unrecovered model error returns `model_error`.

**Why this matters:** Callers (HTTP layer, task runner) currently cannot tell *why* a turn ended. Named reasons let them retry, surface, or mark a task failed correctly.

---

## Test 6: test_buddy_tool_content_is_bounded / test_no_dead_structured_data

**What it verifies:** A tool returning output larger than the remaining context budget has its `content` rendered to a bounded head+tail view (with a marker) before entering context; under-budget output is unchanged. And the buddy tool state emits no `structured_data["tool_result"]` key.

**Why this matters:** Buddy's unbounded `str(tool_result)` is the actual 413; the dead stash is write-only weight that risks leaking the full payload if ever logged. This pins both fixed.

---

## Test 7: ~~test_tool_selected_and_filled_in_one_call~~ — removed

**Removed:** single-call tool selection was cut from Task 4 (see its *Out of scope*). Tool-selection mechanics live in `ARCHITECTURE_SPEC.md`, where the 2-call `select→fill` is a prior **replaced** by native tool-calling, not optimized in place. No harness test needed.

---

## Test 8: test_smithery_timeout_raises_not_hangs

**What it verifies:** A Smithery endpoint that never responds causes the call to raise a typed/classified error within the timeout, not block indefinitely.

**Why this matters:** A hung external dependency hanging the whole agent is one of the worst live-demo failure modes; the timeout converts it into a recoverable error.

---

## Test 9: test_network_blip_skips_server_not_crash

**What it verifies:** A `SmitheryError` from one server during tool discovery is logged and that server skipped; the tool call proceeds with the remaining servers instead of aborting. A 401 surfaces as `AuthRequiredError` (connect prompt), not a generic error.

**Why this matters:** One flaky MCP server must not take down every tool; and auth-needed must be distinguishable from down so the user gets the right prompt.

---

## Test 10: test_required_config_missing_fails_fast

**What it verifies:** Booting with a required key/env missing (e.g. `database.url`) aborts startup with a clear message; `${VAR:-default}` resolves to the default when the env is unset. (Plus `test_failed_spawn_does_not_orphan_running_task`: a spawn failure leaves the row `pending`/`failed`, never a stuck `running`.)

**Why this matters:** Late, cryptic config failures and orphaned "running" tasks both waste debugging time and erode trust in the system's own state — fail-fast and correct lifecycle make failures legible.

---

## Test 11: test_context_tokens_tracked / test_budget_keeps_headroom_below_limit

**What it verifies:** `agent.context_tokens` reflects the assembled context after `get_context()`; `tool_result_budget()` returns `context_window − context_tokens − output_reserve − margin` and never lets a rendered result push the prompt within the safety margin of the limit. When the context is nearly full, the budget shrinks toward zero (the compaction signal).

**Why this matters:** A fixed `[:400]` cap is both wasteful (ignores headroom on a big-context model) and unsafe (blind to a nearly-full context). Budgeting against live context size is what keeps the prompt off the 413 edge across model sizes — and the shrinking budget is the honest trigger for compaction.

---

# Open Questions

1. Real token streaming + interruption: wire `generate_stream` into `step_stream` and thread an `asyncio` cancellation through the loop so a user can interrupt a long run. Sizable enough for its own spec — does it block anything here? *Leaning: separate spec, no dependency on Tasks 1–4.*
2. Fallback-model switching (Claude Code `FallbackTriggeredError`): only meaningful once there is a second model endpoint. Defer until a fallback model exists.
3. Generalize `state_approval`'s human-in-the-loop into a reusable permission gate the buddy (chat) agent can use for destructive tools, not just the executor. Is chat-agent tool use trusted enough to defer? Revisit when the chat agent gains write/destructive tools.
4. Should a transient `ModelError` retry at the **step** level (re-run the state) or the **client** level only (Task 1)? Client-level backoff handles blips; step-level retry handles a state that consumed a bad partial result. *Leaning: client-level in Task 1, step-level retry cap in Task 3, no double-counting.*
5. (Task 7) ~~Token counting: tokenizer endpoint vs `tiktoken`?~~ **Resolved: `tiktoken`** as a cheap heuristic, with a fudge factor + larger safety margin to absorb the Qwen-vs-tiktoken mismatch. **Resolved: `llm.context_window` set in `config.yaml`.** Open caveat: the value must equal the server's launched `--max-model-len`, which may be *below* Qwen2.5-7B's 32k native window — confirm against how vLLM/SGLang was actually started, not the model's theoretical max.

---

# Implementation Notes

*Add entries here as work lands.*

- (pre-work) `step()` has no `try/except` around `state.run` (`agent.py:250`) while `step_stream()` does (`agent.py:318-328`) — Task 3 must converge them, and any caller relying on `step` raising will change behavior.
- (pre-work) `retrieve_short_memory(5)` is hardcoded in the loop (`agent.py:263,345`) — independent of this spec but in the same edit surface; coordinate with `MEMORY_SPEC.md` Task 2 to avoid conflicting edits.
- (pre-work) `generate_stream` (`ArkModelNew.py:171`) already exists and is unused — Open Question 1 is mostly wiring, not new client code.
- (pre-work) Tasks 5–6 came from the 2026-05-31 reliability audit; the cross-user/auth findings from that same audit live in `MULTIUSER_SPEC.md`, not here. Task 5 edits the same file (`smithery.py`) as `MULTIUSER_SPEC.md` Task 1 (registry scoping) — land them together to avoid two passes over the registry/discovery code.
- (pre-work) `memory.py:42` env-key override and the `smithery.py:375` `_pending` pop no-op are tracked in `MEMORY_SPEC.md` Task 1 and `MULTIUSER_SPEC.md` Task 1 respectively — not duplicated as harness tasks.
