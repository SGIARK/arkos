# Feature Spec: Model Layer Rewrite

**Sources**

- Audit: `model_module/ArkModelNew.py` (full read), usage across `agent_module/`,
  `base_module/`, `state_module/`, `computer_module/`
- Prior art: OpenAI SDK connection pooling docs, Anthropic SDK patterns
- Companion specs: `HARNESS_SPEC.md` (retry/error classification), `ISSUES.md`

**Status:** Not started | **Author:** | **Last updated:** 2026-06-17

---

# Problem

`ArkModelLink` is a Pydantic `BaseModel` — a data validation class — doing the job
of a stateful service object. Every call to `self.client` creates a new
`AsyncOpenAI` instance with its own connection pool, orphaning the previous one.
Provider differences (OpenAI vs local) are handled by boolean flags bolted onto
the data class. There is no finish_reason checking, no token usage logging, no
Retry-After respect, and no abstract interface — so adding a third provider or
swapping the retry policy requires touching application code.

**This is not a production model layer. It is a script.**

Specific failures confirmed in production:

| Failure | Evidence |
|---|---|
| New `AsyncOpenAI` instance per call | `@property def client` creates one every invocation |
| Connection pool leak under load | No `await client.close()` ever called; pools accumulate |
| Trailing prose breaks JSON parse | GPT-5.4 appends text after `}`; patched but not fixed |
| Content policy block returns empty string | `finish_reason="content_filter"` never checked |
| Context overflow silently truncates | `finish_reason="length"` never checked |
| Rate limit retry ignores Retry-After | Fixed backoff: 1s, 2s, 4s regardless of header |
| Token usage invisible | `usage.*` discarded; no cost tracking, no debugging |
| Provider flags will not scale | `use_max_completion_tokens: bool` — next model adds another flag |
| Startup fails at first request not boot | OPENAI_API_KEY validated lazily, not at startup |

**Success looks like:** a model call is a typed, observable unit — provider is an
implementation detail, failures are classified at the boundary, usage is logged on
every call, connection lifecycle is managed correctly, and adding a new provider or
model quirk requires a new class, not a new boolean.

---

# Technical Background

**Why `BaseModel` is wrong here.** Pydantic `BaseModel` is for data validation and
serialization. A model client is a stateful service with a connection pool, an API
key, retry state, and async lifecycle (`__aenter__`/`__aexit__`). Mixing these
causes: (1) the client property cannot cache its result (Pydantic fields are
validated at construction; a property that returns a new object is never cached),
(2) `api_key` is trivially serializable via `model.model_dump()` — a security smell,
(3) cleanup (`await client.close()`) has no natural home.

**Connection pooling.** `AsyncOpenAI` manages an internal `httpx.AsyncClient` with
a connection pool. The intended pattern is one `AsyncOpenAI` instance per process,
shared across all requests. Creating one per call defeats pooling and leaks file
descriptors. At 50 req/s this exhausts the default `ulimit -n` (1024) in ~20
seconds of sustained load.

**finish_reason.** OpenAI returns HTTP 200 even when the model was cut off or
blocked. The only signal is `choices[0].finish_reason`:
- `"stop"` — normal completion
- `"length"` — hit `max_completion_tokens`; response is truncated
- `"content_filter"` — policy block; response may be empty or redacted
- `"tool_calls"` — model invoked a tool (not used here, but must not be confused
  with `"stop"`)

Ignoring `finish_reason` means a truncated response (which breaks JSON parsing)
and a policy block (which returns empty content) are both silently treated as
successful completions.

**The provider interface pattern.** Both `AsyncOpenAI` (for OpenAI and local
SGLang, which is OpenAI-compatible) and future providers (Anthropic, Vertex, etc.)
implement the same conceptual operation: `(messages, json_schema) → str`. The
parameter names differ (`max_tokens` vs `max_completion_tokens` vs
`max_output_tokens`), but the interface is the same. The right design is an
abstract `ModelProvider` with concrete implementations that own their own
parameter mapping — not a growing list of boolean flags on a shared class.

---

# Proposed Approach

Three layers replacing `ArkModelNew.py`:

```
model_module/
  provider.py       # Abstract ModelProvider interface + CompletionResult
  openai_provider.py  # OpenAIProvider (covers local SGLang + OpenAI API)
  factory.py        # create_provider(config) — reads llm.provider from config
  errors.py         # ModelError, OutputValidationError (unchanged)
  llm_json.py       # parse_llm_json (unchanged)
  ArkModelNew.py    # DELETED or kept as thin shim for one release cycle
```

`agent_module/agent.py` calls only `provider.complete(messages, json_schema)`.
It never constructs an `AsyncOpenAI` directly. Provider lifecycle is managed by
the FastAPI lifespan handler.

What stays the same: `ModelError`, `OutputValidationError`, `parse_llm_json`,
`Message`/`AIMessage`/`SystemMessage`/`UserMessage`/`ToolMessage`.

---

# Implementation Plan

## Task 1: Abstract interface + CompletionResult

**Problem:** No typed contract between the agent loop and the model backend. The
loop calls `call_llm()` and gets back a string, discarding all metadata.

**Done when:**
- `model_module/provider.py` defines:
  ```python
  @dataclass
  class CompletionResult:
      content: str
      finish_reason: str          # "stop" | "length" | "content_filter" | ...
      prompt_tokens: int
      completion_tokens: int
      request_id: str | None      # provider-assigned, for support tickets
  
  class ModelProvider(ABC):
      @abstractmethod
      async def complete(
          self,
          messages: list[Message],
          json_schema: dict | None = None,
      ) -> CompletionResult: ...
  
      async def close(self) -> None: ...   # default no-op; override if needed
  ```
- `CompletionResult.finish_reason` is checked by every caller:
  `"length"` → `ModelError(retryable=False, "context window exceeded")`
  `"content_filter"` → `ModelError(retryable=False, "content policy block")`
- `prompt_tokens + completion_tokens` are logged via `emit_log` on every call.
- `request_id` is logged on every call and on every error so support tickets
  can be traced.

**Touch point:** `model_module/provider.py` (new), `agent_module/agent.py`
(`call_llm` updated to call `provider.complete` and handle `CompletionResult`).

**Priority:** P0 | **Effort:** ~1 day | **Blockers:** none

**Acceptance test:** `test_finish_reason_length_raises_model_error`,
`test_finish_reason_content_filter_raises_model_error`,
`test_completion_result_tokens_logged` (below).

---

## Task 2: OpenAIProvider — connection lifecycle + retry

**Problem:** New `AsyncOpenAI` per call; connection pool leak; retry ignores
`Retry-After`; no jitter; hard-coded count.

**Done when:**
- `model_module/openai_provider.py` defines `OpenAIProvider(ModelProvider)`:
  - One `AsyncOpenAI` instance created in `__init__`, stored as `self._client`.
  - `async def close()` calls `await self._client.close()`.
  - Accepts both `max_tokens` and `max_completion_tokens` via a single
    `_tokens_kwarg()` helper that inspects `self._use_completion_tokens` (set
    once at construction from config, never a per-call flag).
  - Accepts `temperature: float | None` — `None` means omit the parameter
    entirely (for models like o3 that reject it). Config sets this.
  - Retry loop:
    - Max retries from `config.get("llm.max_retries", 3)`.
    - On `RateLimitError`: extract `Retry-After` header from the exception
      response; use it if present, else exponential backoff with ±20% jitter.
    - On `InternalServerError`/`APITimeoutError`/`APIConnectionError`: retry
      with exponential backoff + jitter.
    - On `BadRequestError`/`AuthenticationError`/`PermissionDeniedError`: raise
      `ModelError(retryable=False)` immediately — no retry.
- FastAPI lifespan (`@asynccontextmanager`) calls `await provider.close()` on
  shutdown so connections drain cleanly.

**Touch point:** `model_module/openai_provider.py` (new), `base_module/app.py`
(lifespan handler), `config_module/config.yaml` (`llm.max_retries`,
`llm.temperature` made nullable).

**Priority:** P0 | **Effort:** ~1 day | **Blockers:** Task 1

**Acceptance test:** `test_single_client_instance_reused`,
`test_retry_respects_retry_after_header`,
`test_non_retryable_400_does_not_retry` (below).

---

## Task 3: Factory + config validation at startup

**Problem:** `OPENAI_API_KEY` checked lazily at first request. Config fields
duplicated (`llm.max_tokens` vs `llm.openai_max_tokens`). Adding a provider
requires editing application code.

**Done when:**
- `model_module/factory.py` defines `create_provider(config) -> ModelProvider`:
  reads `llm.provider`, validates required fields, raises `ValueError` with a
  clear message at startup (not at first request).
- Config is collapsed: no more `llm.openai_*` duplication. One `llm` block, one
  set of keys, provider-specific overrides only where semantics genuinely differ:

  ```yaml
  llm:
    provider: "openai"          # "openai" | "local"
    model: "gpt-5.4"            # model name; interpreted by the provider
    base_url: "https://api.openai.com/v1"   # overridable for local
    max_completion_tokens: 16384
    context_window: 1000000
    temperature: null           # null = omit (for models that reject it)
    max_retries: 3
  ```

- `config.validate_required(["llm.provider", "llm.model", ...])` called in
  `startup()` before any provider is constructed.
- If `provider="openai"` and `OPENAI_API_KEY` is unset, server refuses to start
  with a clear error — not a runtime 400 on the first user message.

**Touch point:** `model_module/factory.py` (new), `config_module/config.yaml`
(consolidated), `base_module/app.py` (`startup()`), `base_module/task_runner.py`
(`_shared_deps` uses factory).

**Priority:** P0 | **Effort:** ~0.5 day | **Blockers:** Task 2

**Acceptance test:** `test_missing_api_key_fails_at_startup`,
`test_factory_returns_correct_provider_type` (below).

---

## Task 4: Observability — token logging + request IDs

**Problem:** `usage.*` and `response.id` discarded on every call. No cost
visibility, no request tracing for support.

**Done when:**
- Every `CompletionResult` carries `prompt_tokens`, `completion_tokens`,
  `request_id` (populated from the API response).
- `agent.call_llm()` logs these via `emit_log` as a structured `LogEvent` with
  kind `"model_call"` including: model name, prompt tokens, completion tokens,
  latency ms, finish_reason, request_id.
- Log is queryable — not just `print()`. Feeds into whatever log sink is
  configured (today: JSON lines; tomorrow: Datadog/BQ).
- `agent.context_tokens` (already tracked) is cross-checked against
  `prompt_tokens` in the response; a large divergence (>20%) triggers a warning
  (tiktoken fudge factor is miscalibrated).

**Touch point:** `agent_module/agent.py` (`call_llm`), `logging_module/`
(new `model_call` LogEvent kind).

**Priority:** P1 | **Effort:** ~0.5 day | **Blockers:** Task 1

**Acceptance test:** `test_model_call_log_emitted_on_every_completion`,
`test_request_id_present_in_log` (below).

---

## Task 5: Delete ArkModelNew.py

**Problem:** The old class exists alongside the new provider; callers might
accidentally use it.

**Done when:**
- `ArkModelNew.py` is deleted.
- `Message`, `AIMessage`, `SystemMessage`, `UserMessage`, `ToolMessage` are moved
  to `model_module/messages.py` (they're just dataclasses; they don't belong in
  the model client file).
- All imports updated across the codebase.
- `ArkModelLink` is gone. Any reference fails loudly at import time.

**Touch point:** `model_module/` (deletion + new `messages.py`), all importers.

**Priority:** P1 | **Effort:** ~0.5 day | **Blockers:** Tasks 1–4 (all callers
migrated first)

**Acceptance test:** `grep -r ArkModelLink` returns no results.

---

# Tests

## Test 1: test_finish_reason_length_raises_model_error

**What it verifies:** A completion with `finish_reason="length"` raises
`ModelError(retryable=False)` with a message indicating context overflow — not a
silent truncated string.

**Why this matters:** Truncated JSON from a context overflow silently breaks
downstream parsing. The right behavior is a classified, non-retryable error that
the loop can surface clearly.

---

## Test 2: test_finish_reason_content_filter_raises_model_error

**What it verifies:** A completion with `finish_reason="content_filter"` raises
`ModelError(retryable=False)` — the empty/redacted content is never passed to the
state.

**Why this matters:** Without this check, a policy block looks identical to an
empty model response. The state tries to parse it, fails, and the user sees a
generic internal error with no indication of why.

---

## Test 3: test_completion_result_tokens_logged

**What it verifies:** After a successful completion, a `model_call` LogEvent is
emitted containing `prompt_tokens`, `completion_tokens`, `latency_ms`, and
`request_id`.

**Why this matters:** Token usage is the only cost signal. Without logging it,
there is no way to attribute cost, detect runaway prompts, or debug context
window issues in production.

---

## Test 4: test_single_client_instance_reused

**What it verifies:** Calling `provider.complete()` 10 times uses the same
underlying `AsyncOpenAI` instance (verified by identity check or mock call count).

**Why this matters:** The old `@property client` created a new instance on every
call. This test pins that the connection pool is never orphaned.

---

## Test 5: test_retry_respects_retry_after_header

**What it verifies:** When a `RateLimitError` carries a `Retry-After: 30` header,
the provider waits ≥30 seconds (or the mock time advances by that much) before
retrying — not the fixed 1s/2s/4s schedule.

**Why this matters:** Retrying a rate-limited endpoint on a fixed schedule
guarantees failure. `Retry-After` is the authoritative signal from the API.

---

## Test 6: test_non_retryable_400_does_not_retry

**What it verifies:** A `BadRequestError` (400) raises `ModelError(retryable=False)`
immediately with zero retries.

**Why this matters:** Retrying a bad request wastes time and burns quota. A 400
means the request is structurally wrong — retrying it identically will always fail.

---

## Test 7: test_missing_api_key_fails_at_startup

**What it verifies:** `create_provider(config)` with `provider="openai"` and no
`OPENAI_API_KEY` in the environment raises `ValueError` before any HTTP call is
made — not an `AuthenticationError` on the first user request.

**Why this matters:** Misconfiguration should fail loudly at startup, not silently
until the first user hits the broken endpoint.

---

## Test 8: test_factory_returns_correct_provider_type

**What it verifies:** `create_provider(config)` with `provider="local"` returns
an `OpenAIProvider` pointed at the local base URL; with `provider="openai"` returns
an `OpenAIProvider` pointed at the OpenAI API with the real key.

**Why this matters:** The factory is the single place where config becomes a
concrete provider. If it returns the wrong type, every downstream call is wrong.

---

# Open Questions

1. **Anthropic as a third provider.** The Anthropic SDK (`anthropic.AsyncAnthropic`)
   is not OpenAI-compatible. When adding it, `AnthropicProvider(ModelProvider)` is
   a new class, not a flag on `OpenAIProvider`. The factory reads `llm.provider:
   "anthropic"` and returns it. No changes to `OpenAIProvider` needed.
   *Leaning: design for it now (the interface already supports it), implement when
   needed.*

2. **Per-agent provider.** Today all agents share one provider. A future design
   might use `gpt-5.4` for the executor (quality matters) and a faster/cheaper
   model for buddy (latency matters). The factory could return a router that maps
   `agent_id` to a provider. *Out of scope for this spec — note for later.*

3. **Streaming.** `generate_stream` exists but is separate from `complete`.
   Should `ModelProvider` have a `stream(messages) -> AsyncIterator[str]` method?
   The current streaming path in `app.py` bypasses the state machine entirely.
   *Flag for the streaming spec (HARNESS_SPEC Open Question 1).*

---

# Implementation Notes

*Add entries here as work lands.*

- `ArkModelNew.py` is a `BaseModel` subclass. Callers use it as
  `ArkModelLink(base_url=..., max_tokens=...)` and pass it into `Agent.__init__`.
  The migration replaces that constructor call with `create_provider(config)` in
  `app.py` and `task_runner._shared_deps`. All other callers (states, computer
  agent) go through `agent.call_llm()` and never touch the provider directly —
  they migrate automatically.
- `computer_module/model.py` has its own `ToolCallingModel` class that also wraps
  `AsyncOpenAI` with a separate `@property client` bug. It needs the same fix but
  is out of scope here — track separately under `COMPUTER_SPEC.md`.
- The `_extract_first_json` patch in `llm_json.py` (2026-06-17) is a workaround
  for GPT-5.4 trailing prose. It should survive this rewrite unchanged — the JSON
  repair layer is independent of the provider layer.
