# Feature Spec: ARKOS Structured Logging and Session Auditability

**Sources**

- structlog docs -- standard Python structured logging library, industry default for structured JSON output
- Current code: `base_module/task_store.py` (`log_event`, `task_events` table), `agent_module/agent.py` (logger calls added in harness work), `base_module/app.py` (uvicorn startup, per-request handling)
- Companion specs: `HARNESS_SPEC.md` (TerminalReason, _run_state), `MEMORY_SPEC.md` (session_id per memory instance), `MULTIUSER_SPEC.md` (CurrentUser auth dependency)

**Status:** Not started | **Author:** | **Last updated:** 2026-06-02

---

# Problem

The server produces two kinds of output today: uvicorn HTTP access lines (`127.0.0.1 GET /health 200 OK`) which are noise, and silence everywhere else.

When the agent runs you cannot tell:

- What message the user sent
- Which state ran and what it decided
- Which tool was called, with what arguments, and what it returned
- Why the turn ended (completed? error? max steps?)
- Whether any session-level event happened for a given user

This makes debugging impossible in real time and makes session auditability -- "what did ARK do for user X on date Y?" -- not possible at all. There is no structured record of chat sessions. The `conversation_context` table stores raw message bytes, not events.

The existing `log_event` / `task_events` table solves this well for background task execution (subagent steps, tool calls, approvals are all logged). The same pattern needs to extend to the chat path, and all output needs to be structured and queryable.

The root cause of the silence: `logging.basicConfig` is never called anywhere, so every `logger.*` call added in the harness work goes silently to `/dev/null`. Uvicorn configures its own handlers (hence the HTTP lines), but our code configures nothing.

**Success looks like:**

In development, a developer sees one clean readable line per meaningful agent action, no HTTP noise:

```
14:32:01 [info ] ark.agent  user=alice request=a3f9 message="What is on my google calendar?"
14:32:01 [info ] ark.agent  user=alice state=agent_reply
14:32:03 [info ] ark.agent  user=alice llm_response elapsed_ms=1240 route=plan
14:32:03 [info ] ark.agent  user=alice transition from=agent_reply to=workshop_plan
14:32:03 [info ] ark.tool   user=alice tool_call tool=calendar_list args={"calendarId":"MIT..."}
14:32:04 [info ] ark.tool   user=alice tool_result tool=calendar_list elapsed_ms=310 result_chars=847
14:32:04 [info ] ark.agent  user=alice loop_exit reason=completed elapsed_ms=2840
```

In production the same call emits a JSON line that pipes into any log aggregator (Datadog, Loki, CloudWatch).

For any chat session or background task, a query against `audit_events` returns a complete ordered record of what happened -- who sent what, what the agent decided, what tools ran, what was returned.

---

# Technical Background

**Two distinct outputs -- different purpose, different retention:**

| Output | Purpose | Format | Retention |
|---|---|---|---|
| Operational log (stdout) | Debugging, monitoring, alerting | structlog pretty (dev) / JSON (prod) | Ephemeral, ships to log aggregator |
| Audit log (Postgres `audit_events`) | Permanent record per session | Structured rows | Permanent (or policy-gated, see Open Question 3) |

**Why structlog:**

structlog is the Python industry standard for structured logging. Every log call produces a dict of key-value context rather than a format string. A processor chain handles formatting: pretty-printed colored output in development (TTY detection), JSON lines in production. The same log call works in both environments with no code change.

**What already exists and must be preserved:**

`log_event(task_id, kind, content, payload)` in `task_store.py` writes to `task_events`. This is audit logging for background tasks and is already working correctly. This spec does not replace it. It adds coverage for the chat path via a new `audit_events` table, and in Open Question 4 considers whether to unify the two tables later.

**Context binding:**

structlog's `contextvars` integration lets you bind fields (user_id, request_id, session_id) once at request start and have every subsequent log call in that request automatically carry them. No explicit context threading required.

**The session_id gap:**

Background tasks already have a `session_id` (minted in `task_runner.py`). Chat turns do not -- the Memory object has one but it is not surfaced at the HTTP layer. For chat, this spec introduces a `request_id` (UUID per HTTP request) as the immediate audit anchor. Chat session continuity -- linking multiple turns from the same user into one logical session -- is deferred to Open Question 1.

---

# Proposed Approach

Four moves, in dependency order:

1. **Configure structlog at startup.** One call. Dev: pretty-print with TTY detection. Prod: JSON lines. Mute uvicorn access log. All existing `logger.*` calls now actually emit.
2. **Bind per-request context** via FastAPI middleware. Every log call in a request automatically carries `user_id` and `request_id` with no threading required.
3. **Add operational log events** at key agent loop points: message in, state enter/exit, tool call/result, transition, turn end, LLM call/retry. structlog only -- no DB write.
4. **Add audit log table + async writer** for events that must be permanently recorded: message received, response sent, tool called and result, errors. New `audit_events` table; background thread write so audit logging never adds latency to the request.

What stays the same: `log_event` / `task_events` is untouched. The executor path already has good audit coverage. This spec adds chat-path coverage and upgrades stdout quality for both.

Explicitly not in scope:
- Log aggregator integration (Datadog, Loki, CloudWatch) -- infra config, not code.
- Chat session continuity (linking N turns into one session) -- Open Question 1.
- PII redaction or field-level encryption -- Open Question 2.
- Audit log API endpoint (`GET /audit/sessions/:id`) -- Task 5, lowest priority, needs auth from MULTIUSER_SPEC first.

---

# Event Taxonomy

`[O]` = operational log (structlog stdout only)
`[A]` = audit log (structlog + `audit_events` DB row)

## Chat session events

| Event | Channel | Key fields |
|---|---|---|
| `message_received` | [A] | user_id, request_id, message_preview (first 200 chars), token_count |
| `message_sent` | [A] | user_id, request_id, response_preview, terminal_reason, elapsed_ms |
| `session_error` | [A] | user_id, request_id, error_type, error_detail |

## Agent loop events

| Event | Channel | Key fields |
|---|---|---|
| `state_enter` | [O] | state_name, step_count |
| `state_exit` | [O] | state_name, completion_signal, elapsed_ms |
| `transition` | [O] | from_state, to_state, route_signal |
| `loop_exit` | [O] | terminal_reason, total_elapsed_ms, step_count |

## LLM events

| Event | Channel | Key fields |
|---|---|---|
| `llm_call` | [O] | context_tokens, schema_name |
| `llm_response` | [O] | elapsed_ms |
| `llm_retry` | [O] | attempt, reason |
| `llm_error` | [A] | retryable, error_type, error_detail |

## Tool events

| Event | Channel | Key fields |
|---|---|---|
| `tool_selected` | [O] | tool_name, server_name |
| `tool_call` | [A] | tool_name, args_preview (truncated, no secrets), user_id |
| `tool_result` | [A] | tool_name, elapsed_ms, result_chars, truncated |
| `tool_error` | [A] | tool_name, error_type, error_detail |

## Task / subagent events (existing task_events table -- unchanged)

These already write to `task_events` via `log_event`. This spec does not change them.
In a later pass (Open Question 4) we may mirror a subset to `audit_events` as well.

---

# Implementation Plan

## Task 1: Configure structlog + mute uvicorn access log

**Problem:** `logging.basicConfig` is never called. Every `logger.*` call goes to /dev/null. Uvicorn access log floods stdout with HTTP noise.

**Done when:**
- `structlog` is in `requirements.txt`.
- A `_configure_logging()` function is called inside `startup()` in `app.py`.
- Dev (TTY detected): `structlog.dev.ConsoleRenderer` -- colored, human-readable, one line per event.
- Prod (no TTY): `structlog.processors.JSONRenderer` -- one JSON object per event.
- `logging.getLogger("uvicorn.access").setLevel(logging.WARNING)` silences HTTP access lines.
- All existing `logger.info/warning/error` calls in `agent.py`, `app.py`, and the tool module now emit.

**Sketch (illustrative, not final):**

```python
import sys, logging, structlog

def _configure_logging() -> None:
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),
    ]
    if sys.stderr.isatty():
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
```

**Touch point:** `base_module/app.py` (startup), `requirements.txt`.

**Priority:** P0 | **Effort:** 0.5 days | **Blockers:** none

**Acceptance test:** `test_log_output_is_not_empty_on_agent_run` (below).

---

## Task 2: Per-request context middleware

**Problem:** Log lines from concurrent requests interleave with no way to filter by user or request.

**Done when:**
- A FastAPI middleware calls `structlog.contextvars.bind_contextvars(user_id=..., request_id=...)` at request start and `clear_contextvars()` after.
- Every log call in that request automatically carries both fields.
- `session_id` is bound when available (task path: from the task row; chat path: deferred to Open Question 1).

**Sketch (illustrative, not final):**

```python
@app.middleware("http")
async def bind_logging_context(request: Request, call_next):
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=uuid.uuid4().hex[:12],
        user_id=request.headers.get("X-User-ID") or "anon",
    )
    return await call_next(request)
```

**Touch point:** `base_module/app.py`.

**Priority:** P0 | **Effort:** 0.5 days | **Blockers:** Task 1

**Acceptance test:** `test_concurrent_requests_have_isolated_request_ids` (below).

---

## Task 3: Operational log events in the agent loop

**Problem:** No visibility into state transitions, LLM calls, tool selections, or turn duration. The structlog calls from the harness work are ready; they just need to use `structlog.get_logger()` and emit at the right points.

**Done when:**
- `agent.step()` / `step_stream()` log `message_received` (user message preview, token_count) at turn start and `loop_exit` (terminal_reason, elapsed_ms, step_count) at turn end.
- `_run_state()` logs `state_enter` before the call and `state_exit` (completion_signal, elapsed_ms) after.
- `choose_transition()` logs `transition` (from_state, to_state, route_signal).
- `ArkModelLink.generate_response()` logs `llm_call` (context_tokens) before and `llm_response` (elapsed_ms) after, and `llm_retry` on each backoff.
- `state_tool.py` (buddy and executor) logs `tool_selected`, `tool_call` (tool_name, args_preview), and `tool_result` (elapsed_ms, result_chars, truncated flag).
- All use `structlog.get_logger("ark.agent")` / `"ark.tool"` not `logging.getLogger()`.

**Sketch (illustrative, not final):**

```python
# agent.py step()
log = structlog.get_logger("ark.agent")
log.info("message_received", preview=content[:120], tokens=self.context_tokens)
# ... loop body using _run_state ...
log.info("loop_exit", reason=self.terminal_reason, elapsed_ms=elapsed, steps=retry_count)

# state_tool.py run()
log = structlog.get_logger("ark.tool")
log.info("tool_call", tool=tool_name, args=_safe_args_preview(tool_args))
log.info("tool_result", tool=tool_name, elapsed_ms=elapsed, chars=len(view), truncated=was_truncated)
```

**Touch point:** `agent_module/agent.py`, `state_module/agent_buddy/state_tool.py`, `state_module/agent_executor/state_tool.py`, `model_module/ArkModelNew.py`.

**Priority:** P0 | **Effort:** 1 day | **Blockers:** Task 1, Task 2

**Acceptance test:** `test_turn_produces_expected_log_events` (below).

---

## Task 4: Audit log table + async writer

**Problem:** No permanent record of chat session events. Sessions cannot be reconstructed after the fact.

**Done when:**
- New migration `db/migrations/0006_audit_events.sql`:

```sql
CREATE TABLE IF NOT EXISTS audit_events (
    event_id    BIGSERIAL    PRIMARY KEY,
    user_id     TEXT         NOT NULL,
    request_id  TEXT         NOT NULL,
    session_id  TEXT,
    event_type  TEXT         NOT NULL,
    payload     JSONB        NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_audit_user_time
    ON audit_events (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_session
    ON audit_events (session_id)
    WHERE session_id IS NOT NULL;
```

- A non-blocking `audit_log(event_type, **payload)` function in `logging_module/audit.py` pulls bound context (user_id, request_id, session_id) from structlog contextvars, writes a JSON row to `audit_events` via a background thread pool, and also calls structlog so audit events appear in the operational log with logger name `ark.audit`.
- The [A]-tagged events in the taxonomy call `audit_log()`: `message_received`, `message_sent`, `tool_call`, `tool_result`, `llm_error`, `session_error`.

**Sketch (illustrative, not final):**

```python
# logging_module/audit.py
from concurrent.futures import ThreadPoolExecutor
import structlog

_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audit_bg")

def audit_log(event_type: str, **payload) -> None:
    """
    Emit to structlog (immediate) and write to audit_events (background).
    Never blocks the caller.
    """
    ctx = structlog.contextvars.get_contextvars()
    structlog.get_logger("ark.audit").info(event_type, **payload)
    _pool.submit(_write_row, event_type, ctx, payload)

def _write_row(event_type: str, ctx: dict, payload: dict) -> None:
    # plain psycopg2 insert, no async, runs in thread pool
    ...
```

**Touch point:** new `logging_module/__init__.py` + `logging_module/audit.py`, new migration `db/migrations/0006_audit_events.sql`, call sites in `agent_module/agent.py` and `state_module/agent_buddy/state_tool.py`.

**Priority:** P1 | **Effort:** 1.5 days | **Blockers:** Task 2

**Acceptance test:** `test_tool_call_writes_audit_row`, `test_audit_write_is_non_blocking` (below).

---

## Task 5: Audit query endpoint

**Problem:** The audit table exists but is only accessible via direct DB query.

**Done when:**
- `GET /audit/sessions` returns distinct sessions for the authenticated user: `[{request_id, session_id, first_event_at, last_event_at, event_count}]`.
- `GET /audit/sessions/{request_id}` returns ordered events for that request: `[{event_type, payload, created_at}]`.
- Both endpoints require a valid Bearer token via the `CurrentUser` dependency from `MULTIUSER_SPEC.md` Task 2 and are scoped strictly to the calling user -- no cross-user access.

**Touch point:** new `base_module/audit_router.py`, `base_module/app.py` (router registration).

**Priority:** P2 | **Effort:** 1 day | **Blockers:** Task 4, `MULTIUSER_SPEC.md` Task 2

**Acceptance test:** `test_audit_session_list_scoped_to_user` (below).

---

# Tests

## Test 1: test_log_output_is_not_empty_on_agent_run

**What it verifies:** After a complete mocked agent turn, at least one structlog event is captured by a test log handler. No log call goes silently to /dev/null.

**Why this matters:** This is the root failure. The test fails today and passes after Task 1. Pins the guarantee that logging configuration is never accidentally removed.

---

## Test 2: test_concurrent_requests_have_isolated_request_ids

**What it verifies:** Two concurrent requests run through the middleware. Log events captured from request A all carry request A's request_id; events from request B carry B's. No bleed between them.

**Why this matters:** structlog contextvars are async-safe via contextvars.copy_context, but only if bind/clear are called correctly per request. A failure here means user A's audit trail gets entries from user B's request -- an auditability failure.

---

## Test 3: test_turn_produces_expected_log_events

**What it verifies:** A complete mocked agent turn (message in, state run, tool call, response out) emits these events in order: `message_received`, `state_enter`, `tool_call`, `tool_result`, `state_exit`, `loop_exit`.

**Why this matters:** Pins the full operational visibility requirement. A future regression that removes a log point breaks this test before anyone notices the terminal went quiet.

---

## Test 4: test_tool_call_writes_audit_row

**What it verifies:** After `audit_log("tool_call", tool="calendar_list", ...)` is called with user context bound, a row appears in `audit_events` with `event_type='tool_call'`, the correct `user_id`, and `tool` in the payload JSON.

**Why this matters:** This is the core auditability requirement. A tool call with no audit row means a session cannot be reconstructed.

---

## Test 5: test_audit_write_is_non_blocking

**What it verifies:** `audit_log()` returns in under 5ms even when the background DB write is artificially delayed to 200ms.

**Why this matters:** If audit writes block requests, the instinct under load will be to remove them. Non-blocking is what keeps audit logging in place permanently.

---

## Test 6: test_audit_session_list_scoped_to_user

**What it verifies:** `GET /audit/sessions` for user A returns only user A's sessions. A request authenticated as user B cannot retrieve user A's audit rows, even with a known request_id.

**Why this matters:** Audit logs contain the full text of what a user said to the agent. Cross-user access is a privacy failure.

---

# Open Questions

1. **Chat session continuity.** A logical "session" should link multiple HTTP requests from the same conversation. A user sends 5 messages -- that is one session, not 5 request_ids. How do we thread a chat session_id through the HTTP layer? Options: (a) the frontend sends `X-Session-ID` that the middleware binds; (b) we generate one on first message and return it to the client for reuse; (c) we surface the Memory object's session_id after `_get_or_create_memory`. Resolve before Task 4 ships so the `session_id` column is populated from day one.

2. **PII and sensitive fields.** Tool arguments and response content may contain PII (names, email addresses, calendar event titles). Should `audit_events` rows be encrypted at rest, or should certain fields be redacted before writing? The `args_preview` in operational logs has the same issue. For now: implement a simple key-name blocklist (token, key, secret, password, credential) to redact from args_preview. Full policy decision deferred.

3. **Audit log retention.** How long are `audit_events` rows kept? Indefinite retention is the safe default but grows unbounded. A 90-day rolling delete is common; some compliance contexts require 1-2 years. Resolve based on MIT/SIPB data-retention policy before going to production.

4. **task_events unification.** `task_events` (executor audit) and `audit_events` (chat audit) overlap in purpose. Should they eventually be one table? Leaning: keep `task_events` as the source of truth for the task progress UI (it is already wired to the frontend), and write a subset of executor events (tool_call, tool_result, task_completed, task_failed) to `audit_events` as well so a single audit query covers the full session. Revisit after Task 4 ships.

---

# Implementation Notes

*Add entries here as work lands.*

- (pre-work) `structlog` is not in `requirements.txt` -- add it alongside tiktoken/httpx from the harness work.
- (pre-work) `logging_module/` does not yet exist. `CLAUDE.md` references it as the home for `emit_log`. Task 4 creates it via `logging_module/audit.py`. Coordinate with `MEMORY_SPEC.md` which also references `emit_log` -- the two should share one module.
- (pre-work) The `args_preview` function that sanitizes tool arguments must be written before Task 3 lands. Key-name blocklist: `token`, `key`, `secret`, `password`, `credential`, `auth`. Truncate value to 80 chars after redaction.
- (sequencing) Tasks 1 and 2 are prerequisites for everything. Task 3 (operational) and Task 4 (audit) are independent of each other after Task 2. Task 5 requires `MULTIUSER_SPEC.md` Task 2 (auth) as an additional prerequisite.
- (cross-link) Open Question 1 (session continuity) must be resolved before Task 4 -- the `session_id` column in `audit_events` is useless if we never populate it.
