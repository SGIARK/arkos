# Feature Spec: ARKOS Multi-User Correctness & Security

**Sources**

- Codebase reliability audit (2026-05-31) — categories C (shared mutable state) and D (unverified identity), the two findings that neither `HARNESS_SPEC.md` nor `MEMORY_SPEC.md` covers
- Current code: `tool_module/smithery.py` (global `_tool_registry`, `_user_tools`, `_pending`), `base_module/app.py` (header-trust identity, `_memory_cache`, `_system_prompt`, `_available_tools`), `base_module/jwt_utils.py` (default secret), `base_module/task_store.py`
- Companion specs: `arkos-inspo/specs/HARNESS_SPEC.md` (brittleness/resilience), `arkos-inspo/specs/MEMORY_SPEC.md` (memory layers)

**Status:** Not started | **Author:** | **Last updated:** 2026-05-31

---

# Problem

The MVP is **multi-user**, but the server treats users as if there were only one. Two classes of bug make a multi-user demo unsafe — and both are *worse* than a crash, because they fail silently and cross the trust boundary between users.

**C — Shared mutable state leaks across users.** A single `tool_manager` singleton (`app.py:99`) holds a **global** tool registry keyed only by tool name:

```python
self._tool_registry: dict[str, str] = {}        # smithery.py:257 — tool_name -> server_name, NO user dimension
...
self._tool_registry[tname] = server_name         # :319, :373 — written per-user during discovery
...
server_name = self._tool_registry.get(tool_name) # :434 — read at execution, for ANY user
```

If user A connects a `search` tool from server X and user B connects `search` from server Y, the last writer wins and **both users' calls route to one user's server** — cross-user data exposure. This is wrong even single-threaded; concurrency only makes which-user-wins nondeterministic. The same singleton also caches `_system_prompt` / `_available_tools` / `_memory_cache` (`app.py:61,194`) mutated without locks, so concurrent requests can read half-written tool lists.

**D — Identity comes from an unverified header.** Every per-user endpoint derives the user from a header, with a fallback that silently impersonates a shared account:

```python
user_id = request.headers.get("X-User-ID") or config.get("memory.fallback_user_id")   # app.py:238,299,344,371,483
```

There is no JWT/Bearer verification on these paths (`jwt_utils.py` exists but isn't applied here). Any client can set `X-User-ID: <victim>` and read/write that user's memory, tasks, and OAuth connections. The JWT secret also defaults to `"ark-dev-secret-change-me"` (`jwt_utils.py:26`), so even where tokens *are* used they're forgeable if the env var is unset.

**What this costs:** in a multi-user demo, user A seeing user B's connected tools, memory, or tasks is a credibility-ending failure. None of it is caught today; it just silently produces the wrong answer for the wrong person.

**Success looks like:** every per-user resource (tools, memory, tasks, OAuth state) is keyed by a *verified* user id; no request can read or mutate another user's state; shared caches are either per-user or concurrency-safe; and a missing/forged token is rejected, not silently mapped to a fallback account.

---

# Technical Background

**Where identity flows today:** `app.py` reads `X-User-ID` (or a query param, or the JSON body) and passes that `user_id` straight into `Memory(user_id=...)`, `tool_manager.call_tool(user_id=...)`, and the task tables. The string is trusted end to end; nothing verifies the caller *is* that user.

**Why the registry is global:** `SmitheryManager` was written for a single-tenant assumption — one registry mapping tool→server. Per-user OAuth (`_user_tools`, `_pending` *are* keyed by user) was bolted on later, but `_tool_registry` (the map execution actually uses, `:434`) was never given the user dimension. So per-user *discovery* is isolated but per-user *routing* is not.

**Existing material to build on:**
- `jwt_utils.py` already has encode/verify helpers — the gap is *applying* them as a FastAPI dependency, not writing crypto.
- `_user_tools[user_id][server]` and `_pending[user_id][server]` already prove the per-user keying pattern; `_tool_registry` just needs the same shape.
- `task_store.py` queries already filter by `user_id` in SQL — they're correct *if* the `user_id` handed in is trustworthy, which is exactly what Task 2 fixes.

**Key constraint:** this is additive and mostly structural — add a user dimension to one dict, add one auth dependency, wrap or per-user-key three caches. No change to the agent loop, the state graph, or memory layers. It must land *before* a multi-user demo, independent of the harness work.

---

# Proposed Approach

Four moves, trust-boundary-first:

1. **User-scope the tool registry.** Give `_tool_registry` a user dimension (`[user_id][tool] -> server`) so routing can never cross users. Mirror the keying `_user_tools` already uses.
2. **Verify identity at the edge.** A single FastAPI `CurrentUser` dependency that validates a Bearer/JWT and yields the user id; every per-user endpoint depends on it. `fallback_user_id` is allowed *only* when an explicit `ARK_DEMO_MODE` flag is set, never by default.
3. **Make shared server state safe.** Per-user-key or lock `_memory_cache`, `_system_prompt`, `_available_tools`, and the registry mutations so concurrent requests can't read half-written or another user's data.
4. **Secrets fail-fast.** Refuse to start if the JWT secret is the built-in default outside demo mode; remove the `memory.py` env-key override (tracked in `MEMORY_SPEC.md` Task 1, cross-linked here).

What stays the same: the agent loop, state graph, memory layers, and the per-user OAuth *discovery* flow (already correctly keyed).

Explicitly **not in scope:**
- Full RBAC / roles / per-tool authorization — only owner-level isolation (a user reaches only their own resources).
- Rate limiting, audit logging, CORS hardening — separate hardening pass (CORS noted in `HARNESS_SPEC.md` Task 6).
- Encrypting per-user OAuth tokens at rest (Smithery holds them write-only today) — revisit if storage moves in-house.

---

# Implementation Plan

## Task 1: User-scope the tool registry

**Problem:** A global `_tool_registry` (`smithery.py:257`) routes tool execution by tool name with no user dimension; users with colliding tool names hit each other's MCP servers.

**Done when:**
- `_tool_registry` becomes `dict[user_id, dict[tool_name, server_name]]` (mirrors `_user_tools`); writes at `:319,:373` and the read at `:434` (and the scan at `:457`) are user-scoped.
- `call_tool` resolves the server from the **calling user's** registry only; a tool not in that user's map raises a clear "not connected" / `AuthRequiredError`, never another user's server.
- `reset()` (`:512`) and the `_pending` pop bug (`:375`, operates on a throwaway dict) are fixed in the same pass.

**Touch point:** `tool_module/smithery.py`.

**Priority:** P0 | **Effort:** ~1 day | **Blockers:** none

**Out of scope:** Tool-list caching strategy / TTL (functional correctness only here).

**Acceptance test:** `test_tool_registry_is_user_scoped`, `test_colliding_tool_names_do_not_cross_users` (below).

---

## Task 2: Verify identity at the edge

**Problem:** Per-user endpoints trust `X-User-ID` (`app.py:238,299,344,371,483`); any client can impersonate any user.

**Done when:**
- A `CurrentUser` FastAPI dependency validates a Bearer token via `jwt_utils` and returns the verified user id; all per-user endpoints (`/chat`, `/list_services`, `/connect`, `/disconnect`, `/oauth/callback`, task endpoints) depend on it instead of reading the header directly.
- `fallback_user_id` is used **only** when `ARK_DEMO_MODE` is explicitly set; in normal mode a missing/invalid token returns `401`, never a fallback identity.
- The verified id — not a header — is what flows into `Memory`, `call_tool`, and the task tables.

**Touch point:** `base_module/app.py`, `base_module/jwt_utils.py`, new `CurrentUser` dependency.

**Priority:** P0 | **Effort:** ~1–2 days | **Blockers:** none

**Out of scope:** Login/token issuance UX, refresh tokens, roles (owner-level isolation only).

**Acceptance test:** `test_missing_token_rejected_outside_demo`, `test_header_cannot_override_verified_identity` (below).

---

## Task 3: Make shared server state concurrency- and user-safe

**Problem:** `_memory_cache`, `_system_prompt`, `_available_tools` (`app.py:61,194`) and registry mutations are read/written across concurrent requests without synchronization → corrupted tool lists mid-turn, duplicate `Memory` init.

**Done when:**
- `_memory_cache` access is guarded by an `asyncio.Lock` (or keyed factory) so a user's `Memory` is created once.
- `_system_prompt` / `_available_tools` are swapped atomically (build new, assign) rather than mutated in place; readers never see a half-built value.
- Any remaining cross-request dict mutation in `app.py`/`smithery.py` is either per-user-keyed (Task 1) or lock-guarded.

**Touch point:** `base_module/app.py`, `tool_module/smithery.py`.

**Priority:** P0 | **Effort:** ~1 day | **Blockers:** Task 1 (registry shape)

**Out of scope:** Moving to an external cache/store (in-process is fine at MVP scale).

**Acceptance test:** `test_concurrent_requests_get_isolated_memory` (below).

---

## Task 4: Secrets fail-fast

**Problem:** JWT secret defaults to `"ark-dev-secret-change-me"` (`jwt_utils.py:26`) → forgeable tokens if env unset; `memory.py:42` overwrites the real LLM key.

**Done when:**
- Startup refuses to boot if the JWT secret equals the built-in default and `ARK_DEMO_MODE` is not set.
- The `memory.py:42` `os.environ["OPENAI_API_KEY"] = "sk"` override is removed (cross-linked to `MEMORY_SPEC.md` Task 1 — fix in whichever lands first, don't duplicate).
- No hardcoded credentials remain on the per-user request path.

**Touch point:** `base_module/jwt_utils.py`, `base_module/app.py` (startup check), `memory_module/memory.py`.

**Priority:** P0 | **Effort:** ~0.5 day | **Blockers:** none

**Out of scope:** Secret rotation, a secrets manager.

**Acceptance test:** `test_startup_fails_on_default_secret_in_prod` (below).

---

# Tests

## Test 1: test_tool_registry_is_user_scoped

**What it verifies:** Tools discovered for user A are resolvable only via user A's registry; user B's `call_tool` for a tool A connected raises "not connected", not a silent route to A's server.

**Why this matters:** This is the cross-user data-exposure bug. The test pins that routing can never leave the calling user's namespace.

---

## Test 2: test_colliding_tool_names_do_not_cross_users

**What it verifies:** When A and B both have a tool named `search` from different servers, each user's call routes to *their own* server regardless of discovery order or interleaving.

**Why this matters:** The collision case is the exact trigger of the global-registry bug and is invisible in single-user testing — it must be tested explicitly.

---

## Test 3: test_missing_token_rejected_outside_demo

**What it verifies:** With `ARK_DEMO_MODE` unset, a request with no/invalid Bearer token returns 401 and never falls back to `fallback_user_id`; with the flag set, the fallback applies.

**Why this matters:** The fallback-as-default is the impersonation hole; the test pins that real mode requires a verified identity.

---

## Test 4: test_header_cannot_override_verified_identity

**What it verifies:** A request with a valid token for user A but `X-User-ID: B` operates as A; the header cannot override the verified id.

**Why this matters:** Stops header-trust from re-entering through a side door once tokens are in place.

---

## Test 5: test_concurrent_requests_get_isolated_memory

**What it verifies:** Concurrent requests for the same user create exactly one `Memory`; concurrent requests for different users never share or corrupt cached tool/prompt state.

**Why this matters:** Guards the shared-mutable-state race that corrupts tool lists mid-turn under real (concurrent) load.

---

## Test 6: test_startup_fails_on_default_secret_in_prod

**What it verifies:** Booting with the default JWT secret and no demo flag aborts startup with a clear error; a real secret boots normally.

**Why this matters:** Forgeable tokens defeat all of Task 2; fail-fast turns a silent vulnerability into a loud, unmissable startup error.

---

# Open Questions

1. Token issuance: where do users get a JWT for the MVP — existing `users` table + a login endpoint, or an external IdP? Task 2 assumes verification exists; issuance may be a prerequisite. *Resolve before Task 2 lands.*
2. Does the MVP need *any* unauthenticated surface (health, public landing)? If so, enumerate it explicitly so `CurrentUser` is applied everywhere else by default-deny, not opt-in.
3. `fallback_user_id` exists in config and is referenced in 6 places — is demo mode a launch requirement, or can the fallback be deleted entirely? Deleting is safer; keeping it requires the `ARK_DEMO_MODE` gate to be airtight.
4. Per-user OAuth tokens live in Smithery (write-only). If any token material ever lands in our logs/DB, this spec's scope expands to encryption-at-rest — confirm none does.

---

# Implementation Notes

*Add entries here as work lands.*

- (pre-work) `_user_tools` and `_pending` are already `[user_id][...]`-keyed — Task 1 makes `_tool_registry` match them, so the shape is proven, not novel.
- (pre-work) `jwt_utils.py` already has verify helpers; Task 2 is wiring a dependency, not writing crypto.
- (pre-work) `_pending.get(user_id, {}).pop(server_name, None)` (`smithery.py:375`) pops from a throwaway dict — entries never clear. Fold the one-line fix into Task 1.
- (cross-link) `memory.py:42` key override is listed in both this spec (Task 4) and `MEMORY_SPEC.md` Task 1 — fix once, in whichever lands first.
