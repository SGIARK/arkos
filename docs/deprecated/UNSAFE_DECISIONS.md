# Accepted Risks & Security Debt Register

*A living list of decisions where we knowingly traded safety/correctness for MVP speed. Each entry: what it is, why it's unsafe, why we accepted it, the blast radius, and the trigger that forces a fix. Nothing here is a bug we missed -- it's debt we chose. Review this list before any public / multi-tenant deployment.*

**Status:** Living | **Last updated:** 2026-06-05

Legend -- **Blast radius:** who is harmed if it's exploited. **Fix-by:** the condition that turns "accepted" into "must fix."

---

## U1 -- OAuth callback trusts the `user_id` query param

**Decision:** `/oauth/callback/{service}` (`app.py:369`) reads `user_id` from the URL query string with no token verification.

**Why unsafe:** A crafted GET to the callback could complete/refresh a Smithery connection for an arbitrary `user_id`. There's no proof the caller is that user.

**Why accepted:** It's a third-party browser redirect *from Smithery* after OAuth -- it physically cannot carry our Bearer token. The `setup_url` it returns to was minted for that specific user, so in the normal flow the `user_id` is correct.

**Blast radius:** Cross-user -- could attach/refresh a service connection under someone else's account. Limited (no data read), but a trust-boundary crossing.

**Fix-by:** Before real multi-user production. Fix = sign the `user_id` into the OAuth `state`/return URL when minting the `setup_url`, and verify the signature on callback. Tracked in `OPENHANDS_SPEC`/`MULTIUSER` discussions.

---

## U2 -- `ARK_DEMO_MODE` allows `X-User-ID` impersonation

**Decision:** When `ARK_DEMO_MODE` is set, a request with no Bearer token may pass an `X-User-ID` header and be treated as that user (`jwt_utils.get_current_user`). It is **ON in our dev `.env`**.

**Why unsafe:** With the flag on, any client can impersonate any user by setting one header -- full read/write of that user's memory, tasks, tools, and (soon) their computer/files.

**Why accepted:** Dev/testing convenience -- curl, test scripts, and internal tools hit the API without minting a token. The real frontend always sends a Bearer token, so it does not rely on this.

**Blast radius:** Total, per-user, **only when the flag is on.** With it off, this path returns 401.

**Fix-by:** `ARK_DEMO_MODE` **must be unset/false in any shared or public deployment.** This is the single most important switch in the system. Startup should warn loudly when it is on.

---

## U3 -- `fallback_user_id` (unauthenticated -> shared account)

**Decision:** Historically, requests with no identity fell back to `config.memory.fallback_user_id` ("arkos-agent"). After Task 1, this only applies under `ARK_DEMO_MODE`.

**Why unsafe:** Multiple unauthenticated callers share one account's memory/tools -- data bleed between whoever hits the API unauthenticated.

**Why accepted:** Kept for demo mode only; gated by U2's flag.

**Blast radius:** Anyone hitting the API unauthenticated under demo mode shares one bucket.

**Fix-by:** Delete `fallback_user_id` entirely once demo mode is no longer needed. Same gate as U2.

---

## U4 -- User files live in e2b's cloud (no independent copy)

**Decision:** The per-user computer uses e2b pause/resume; the filesystem is the system of record on **e2b's** infrastructure. We store only the `sandbox_id`.

**Why unsafe:** User data lives on a third party. An e2b outage, snapshot eviction, or account/billing lapse loses the user's files. We have no backup to restore from.

**Why accepted:** Fastest path to a working computer; e2b absorbs all the isolation/ops. Proven in the Task 0 spike.

**Blast radius:** Data loss (availability/durability), per user. Not a confidentiality leak (e2b isolates per sandbox).

**Fix-by:** If users accumulate valuable work -> move to volume-mount persistence (durable copy in our storage) or snapshot important files to our object store on task completion. `SandboxManager` is the only touch point, so this is swappable.

---

## U5 -- No encryption at rest

**Decision:** `conversation_context` (chat text), task payloads, and any future stored tokens/PII are plaintext in Postgres.

**Why unsafe:** A DB compromise exposes everything users said and did in cleartext. Becomes acute when we store a raw GitHub PAT (deferred) or other credentials.

**Why accepted:** MVP; Smithery currently holds OAuth tokens write-only, so no raw third-party tokens live in our DB *yet*.

**Blast radius:** Everything in the DB on a DB breach.

**Fix-by:** Before storing any raw credential (GitHub PAT path) or going multi-tenant prod. Encrypt sensitive columns / use a secrets manager.

---

## U6 -- Weak model with shell access can trash a user's own workspace

**Decision:** The computer-agent runs the local Qwen with `run_command` in the user's sandbox. The only guardrail in v1 is the step cap.

**Why unsafe:** A weak model can issue wrong/destructive commands (`rm -rf`, overwrite files) inside the user's sandbox. Contained to that user's own computer (e2b isolation prevents cross-user harm), but it can destroy *their* files.

**Why accepted:** MVP; we want to see how Qwen does before adding friction. e2b contains the blast to one sandbox.

**Blast radius:** The acting user's own workspace only. No cross-user reach.

**Fix-by:** Add the approval gate for destructive actions (already a planned pattern), per-task snapshots/undo, and/or swap to a frontier model. Revisit once real tasks run.

---

## U7 -- Long-lived JWTs, no revocation

**Decision:** `issue_token` mints 30-day HS256 tokens (`jwt_utils.py`); there is no revocation list or refresh.

**Why unsafe:** A leaked token is valid for 30 days with no way to revoke it short of rotating the signing secret (which invalidates everyone).

**Why accepted:** Demo-grade auth; `/auth/demo-login` issues without a password.

**Blast radius:** A single leaked token = that user for 30 days.

**Fix-by:** Real auth phase -- shorter TTL + refresh tokens + a revocation/blocklist, and replace passwordless demo-login.

---

## U8 -- CORS open + no rate limiting

**Decision:** `CORSMiddleware(allow_origins=["*"])` (`app.py`); no rate limiting on any endpoint, including the expensive computer-dispatch path.

**Why unsafe:** Any origin can call the API; a caller can spam expensive operations (sandbox spins, model calls) unthrottled.

**Why accepted:** Dev convenience; single-tenant scale.

**Blast radius:** Resource exhaustion / cost; broad CSRF surface (mitigated somewhat by Bearer-in-header, not cookies).

**Fix-by:** Before public exposure -- restrict origins to known frontends; add rate limits, especially on `/computer/*` dispatch.

---

## U9 -- Global system prompt is shared-tools-only (per-user tools not advertised in prose)

**Decision:** The shared `_system_prompt` / `_available_tools` (built globally in `app.py`) now lists only shared (no-auth) tools. Per-user tools are no longer named in the global prompt.

**Why it exists:** `list_all_tools()` used to union *all* users' connected tools into the one global prompt -- a cross-user visibility leak and a race (last request's user leaked into everyone's prompt). The fix (Task 2) scopes per-user tools to the calling user, so they no longer feed the *global* prompt.

**Why accepted:** Per-user tools are still fully callable -- the per-user *tool enum* (`create_tool_option_class` / `state_tool`) is user-scoped and includes them, and unconnected services are advertised via the deferred-services list. Only the *prose advertisement* in buddy's shared system prompt is dropped.

**Blast radius:** None security-wise (this entry documents a fix that *removed* a leak). Minor UX: buddy's system prompt won't name a user's connected tools, though it can still call them.

**Fix-by:** When desired, build the system prompt *per-user* in `_make_agent` using `list_all_tools(user_id)` instead of a single global prompt. Bigger change (per-user prompts); deferred.

---

## U10 -- Sandbox state is lost on hard timeout (no idle-pause lifecycle)

**Decision:** The per-user e2b sandbox has a short active timeout (`computer_agent.sandbox.timeout_seconds`, currently 300s). If it sits idle past that **without being paused**, e2b reaps it; the next access transparently creates a **fresh, empty** sandbox (`SandboxManager.get_or_create`, hardened in commit `bf88386`).

**Why unsafe:** Silent data loss. A user's files/work in the sandbox can disappear after ~5 min of inactivity. The recovery path stops the 500s but does not restore the lost filesystem.

**Why accepted:** The common path is covered -- the runner calls `sandbox.pause()` after each task, and `get_or_create` now refreshes the idle timer (`set_timeout`) on every op so an *active* session stays alive. The gap is a sandbox kept alive only by browsing, or left active and then abandoned. `idle_timeout_seconds: 900` ("pause after idle") is noted in `config.yaml` but **not yet implemented**.

**Blast radius:** Durability only (no security boundary crossed) -- per-user loss of un-paused sandbox contents.

**Fix-by:** Implement the idle->pause lifecycle (pause on idle instead of letting e2b hard-reap), and/or move to e2b persistent volumes when available. Until then, treat the sandbox as scratch space that only survives across a *paused* gap.

---

# How to use this doc

- **Before any deployment that is multi-user or internet-facing**, walk this list. The hard blockers are **U2** (turn `ARK_DEMO_MODE` off) and **U1** (sign the OAuth callback). U5/U7/U8 are required before *public* prod.
- When you fix one, strike it through with a note and the commit, don't delete it -- the history of accepted risk is useful.
- When you add a new shortcut that trades safety for speed, **add it here in the same PR.** A shortcut that isn't written down is the dangerous kind.
