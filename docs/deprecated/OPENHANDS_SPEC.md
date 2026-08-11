# Feature Spec: ARKOS x OpenHands -- Delegated Code Execution from Mobile/Web

**Sources**

- OpenHands (formerly OpenDevin) -- self-hostable coding agent: REST API, web UI, Docker-container-per-conversation isolation (https://github.com/All-Hands-AI/OpenHands)
- `MULTIUSER_SPEC.md` -- Task 1 (user-scope tool registry) and Task 2 (verify identity) are imported here as immediate prerequisites
- Current code: `base_module/tasks.py` + `task_runner.py` + `task_store.py` (existing async background-task + polling + approval pattern to reuse), `state_module/agent_executor/state_approval.py` (human-in-the-loop gate), `tool_module/smithery.py` (per-user OAuth tokens, global `_tool_registry`), `base_module/app.py` (header-trust identity), `base_module/jwt_utils.py` (verify helpers, not yet wired)
- Companion specs: `MULTIUSER_SPEC.md`, `HARNESS_SPEC.md`, `ENVIRONMENT_SPEC.md` (the workspace problem this supersedes)

**Status:** Not started | **Author:** | **Last updated:** 2026-06-02

---

# Problem

The dev process is backwards. To workshop a spec, write a feature, or open a PR, a developer has to be at a keyboard with a terminal. The goal: **talk to buddy from a phone over the web -- "implement logging spec task 1" -- and have it produce a PR**, the way OpenHands / OpenDevin does, without reinventing a coding agent or a sandbox.

Two wrong turns to avoid:

1. **Claude Code has no API.** It is a local CLI; its "API" is the terminal. Spawning it as a subprocess and parsing output is fragile and surrenders control of the agent loop -- which is the thing ARKOS exists to own. Dead end.
2. **Building our own sandboxed coding executor.** The `agent_executor` path is a simple in-process task runner with no filesystem isolation. Extending it into a multi-user, file-writing, git-pushing coding agent means rebuilding container isolation, credential injection, and resource limits -- exactly what OpenHands already solved.

The right shape: **ARKOS/buddy is the conversation and planning layer; OpenHands is the execution and isolation layer.** Buddy workshops the spec with you, then delegates the actual implementation to OpenHands as a tool call. OpenHands runs the task in an isolated Docker container, opens a PR, and returns the link. Buddy reports it back.

The blocker: this puts **container creation and per-user GitHub credential injection behind a web-facing chat**. Today any client can set `X-User-ID: <victim>` (header-trust identity, `app.py:238` et al.) and any tool call can route to the wrong user's server (global `_tool_registry`, `smithery.py:257`). Exposing "create a container with this user's GitHub token and push to their repo" on top of unverified identity is a credential-leak waiting to happen. So a minimal slice of `MULTIUSER_SPEC` must land first, and it is included in this spec as Phase 0.

**Success looks like:** From a phone, an authenticated user tells buddy to implement a task; buddy delegates to OpenHands; OpenHands runs it in that user's isolated container with that user's credentials; a PR appears; buddy returns the link. User B's parallel task runs in a separate container with separate credentials and never touches User A's files, branch, or tokens.

---

# Technical Background

**What OpenHands solves (config, not code):**

OpenHands creates a fresh Docker container per conversation. That gives, at the kernel level:

| Isolation | Mechanism | Our work |
|---|---|---|
| Filesystem (A can't read B's files) | Separate container per conversation | none -- config |
| Process (A's run can't touch B's) | Separate container | none -- config |
| Resource (A can't starve B) | Docker `mem_limit` / `cpu_quota` | set limits in OpenHands config |
| Branch conflicts | One git branch per task | enforce branch naming on task creation |

So the **hard isolation is done by Docker.** What remains is not a sandboxing problem; it is a *coordination* problem: identity, credential routing, ownership tracking, and result polling.

**What ARKOS still owns:**

| Concern | Who | Our work |
|---|---|---|
| Verified identity | ARKOS | wire `jwt_utils` as a dependency (Phase 0) |
| Credential injection | ARKOS | retrieve user's GitHub token, pass at conversation creation |
| Conversation ownership | ARKOS | `opencode_tasks` table mapping user_id -> OpenHands conversation |
| Result polling / approval | ARKOS | reuse `task_runner` + `state_approval` pattern |
| Resource limits | OpenHands | deployment config |

**The credential wrinkle (decided).** Smithery holds OAuth tokens *write-only* and proxies tool calls -- it never hands ARKOS a raw token. But OpenHands runs its own container and needs a *usable* GitHub token to `git push` and `gh pr create`. So the existing Smithery GitHub MCP is **not** sufficient to credential OpenHands. **Decision: the user supplies a GitHub fine-grained PAT, collected through the same connections panel where they connect Linear and Google Calendar** -- but as a "bring your own token" entry (a paste-a-token field), not a Smithery OAuth redirect. ARKOS stores it per-user (encrypted, Open Question 3) and injects it into that user's OpenHands container only. The connections panel thus needs to support two entry types: Smithery OAuth services (existing) and BYO-token services (new, for GitHub).

**The model (decided).** OpenHands runs a **frontier model via the Claude API (Sonnet 4.6), not ARKOS's local Qwen.** Coding is a free-form, long-horizon, native-tool-calling loop that a 7B model cannot do -- it would wander and open broken PRs. This is the capability-dial split in practice: buddy plans on the cheap local model; OpenHands executes on Claude. OpenHands' model config lives in `config.yaml: openhands.llm` (model + Anthropic key), separate from `llm.*`. Cost note: coding tasks are token-heavy and billed per token -- see Open Question 5.

**Existing ARKOS patterns to reuse (do not rebuild):**
- `base_module/task_runner.py` -- async background task with status tracking. OpenHands tasks are async; this is the polling skeleton.
- `task_events` / `log_event` -- progress log. OpenHands task progress maps onto this.
- `state_module/agent_executor/state_approval.py` -- human-in-the-loop gate. "Buddy wants to delegate this to OpenHands -- approve?" reuses it.
- `tasks` table schema (`task_id`, `user_id`, `status`, `session_id`, `agent_kind`, `parent_task_id`) -- `opencode_tasks` mirrors it.

**Architectural choice -- Option A (chosen).** ARKOS calls OpenHands as a tool; the executor stays for simple in-process tasks (no Docker overhead), OpenHands handles real coding work. The alternative (Option B: OpenHands replaces the executor entirely) is cleaner long-term but a bigger commitment; defer that decision until Option A is in use (Open Question 4).

---

# Proposed Approach

Two phases. Phase 0 is the minimal `MULTIUSER_SPEC` slice that must land before any web-facing delegation; Phase 1 is the integration.

**Phase 0 -- Identity and routing (blockers, imported from MULTIUSER_SPEC):**
1. Verify identity at the edge -- a `CurrentUser` JWT dependency replaces header-trust on every per-user endpoint.
2. User-scope the tool registry -- `_tool_registry[user_id][tool]` so a tool call can never route to another user's server.

**Phase 1 -- OpenHands integration:**
3. `opencode_tasks` table + ownership tracking.
4. An `opencode_task` tool buddy can call, which creates an OpenHands conversation with the calling user's GitHub credential injected.
5. Result polling that reuses `task_runner` + `state_approval` and reports the PR link back to buddy.
6. OpenHands deployment with per-container resource limits (ops/config).

What stays the same: buddy's conversation/planning loop, the executor for in-process tasks, the existing task/approval tables.

Explicitly **not in scope:**
- The rest of `MULTIUSER_SPEC` (Task 3 shared-state races, Task 4 secrets fail-fast) -- important but not blockers for a dev-server rollout; run alongside.
- Deprecating `agent_executor` in favor of OpenHands (Option B) -- Open Question 4.
- A custom sandbox / the full `ENVIRONMENT_SPEC` workspace -- OpenHands' Docker model supersedes it for this workflow.
- Mobile web UI work in `arkos-webui` -- tracked separately; this spec is the backend that it talks to.

---

# Implementation Plan

## Task 1 (Phase 0, from MULTIUSER_SPEC Task 2): Verify identity at the edge

**Problem:** Per-user endpoints derive identity from `X-User-ID` (`app.py:238,299,344,371,483`) with no verification. Any client can impersonate any user -- unacceptable once a request can inject that user's GitHub token into a container.

**Done when:**
- A `CurrentUser` FastAPI dependency validates a Bearer token via `jwt_utils` and returns the verified user id; all per-user endpoints depend on it instead of reading the header.
- `fallback_user_id` is honored only when `ARK_DEMO_MODE` is explicitly set; otherwise a missing/invalid token returns `401`.
- The verified id -- not a header -- flows into `Memory`, `call_tool`, the task tables, and (Task 4) the OpenHands credential injection.

**Touch point:** `base_module/app.py`, `base_module/jwt_utils.py`, new `CurrentUser` dependency.

**Priority:** P0 (blocker) | **Effort:** ~1-2 days | **Blockers:** none

**Out of scope:** Token issuance UX, refresh tokens, roles (owner-level isolation only).

**Acceptance test:** `test_missing_token_rejected_outside_demo`, `test_header_cannot_override_verified_identity`.

---

## Task 2 (Phase 0, from MULTIUSER_SPEC Task 1): User-scope the tool registry

**Problem:** A global `_tool_registry` (`smithery.py:257`) maps tool name -> server with no user dimension; the `opencode_task` tool (Task 4) must resolve to the calling user's context, and colliding tool names across users currently cross-route.

**Done when:**
- `_tool_registry` becomes `dict[user_id, dict[tool_name, server_name]]` (mirrors `_user_tools`); writes (`:319,:373`) and reads (`:434,:457`) are user-scoped.
- `call_tool` resolves the server from the calling user's registry only.
- The `_pending` pop no-op (`:375`) is fixed in the same pass.

**Touch point:** `tool_module/smithery.py`.

**Priority:** P0 (blocker) | **Effort:** ~1 day | **Blockers:** none

**Out of scope:** Tool-list cache TTL.

**Acceptance test:** `test_tool_registry_is_user_scoped`, `test_colliding_tool_names_do_not_cross_users`.

---

## Task 3: opencode_tasks table + ownership tracking

**Problem:** OpenHands has no concept of users -- it has conversations. ARKOS must record which conversation belongs to which user so results are scoped and queryable.

**Done when:**
- New migration `db/migrations/0006_opencode_tasks.sql`:

```sql
CREATE TABLE IF NOT EXISTS opencode_tasks (
    task_id     UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     TEXT         NOT NULL,
    oh_conv_id  TEXT,                 -- OpenHands conversation id (null until created)
    repo        TEXT         NOT NULL,
    branch      TEXT         NOT NULL,
    prompt      TEXT         NOT NULL,
    status      TEXT         NOT NULL DEFAULT 'pending',  -- pending|running|awaiting_approval|completed|failed
    pr_url      TEXT,
    error       TEXT,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_opencode_user ON opencode_tasks (user_id, created_at DESC);
```

- Helpers in a new `base_module/opencode_store.py` (mirroring `task_store.py`): `create_opencode_task`, `set_opencode_status`, `get_opencode_task`, `list_opencode_tasks(user_id)`.
- All reads are scoped by `user_id` -- a user can only see their own tasks, enforced in the query (User B cannot fetch User A's task even with a known task_id).

**Touch point:** new `base_module/opencode_store.py`, `db/migrations/0006_opencode_tasks.sql`.

**Priority:** P1 | **Effort:** ~1 day | **Blockers:** Task 1

**Acceptance test:** `test_opencode_tasks_scoped_to_user`.

---

## Task 4: opencode_task tool + credential injection

**Problem:** Buddy needs a tool that delegates a task to OpenHands, running in the calling user's isolated container with the calling user's GitHub credential.

**Done when:**
- A new per-user PAT store holds GitHub fine-grained tokens, populated via the connections panel (Task 4b). `get_user_github_token(user_id)` reads it; tokens are encrypted at rest (Open Question 3).
- An `opencode_task(prompt: str, repo: str)` tool is available to buddy (in-process tool, not Smithery -- it needs the verified `user_id` and the raw token, neither of which Smithery carries).
- The tool: (a) derives a branch `task/{short_task_id}-{slug}`, (b) creates an `opencode_tasks` row, (c) retrieves the user's GitHub PAT, (d) calls the OpenHands REST API to create a conversation with `{repo, branch, prompt, llm: <openhands.llm from config>, env: {GITHUB_TOKEN: <user token>}}`, (e) stores the returned `oh_conv_id`, sets status `running`, returns a task handle.
- The PAT is injected only into that user's conversation; never logged (add to `LOGGING_SPEC.md` redaction blocklist) and never reused across users.
- If the user has no GitHub PAT connected, the tool returns a connect-prompt pointing at the connections panel (mirrors `AuthRequiredError`), not a failure.

**Task 4b (connections panel BYO-token entry):** the connections UI/endpoints gain a "bring your own token" service type for GitHub: a paste-a-PAT field next to the Smithery OAuth services, a `POST /services/github/token` endpoint that stores it per-user (encrypted), and status display ("connected" / "needs token"). This is the same panel as Linear/GCal so the UX is consistent, but the mechanism is token-paste, not OAuth redirect.

**Sketch (illustrative, not final):**

```python
async def opencode_task(prompt: str, repo: str, *, user_id: str) -> dict:
    token = await get_user_github_token(user_id)        # Open Question 1
    if not token:
        return {"auth_required": True, "service": "github", "setup_url": ...}
    task = create_opencode_task(user_id=user_id, repo=repo,
                                branch=_branch_for(prompt), prompt=prompt)
    conv = await openhands.create_conversation(
        repo=repo, branch=task["branch"], task=prompt,
        env={"GITHUB_TOKEN": token},                    # injected into THIS user's container only
    )
    set_opencode_status(task["task_id"], "running", oh_conv_id=conv["id"])
    return {"task_id": task["task_id"], "status": "running"}
```

**Touch point:** new `tool_module/opencode_tool.py`, registration in the in-process tool path (`agent.py` / buddy graph), `base_module/opencode_store.py`.

**Priority:** P1 | **Effort:** ~2 days | **Blockers:** Task 1, Task 2, Task 3, Open Question 1 (token source)

**Out of scope:** Streaming OpenHands' intermediate steps back to buddy (just start + final result for v1).

**Acceptance test:** `test_opencode_task_injects_only_calling_users_token`, `test_no_github_credential_returns_connect_prompt`.

---

## Task 5: Result polling + approval flow

**Problem:** OpenHands tasks are async. Buddy must learn when a task finishes, surface the PR link, and gate destructive delegation behind the existing human approval.

**Done when:**
- A poller (reusing `task_runner`'s async pattern) checks OpenHands conversation status for `running` `opencode_tasks` and, on completion, extracts the PR URL, sets status `completed`, and records it. On failure it sets `failed` with the error.
- Buddy can report status: "task running", "PR #47 open: <url>", or "failed: <reason>".
- Delegation is gated by the existing `state_approval` pattern when configured -- buddy proposes "delegate this to OpenHands on repo X branch Y", the user approves/declines before the container is created (so a misunderstood request never spends compute or touches the repo).
- Progress is written to `task_events` / `audit_events` (coordinate with `LOGGING_SPEC.md`) so the session is auditable.

**Touch point:** `base_module/task_runner.py` (or a sibling `opencode_runner.py`), `state_module/agent_executor/state_approval.py` (reuse), `base_module/opencode_store.py`.

**Priority:** P1 | **Effort:** ~2 days | **Blockers:** Task 4

**Out of scope:** Webhooks (poll for v1; webhook is an optimization, Open Question 2).

**Acceptance test:** `test_completed_task_surfaces_pr_url`, `test_delegation_requires_approval_when_configured`.

---

## Task 6: OpenHands deployment with resource limits

**Problem:** OpenHands must be running and reachable from ARKOS, with per-container limits so one user's task cannot starve another's.

**Done when:**
- OpenHands runs as a service (docker-compose alongside ARKOS) reachable at a configured base URL (`config.yaml: openhands.base_url`).
- Per-container `mem_limit` and `cpu_quota` are set in OpenHands config so a heavy task is bounded.
- An API key / shared secret authenticates ARKOS -> OpenHands calls (ARKOS is the only client; end users never call OpenHands directly).
- A health check is wired into `/health` (the async one from `HARNESS_SPEC.md` Task 6).

**Touch point:** `docker-compose.yml`, `config_module/config.yaml` (`openhands` section), `base_module/app.py` (health check).

**Priority:** P1 | **Effort:** ~1 day (ops) | **Blockers:** none (parallel with Tasks 3-5)

**Out of scope:** Autoscaling, multi-host OpenHands. Single-host is fine at MVP scale.

**Acceptance test:** Manual: a delegated task runs end to end in a limited container and opens a PR.

---

# Tests

## Test 1: test_missing_token_rejected_outside_demo
**Verifies:** With `ARK_DEMO_MODE` unset, a request with no/invalid Bearer token returns 401 and never falls back to `fallback_user_id`.
**Why it matters:** The impersonation hole; nothing downstream is safe until identity is verified.

## Test 2: test_header_cannot_override_verified_identity
**Verifies:** A valid token for user A with `X-User-ID: B` operates as A.
**Why it matters:** Stops header-trust re-entering through a side door once tokens are in place -- the exact vector that would mis-route a GitHub token.

## Test 3: test_tool_registry_is_user_scoped / test_colliding_tool_names_do_not_cross_users
**Verifies:** Tool resolution stays within the calling user's namespace even when two users have the same tool name from different servers.
**Why it matters:** The `opencode_task` tool must resolve to the right user; cross-routing here would run a task with the wrong identity.

## Test 4: test_opencode_tasks_scoped_to_user
**Verifies:** `list_opencode_tasks(A)` returns only A's tasks; a request as B cannot fetch A's task even with a known task_id.
**Why it matters:** Task records reference repos and branches; cross-user read is an information leak.

## Test 5: test_opencode_task_injects_only_calling_users_token
**Verifies:** Creating a conversation for user A injects A's GitHub token; a concurrent task for B injects B's; neither container sees the other's token.
**Why it matters:** This is the core multi-user safety property -- a leaked token means another user can act on a victim's GitHub.

## Test 6: test_no_github_credential_returns_connect_prompt
**Verifies:** A user with no GitHub credential gets a connect-prompt, not a failure or a crash.
**Why it matters:** Graceful onboarding; mirrors the existing `AuthRequiredError` UX.

## Test 7: test_completed_task_surfaces_pr_url / test_delegation_requires_approval_when_configured
**Verifies:** A finished task yields the PR URL through buddy; a configured approval gate blocks container creation until the user approves.
**Why it matters:** The payoff (PR link back to the phone) and the safety rail (no surprise compute/repo writes).

---

# Open Questions

1. ~~**GitHub token source.**~~ **Resolved: user supplies a fine-grained PAT via the connections panel (Task 4b), stored encrypted per-user, injected into their container only.** Same panel as Linear/GCal but a paste-a-token entry, not OAuth. A dedicated ARKOS GitHub OAuth App (smoother UX, no manual PAT) remains a future upgrade but is not needed to ship.

2. **Polling vs webhook (Task 5).** Poll OpenHands for status (simple, slightly laggy, more load) or have OpenHands call back on completion (lower latency, needs a public callback URL and auth). *Leaning poll for v1; webhook as an optimization once it works.*

3. **Token storage + encryption.** If we store GitHub tokens (Open Question 1 option a/b), they must be encrypted at rest. This pulls in `MULTIUSER_SPEC` Open Question 4 (token-at-rest encryption), previously deemed out-of-scope. A web-facing PR-creating agent makes it in-scope. Decide the storage + encryption approach with Open Question 1.

4. **Option B -- replace the executor.** Once `opencode_task` is in use, is `agent_executor` + `task_runner` still worth keeping for in-process tasks, or should all execution route to OpenHands? *Defer until Option A is in real use; revisit with usage data.*

5. **OpenHands API surface.** Exact conversation-creation / status endpoints depend on the deployed OpenHands version and may differ from the sketch. Pin the version in `docker-compose.yml` and verify the API shape against it before Task 4. Treat the `openhands.create_conversation(...)` call as a thin adapter to isolate version drift.

6. **Cost per task (Claude API).** OpenHands runs on Claude Sonnet 4.6 (`openhands.llm`), billed per token. Coding tasks read many files and run many turns, so a single "implement this task" delegation can be a meaningful number of tokens. Estimate real per-task cost after the first few runs and decide whether to (a) cap turns/tokens per task, (b) gate delegation behind the approval flow always (not just when configured), or (c) offer Opus only for explicitly-flagged hard tasks. *Measure before optimizing.*

---

# Implementation Notes

*Add entries here as work lands.*

- (sequencing) Phase 0 (Tasks 1-2) MUST land before any web-facing delegation. They are the smallest tasks here -- Task 1 is wiring existing `jwt_utils` as a dependency; Task 2 is adding a user dimension to one dict. Do them this week, then Phase 1 is safe to build.
- (cross-link) Tasks 1-2 are the canonical scheduling of `MULTIUSER_SPEC` Tasks 2 and 1 respectively. Implement once; mark them done in `MULTIUSER_SPEC` when they land. `MULTIUSER_SPEC` Tasks 3-4 remain there and run in parallel, not blockers for this work.
- (security) The GitHub token must never reach a log. Add `github_token`, `token`, `oh_conv_env` to the `LOGGING_SPEC.md` args-redaction blocklist before Task 4.
- (reuse) Do not write a new async runner from scratch for Task 5 -- the `task_runner.py` spawn + status-poll + `log_event` skeleton already does 80% of it. Mirror it.
- (supersedes) This spec supersedes `ENVIRONMENT_SPEC.md` for the coding-from-phone workflow: OpenHands' Docker-per-conversation is the sandbox, so the custom workspace/grep environment is no longer the path for this use case. Keep `ENVIRONMENT_SPEC` only if an in-process workspace is still wanted for the lightweight executor.
