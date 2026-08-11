# ARKOS Code Review — Unclean Practices & Slop

**Reviewer pass:** 2026-06-15. Judged against general production / large-org
standards, **not** `CLAUDE.md` (which is itself stale — see F1). Findings are
ranked by impact. Line references verified against the working tree.

Severity key: **P0** = correctness/security/encapsulation risk; **P1** =
maintainability landmine; **P2** = cleanup/hygiene.

---

## P0 — Structural

### F1. `CLAUDE.md` is inaccurate — the "authoritative contract" lies about the code
The repo's contract doc describes a reality that does not exist:
- It lists a `logging_module/` ("LogEvent, emit_log, structured JSON lines") as a
  core module. **There is no `logging_module/`** in the tree, and `emit_log` /
  `LogEvent` are referenced **nowhere** in code. The actual logging is stdlib
  `logging.getLogger` (26 sites) plus stray `print()`.
- It states, repeatedly and in the "Hard No's": **"No `@register_state`
  decorator."** Every state file uses `@register_state` (see F2).
- It says agent `__init__.py` files are empty and discovery is automatic; the
  `state_registry.py` docstring says the opposite ("each agent package's
  `__init__.py` does this explicitly").

A contract doc that contradicts the code is worse than none — new contributors
(and AI assistants) follow it and write the wrong thing. Either regenerate it from
the code or delete it.

### F2. Dead, contradictory state-registration subsystem
There are **two** state-registration mechanisms, and one is entirely dead:
- `StateHandler._discover_states` uses `pkgutil.iter_modules` and explicitly
  "does not consult `STATE_REGISTRY`" (`state_handler.py:7`). This is the live one.
- `state_module/core/state_registry.py` defines a global `STATE_REGISTRY` and a
  `@register_state` decorator. **Every** state file imports and applies it
  (`state_ai`, `state_plan`, `state_tool`, `state_user`, `state_computer_plan`,
  and all four executor states). Nothing ever reads `STATE_REGISTRY`.

So the decorator is pure dead weight, and worse, it executes a **`print()` on every
import**:
```
state_registry.py:21:  print(f"[registry] registered state type: {state_type!r} -> {cls.__name__}")
```
Every process boot spews `[registry] registered ...` to stdout. Delete the registry,
the decorator, and the imports, or wire discovery to actually use it. Right now it's
a parallel subsystem that only adds noise and confusion.

### F3. Cross-module reach-ins to `MCPToolManager` internals (encapsulation breach)
14 call sites outside the tool module poke at private attributes/methods of the
shared `tool_manager`: `._user_tools`, `._pending`, `._shared_tools`,
`._user_tool_registry`, `._ensure_user_server`, `._user_conn_id`. Examples in
`base_module/app.py` (the chat lazy-connect loop, `_make_agent`, connect/disconnect)
and the task path. The private dicts *are* the API. Consequences:
- Any refactor of the manager's internals silently breaks callers.
- The same-user mutation race on `._user_tools` across `await`s (noted in the
  concurrency audit) lives precisely because the mutation happens outside the class.

Give `MCPToolManager` a public surface (`ensure_connected(user_id, service)`,
`tools_for(user_id)`, `connection_status(user_id)`) and make the underscored members
truly private.

---

## P1 — Maintainability landmines

### F4. `print()` in production paths
Stdlib `logging` is used in 26 places, but debug `print()`s remain in hot code:
- `agent_module/agent.py:367-369` — `print("[TIMING] add_context: ...")` and
  `print("agent.py received message")` fire on **every** agent step.
- `model_module/ArkModelNew.py:124-125` — `print(type(msg)); print(msg)` before
  raising on an unknown message type.
- `memory_module/memory.py:137,205,245` — error handling prints instead of logging.
- `state_registry.py:21` — see F2.

These bypass log levels/formatting/aggregation and pollute stdout in prod. Route
through `logging`.

### F5. Functions far too large to test or review
~20 functions exceed 45 lines; several are enormous:
| Lines | Location | Function |
|------:|----------|----------|
| 241 | `state_module/agent_executor/state_executor.py:96` | `run` |
| 128 | `base_module/task_runner.py:80` | `_run_task_inner` |
| 122 | `base_module/app.py:550` | `chat_completions` |
| 113 | `computer_module/agent.py:94` | `run` |
| 107 | `state_module/agent_executor/state_approval.py:58` | `run` |
| 104 | `state_module/agent_buddy/state_plan.py:49` | `run` |
| 89 | `agent_module/agent.py:348` | `step` |

A 241-line `run()` cannot be unit-tested in isolation or reviewed with confidence.
Extract the tool-listing, decision-parse, and routing blocks into named helpers.

### F6. Broad `except Exception` that swallows
Catch-all handlers are pervasive (`app.py` ×6, `computer_module/agent.py` ×5,
`memory.py` ×4, `computer_router.py` ×4, executor states ×7). Many log-and-continue
or return an empty default, hiding the real failure — the same "silent" theme the
ISSUES.md surfacing work is fighting. Catch the specific exception; let the unknown
ones propagate or fail loud.

### F7. Committed dead code and a misspelled directory
Shipped but unused:
- `frontend/frontend_old/` (3 files)
- `memory_module/depricated/` and `base_module/depricated/` — **note the typo,
  "depricated"** — including `OAI_Compat_Main.py` and stale JSON schemas.
- `computer_module/spike_sandbox.py` (a one-off spike, 60+ prints).
- `base_module/main_interface.py` + `main_interface_rich.py` (CLI front-ends, ~24
  prints, not wired to anything served).
- `arkos-webui` (per ISSUES.md, "not served — ignore/delete").

Dead code rots, confuses search, and inflates the attack/maintenance surface. Delete
it (git history keeps it).

### F8. Fake data shipped in the product UI
`frontend/seed.jsx` hardcodes a `WATCHING` list (linear/mail/calendar/github) shown
on "buddy's desk" as live sources with cadences ("every 5m", "live"). The comment
admits "watching has no backend table yet — keep it local so the zone isn't empty."
The desk presents mock monitoring as real. Either build the feature or hide the zone;
don't ship decorative fake state.

---

## P2 — Hygiene

### F9. Scattered in-function imports
31 in-function imports across `base_module`/`computer_module`; `app.py` alone imports
`json`, `urlencode`, `contextlib`, `asyncio`, `aiohttp`, `psycopg2`, `httpx`,
`HTMLResponse` inside function bodies, several aliased to dodge shadowing
(`import json as _json`, `import asyncio as _asyncio`). A few (heavy/optional deps)
are defensible; most are slop. Hoist to module top; reserve local imports for genuine
cycle/cost reasons and comment why.

### F10. Hardcoded default JWT secret
`base_module/jwt_utils.py:26` — `_DEFAULT_SECRET = "ark-dev-secret-change-me"`. It's
gated by `assert_secure_secret()` (refuses to boot with it outside demo mode), so not
an active leak, but a literal secret in source is a smell; move to env-only with no
in-source default.

### F11. Inconsistent typing
~386 functions lack a return type annotation, including FastAPI handlers in
`computer_router.py` and core helpers. Not every function needs one, but the public
API surface and core agent/memory functions should be uniformly typed. Pick a
standard and enforce it in CI (ruff/mypy), rather than the current ~50/50.

### F12. Manual migrations, no startup/CI gate
`db/migrate.py` is never invoked by app startup or compose (also in ISSUES.md #7). A
schema change ships without any guarantee it's applied; the failure mode is a 500
that the frontend hides. Run pending migrations on boot (or block boot on drift).

### F13. Brittle string-matching heuristics
`state_plan.py` filters "clarify-only" plans with a hand-rolled `_CLARIFY_TOKENS`
list ("ask the user", "inquire", "gather from the user", ...) and prefix-stripping.
This is fragile NLP-by-substring that will both over- and under-match. If the
guardrail matters, drive it off the model's structured output, not English keyword
soup.

---

## Suggested order of attack
1. F2 (delete dead registry + boot-time print) and F4 (kill stray prints) — fast,
   high signal-to-noise.
2. F7 (delete dead dirs) and F1 (fix or remove CLAUDE.md) — removes the misleading
   surface.
3. F3 (public tool-manager API) — closes the encapsulation/race surface.
4. F5/F6 (split giant functions, narrow excepts) — the real maintainability debt.
5. F8–F13 — hygiene, bundle into a cleanup PR.
