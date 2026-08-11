# ARKOS — Coding Guidelines

> **Start at `docs/single_loop_redesign_spec.md`** — it routes to everything else.
> `docs/contracts.md` is law: the spec says what to build, contracts says whether
> it is correct. If they disagree, contracts wins.
>
> **Never read `docs/deprecated/`.** It is the pre-redesign architecture (state
> graphs, mem0, ComputerAgent) kept only so old commits resolve. Do not open it,
> cite it, or follow its task numbers. Stale pointers to it in docstrings should
> be deleted, not followed. The current law is `docs/contracts.md`,
> `docs/decisions.md`, `docs/decision_tables.md`, `docs/schema.md`.


This file is read automatically by Claude Code. When using ChatGPT or another AI
assistant, paste this file as your first message before asking for any code help.

---

## Project at a Glance

ARKOS is a Python async agent framework. FastAPI backend, PostgreSQL + mem0 memory,
MCP tool compatibility, YAML-based state graphs with Pydantic-enforced transitions.

| Module | What it owns |
|---|---|
| `agent_module/` | Agent loop, step(), step_stream(), choose_transition() |
| `state_module/` | State base class, StateHandler, per-agent state packages and graph YAMLs |
| `tool_module/` | MCP client, tool registry, per-user auth (token_store) |
| `base_module/` | FastAPI app, API endpoints, auth router, task runner |
| `config_module/` | YAML config loader |
| `memory_module/` | Short-term + long-term memory (mem0 + PostgreSQL) |
| `model_module/` | LLM client wrapper (ArkModelLink) |
| `logging_module/` | Structured JSON lines logger (LogEvent, emit_log) |

**Stack:** Python 3.11+, FastAPI, PostgreSQL, Pydantic v2, asyncio, ruff, pytest.

---

## Architecture Contracts

These are permanent decisions for this codebase. Violating them silently breaks the
agent loop. Do not work around them — if a rule seems wrong for your use case, raise
it in review before merging.

### 1. State outputs are typed — never raw messages

Every `state.run()` returns a `StateOutput`. Never return a raw `AIMessage`.

```python
# WRONG
async def run(self, context, agent) -> AIMessage:
    return AIMessage(content="done")

# RIGHT
async def run(self, context, agent) -> StateOutput:
    return StateOutput(
        content="done",
        completion_signal="complete",
        structured_data={"route": "signal_name"},
    )
```

`completion_signal` must be one of: `complete`, `incomplete`, `error`, `needs_input`.
`structured_data` must include a `"route"` key when the state has a registered router.
The route value is a signal string (e.g. `"tool"`, `"ask"`, `"done"`, `"continue"`) —
never a hard-coded next state name.

### 2. Transition gates are deterministic — never LLM calls

`check_transition_ready()` reads `completion_signal` only. No LLM call, no prompt,
no interpretation. A prompt-based gate hallucinates under context pressure.

```python
# WRONG
def check_transition_ready(self, context):
    return call_llm("is this state finished?")

# RIGHT
def check_transition_ready(self, context) -> bool:
    return True  # or check last output signal deterministically
```

### 3. Agents are scoped to folders — states are auto-discovered

Each agent has its own package under `state_module/` with the `agent_` prefix:

```
state_module/
  core/               # shared infrastructure (State, StateHandler, StateOutput, ...)
  agent_buddy/        # chat agent
    __init__.py       # empty — discovery is automatic
    graph.yaml
    routers.py
    state_ai.py
    state_tool.py
    state_user.py
    state_workshop_plan.py
  agent_executor/     # subagent (background task runner)
    __init__.py       # empty — discovery is automatic
    graph.yaml
    routers.py
    state_executor.py
    state_tool.py
    state_approval.py
    state_executor_done.py
```

`StateHandler` scans one level below the given `agent_pkg` with `pkgutil.iter_modules`
at construction time. Any class that inherits from `State`, has a `type` attribute,
and is defined in that module (not just imported) is registered automatically.
No `@register_state` decorator. No manual import list in `__init__.py`.

The `obj.__module__ == mod.__name__` guard is critical: it prevents an imported class
from a sibling module being registered twice or in the wrong agent's handler.

### 4. StateHandler requires an agent package

`StateHandler` now takes a required `agent_pkg` positional argument. Always pass the
dotted Python package string for the agent being constructed.

```python
# WRONG
flow = StateHandler(yaml_path)

# RIGHT
flow = StateHandler(
    yaml_path="state_module/agent_buddy/graph.yaml",
    agent_pkg="state_module.agent_buddy",
    routers=BUDDY_ROUTERS,
)
```

Each agent gets its own `StateHandler` instance with its own discovered type map.
Two agents can define states with the same type string without conflict.

### 5. Routing uses signal strings — never next-state names in structured_data

States emit a route signal. A router function (registered in `routers.py`) translates
that signal to a concrete next-state name. The state itself never hardcodes the name
of the next state.

```python
# WRONG — state hardcodes the next state name
structured_data={"next_state": "agent_reply"}

# RIGHT — state emits a signal, router resolves it
structured_data={"route": "plan"}
```

Router functions live in `state_module/agent_{name}/routers.py` and are collected
in a `ROUTERS` dict keyed by state name as declared in `graph.yaml`.

```python
# routers.py
def agent_reply_router(output: StateOutput) -> str:
    route = (output.structured_data or {}).get("route", "")
    if route == "plan":
        return "workshop_plan"
    return "ask_user"

ROUTERS: dict[str, Callable] = {
    "agent_reply": agent_reply_router,
}
```

If a state has no router, `StateHandler` falls back to an LLM transition call
(multi-choice) or the single declared transition if there is only one.

### 6. Per-user tool auth follows one pattern

If a tool needs per-user credentials, add it to `PER_USER_SERVICES` in
`tool_module/tool_call.py` and follow the existing pattern (Google Calendar is
the reference implementation). Do not write a separate auth system.

### 7. Structured logging only — no print()

Use `emit_log()` from `logging_module`. `print()` is for local debugging only
and must be removed before opening a PR.

---

## Adding a New State to an Agent

1. Create `state_module/agent_{agentname}/state_{name}.py`
2. Inherit from `State`, set `type = "{name}"` as a class attribute
3. Implement `run()` — returns `StateOutput` with a `"route"` key in `structured_data`
4. Implement `check_transition_ready()` — deterministic, no LLM
5. Do NOT add `@register_state` and do NOT edit `__init__.py` — discovery is automatic
6. Add the state to the agent's `graph.yaml`:

```yaml
states:
  my_state:
    description: "one sentence — what this state does and does not do"
    type: my_state          # must match the class's `type` attribute exactly
    transition:
      next: [next_state_name]
```

7. If this state needs to drive routing, add a router in `routers.py` and register it
   in `ROUTERS` under the state's name from `graph.yaml`
8. Add tests in `tests/test_state_module.py`

Example minimal state:

```python
# state_module/agent_buddy/state_my_feature.py

from state_module.core.base_state import StateOutput
from state_module.core.state import State
from state_module.core.state_registry import register_state


class StateMyFeature(State):
    type = "my_feature"

    def __init__(self, name: str, config: dict) -> None:
        super().__init__(name, config)
        self.is_terminal = False

    def check_transition_ready(self, context) -> bool:
        return True

    async def run(self, context, agent=None) -> StateOutput:
        # ... do work ...
        return StateOutput(
            content="result text",
            completion_signal="complete",
            structured_data={"route": "continue"},
        )
```

---

## Adding a New Tool (MCP)

1. Add server config to `tool_module/mcp_config.json`
2. If the tool needs per-user auth, add to `PER_USER_SERVICES` in `tool_call.py`
3. Follow `token_store.py` for credential storage — do not invent a new pattern
4. Add tests in `tests/test_tool_module.py`, mocking the MCP transport

---

## Code Style

Formatter: **ruff**. Run `ruff check .` before every commit. The CI rejects failures.

### Naming

| Thing | Convention | Example |
|---|---|---|
| Files | `snake_case.py` | `state_tool_call.py` |
| Classes | `PascalCase` | `ToolCallState` |
| Functions / methods | `snake_case` | `check_transition_ready` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_RETRIES` |
| State files | `state_{type}.py` | `state_planning.py` |
| State classes | `State{Type}` | `StatePlanning` |
| Agent packages | `agent_{name}/` | `agent_buddy/`, `agent_executor/` |
| Test files | `test_{module_name}.py` | `test_agent_module.py` |

### Type hints

Required on every function signature. No exceptions.

```python
# WRONG
async def run(self, context, agent):

# RIGHT
async def run(self, context: list, agent: Agent) -> StateOutput:
```

Use `from collections.abc import Callable` — not `from typing import Callable` (ruff UP035).

### Async

Every function that awaits must be `async def`. Never call `asyncio.run()` inside
application code — only at entrypoints. Never use `time.sleep()` in async code,
use `await asyncio.sleep()`.

### Pydantic

Use Pydantic models for structured data that crosses module boundaries.
Parse LLM output with `model_validate_json()`, not `json.loads()` + manual dict access.
Handle parse failures explicitly — never let a `ValidationError` propagate uncaught
to the agent loop.

---

## Comments

Comment the **why**, not the **what**. If the code is self-explanatory, skip the comment.

**Required:**

- Module docstring at the top of every new file — 3 lines max covering: what it does,
  its key constraint, and what it explicitly does NOT do
- Docstring on every public method where params, return value, or exceptions are non-obvious
- Inline comment on every LLM prompt string explaining what behavior it enforces

**Skip:**

- Simple variable assignments and obvious loops
- Getters, setters, and anything the function name already says clearly

**Never:**

- Commented-out code in a PR — delete it or keep it in a branch
- `# TODO` in a PR — file a Linear issue instead

### Docstring format

```python
async def fetch_items(user_id: str, limit: int = 20) -> list[dict]:
    """
    Retrieve items for a user from the external service.

    Args:
        user_id: Used to load credentials from user_credentials table.
        limit: Max items to return. Keep low — large payloads exceed context window.

    Returns:
        List of dicts with keys: id, title, created_at.

    Raises:
        AuthRequiredError: If no valid token exists for this user.
    """
```

---

## Testing

**One test file per module.** Lives in `tests/`, named `test_{module_name}.py`.
Run with `pytest`. Async tests use `@pytest.mark.asyncio`.

### What to test

- The `completion_signal` value and required `structured_data` keys from `state.run()`
- The `"route"` signal in `structured_data` — this drives routing, test it explicitly
- Every API endpoint: happy path + one meaningful error case
- Every Pydantic model: invalid input raises `ValidationError`
- Auth flows: token missing, token expired, token valid
- Any function with branching logic: one test per branch

### What not to test

- LLM output quality — non-deterministic, always mock the LLM call
- Private methods (`_prefixed`) — test through the public interface
- Third-party library internals (Pydantic, FastAPI routing, MCP transport)
- Live external APIs — mock them without exception

### StateHandler in tests

Pass `agent_pkg` whenever constructing `StateHandler`:

```python
_BUDDY_PKG = "state_module.agent_buddy"

handler = StateHandler(
    yaml_path="state_module/agent_buddy/graph.yaml",
    agent_pkg=_BUDDY_PKG,
)
```

### Test structure

```python
# tests/test_state_module.py

import pytest
from unittest.mock import AsyncMock, MagicMock
from state_module.agent_buddy.state_my_feature import StateMyFeature
from state_module.core.base_state import StateOutput


class TestStateMyFeature:
    @pytest.fixture
    def state(self):
        config = {"transition": {"next": ["next_state"]}}
        return StateMyFeature("my_feature", config)

    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.call_llm = AsyncMock(return_value=MagicMock(
            content='{"key": "value"}'
        ))
        return agent

    @pytest.mark.asyncio
    async def test_returns_complete_on_success(self, state, mock_agent):
        result = await state.run(context=[], agent=mock_agent)
        assert isinstance(result, StateOutput)
        assert result.completion_signal == "complete"
        assert result.structured_data.get("route") == "continue"

    @pytest.mark.asyncio
    async def test_returns_error_on_llm_failure(self, state, mock_agent):
        mock_agent.call_llm = AsyncMock(side_effect=Exception("timeout"))
        result = await state.run(context=[], agent=mock_agent)
        assert result.completion_signal == "error"
        assert result.error_detail is not None
```

**Max 5 tests per state or function.** If you need more, the function does too much — split it.

---

## PR Conventions

### Title

```
[module] verb: short description
```

Keep the description under 60 characters. Module is the primary module being changed.

| Verb | Use when |
|---|---|
| `add` | New file, new feature, new capability |
| `fix` | Correcting broken behavior |
| `refactor` | No behavior change, code quality improvement |
| `remove` | Deleting dead code or a deprecated feature |
| `update` | Changing existing behavior intentionally |
| `test` | Tests only, zero production code change |

```
[agent] fix: replace prompt-based transition gate with signal check
[state] add: planning state as new initial state in buddy graph
[tool] add: per-user OAuth via token_store for new MCP service
[config] update: add max_iterations field to all state YAML configs
[logging] add: structured LogEvent schema and JSON lines writer
[base] fix: task queue status not updating on needs_input signal
```

### Description

```markdown
## What
One sentence.

## Why
One sentence. Link the Linear issue: LINEAR-123

## How
- Key decision or approach (max 4 bullets)
- If you need more than 4, the PR is too large

## Test
What you ran to verify this works.

## Checklist
- [ ] ruff check passes with zero warnings
- [ ] tests/test_{module}.py updated or confirmed unchanged
- [ ] No hardcoded credentials or .env files committed
- [ ] No print() statements left in production paths
- [ ] If adding a state: added to agent's graph.yaml; router added if needed
- [ ] If adding a tool: follows token_store.py auth pattern
```

### Size and scope

- Target under 300 lines changed. Hard limit 500 — split if over.
- One concern per PR. A refactor and a feature are two PRs.
- Do not mix agent loop changes with state logic changes in the same PR.

---

## Hard No's

These are rejected in review without discussion.

```python
# Returning raw AIMessage from state.run()
return AIMessage(content="done")

# LLM call inside check_transition_ready()
def check_transition_ready(self, context):
    return self.llm.call("are we done?")

# Hardcoding next-state name in structured_data (use route signals instead)
structured_data={"next_state": "agent_reply"}

# Calling StateHandler without agent_pkg
flow = StateHandler(yaml_path)

# Manually importing states in __init__.py (discovery is automatic)
from state_module.agent_buddy.state_ai import StateAI  # in __init__.py

# Bare except — always catch a specific exception
try:
    result = await do_thing()
except:
    pass

# Hardcoded credentials
API_TOKEN = "sk-abc123..."

# Blocking sleep in async code
time.sleep(2)

# Commented-out code in a PR
# old_result = await deprecated_call()

# print() in production paths
print(f"[debug] state ran: {self.name}")

# Functions over 40 lines — extract the logic
# More than 3 levels of nesting — extract to a function
```

---

## Quick Reference — Paste Into Any AI Assistant

Use this block when starting a session with ChatGPT, Gemini, or any tool that
doesn't auto-read this file:

```
This is ARKOS, a Python async agent framework. Rules before writing any code:

1. state.run() returns StateOutput — never AIMessage.
   completion_signal: complete | incomplete | error | needs_input
   structured_data must include {"route": "<signal>"} when a router is registered.

2. check_transition_ready() reads completion_signal only — never calls the LLM.

3. Agents are scoped to state_module/agent_{name}/ folders. StateHandler auto-discovers
   all State subclasses one level below the given agent_pkg via pkgutil.iter_modules.
   No @register_state decorator. No manual imports in __init__.py.

4. StateHandler signature: StateHandler(yaml_path, agent_pkg="state_module.agent_buddy", routers=ROUTERS)
   agent_pkg is required. Each agent has its own isolated type map.

5. Routing uses signal strings in structured_data["route"], not next-state names.
   Router functions in routers.py translate signals to state names.
   ROUTERS dict is keyed by state name as declared in graph.yaml.

6. Per-user tool auth follows token_store.py — do not create a new pattern.

7. Tests: one file per module in tests/, max 5 per function, mock all external
   APIs and all LLM calls. Pass agent_pkg to StateHandler in tests.

8. PR titles: [module] verb: description (under 60 chars). ruff must pass.

9. Type hints on every function. async def for anything that awaits.
   Use collections.abc.Callable, not typing.Callable (ruff UP035).

10. Comment the why, not the what. No commented-out code. No print() in PRs.
```
