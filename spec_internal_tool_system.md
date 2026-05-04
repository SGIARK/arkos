# spec_internal_tool_system

**Status:** Draft  
**Relates to:** `spec_web_search`, `tool_module/`, `state_module/`

---

## Problem

All tools in ArkOS currently share one execution path:

```
agent_reply -> workshop_plan -> (user approval on dashboard) -> use_tool
```

This is the right flow for consequential user-facing actions -- creating a Linear ticket,
sending a calendar invite, etc. But some tools exist purely to help the agent reason
better. Web search, for instance, should fire silently mid-turn and feed its result
back into the agent's reply. Forcing it through the approval flow would be disruptive
and confusing to the user.

There is also a secondary issue: the approval flow assumes the tool was chosen by a
plan drafted by the LLM. Internal tools need to be invocable by `StateAI` directly,
without going through `workshop_plan` or `StateExecutor`.

---

## Goal

Introduce a first-class concept of **internal tools**: Smithery-backed tools that are:

1. Available system-wide (no per-user OAuth).
2. Invocable inline by `StateAI` without an approval step.
3. Invisible to the user as a separate "action" -- their results are absorbed into the
   agent's next reply.
4. Still routed through `SmitheryManager` so there is one tool invocation path in the
   codebase.

---

## Non-goals

- Creating a local tool execution path that bypasses Smithery entirely.
- Giving users the ability to define their own internal tools at runtime.
- Implementing "skills" (predefined multi-step task templates) -- that is deferred to
  `spec_skills`.
- Changing how user-authenticated tools (Linear, Google Calendar, etc.) work.

---

## Design

### 1. Configuration

Internal tools are marked in `config.yaml` with `internal: true`. They must also have
`requires_auth: false` (internal implies no per-user auth). The `internal` flag is the
only new config key.

```yaml
mcp_servers:
  brave-search:
    mcp_url: "https://brave.run.tools"
    requires_auth: false
    internal: true
    name: "Brave Search"

  linear:
    mcp_url: "https://linear.run.tools"
    requires_auth: true
    internal: false          # default, can be omitted
    name: "Linear"
```

`SmitheryManager` already handles `requires_auth: false` servers correctly -- they are
initialized as shared connections at startup via `initialize_servers()`. No changes
to the connection or auth logic are needed.

### 2. SmitheryManager additions

Add two lightweight helpers:

```python
def _is_internal(self, server_name: str) -> bool:
    return bool(self.servers.get(server_name, {}).get("internal"))

def list_internal_tools(self) -> dict[str, dict[str, Any]]:
    """Returns the tool spec map for internal (system-invokable) servers only."""
    return {
        server: tools
        for server, tools in self._shared_tools.items()
        if self._is_internal(server)
    }
```

`call_tool` does not need to change. Internal tools are already reachable via the
shared connection path.

### 3. New state: StateInternalTool

A new state class parallel to `StateTool` but without the approval dependency.
It is entered directly from `StateAI` and exits back to `agent_reply`.

```python
# state_module/state_internal_tool.py

@register_state
class StateInternalTool(State):
    type = "internal_tool"

    async def run(self, context, agent=None):
        pending = getattr(agent, "pending_internal_tool", None)
        if not pending or not pending.get("tool_name"):
            return StateOutput(
                content="",
                completion_signal="error",
                error_detail="StateInternalTool entered without a pending tool",
                structured_data={"next_state": "agent_reply"},
            )

        tool_name = pending["tool_name"]
        tool_args = pending.get("tool_args", {})
        agent.pending_internal_tool = None

        try:
            result = await agent.tool_manager.call_tool(
                tool_name=tool_name,
                arguments=tool_args,
                user_id=None,          # internal tools never need a user_id
            )
        except Exception as e:
            return StateOutput(
                content=f"[internal tool error: {e}]",
                completion_signal="complete",
                structured_data={"next_state": "agent_reply", "tool_result": None},
            )

        return StateOutput(
            content=str(result),
            completion_signal="complete",
            structured_data={"next_state": "agent_reply", "tool_result": result},
        )
```

### 4. StateAI routing changes

`StateAI` gains a fourth route: `internal_tool`. When the LLM decides it needs to call
an internal tool before it can answer, it returns `route = "internal_tool"` along with
the chosen tool name and args in structured output.

The `ReasonedOutput` Pydantic model in `state_ai.py` gains:

```python
class _Route(StrEnum):
    reply        = "reply"
    ask          = "ask"
    plan         = "plan"
    internal_tool = "internal_tool"   # new

class ReasonedOutput(BaseModel):
    ...
    route: _Route
    # populated only when route == internal_tool
    internal_tool_name: str | None = Field(None)
    internal_tool_args: dict | None = Field(None)
```

When `route == "internal_tool"`, `StateAI.run()` sets `agent.pending_internal_tool`
and returns `next_state = "internal_tool"` in `structured_data`.

### 5. State graph update

`state_graph.yaml` gains the new state and the new transition from `agent_reply`:

```yaml
agent_reply:
  type: agent
  transition:
    next: [ask_user, use_tool, workshop_plan, internal_tool]   # added

internal_tool:
  description: "invokes a system-level tool inline; result is fed back to agent_reply"
  type: internal_tool
  transition:
    next: [agent_reply]
```

The executor graph (`graphs/executor.yaml`) does not change. Internal tools are never
invoked from a subagent plan step -- the executor routes through `use_tool` regardless.

### 6. System prompt injection

`StateAI` already receives the agent's `system_prompt` which lists available tools. The
prompt-building code should separately enumerate internal tools with a label so the LLM
understands when to use `internal_tool` vs `plan`:

```
Internal tools (call these inline without a plan):
  - brave_web_search: search the web for current information

User tools (require a plan and approval):
  - create_issue: create a Linear issue
  - ...
```

This is done in the same place `state_ai.py` currently builds `system_parts`.

---

## Data flow (before vs. after)

**Before (all tools):**
```
user message
  -> agent_reply (StateAI decides to act)
  -> workshop_plan (LLM writes plan, sent to dashboard)
  -> [user approves]
  -> use_tool (StateTool calls SmitheryManager)
  -> agent_reply (result shown)
```

**After (internal tool):**
```
user message
  -> agent_reply (StateAI decides it needs web search to answer)
  -> internal_tool (StateInternalTool calls SmitheryManager, no approval)
  -> agent_reply (LLM composes reply using tool result)
```

**After (user tool, unchanged):**
```
user message
  -> agent_reply -> workshop_plan -> [approval] -> use_tool -> agent_reply
```

---

## What does NOT change

- `SmitheryManager.call_tool()` -- same for all tool types.
- `StateTool` and `StateExecutor` -- untouched.
- The Smithery connection lifecycle -- internal tools are already handled by the
  existing `requires_auth: false` shared connection path.
- The `AuthRequiredError` path -- internal tools must have `requires_auth: false`,
  so this error should never fire for them.

---

## Testing

- Unit test `SmitheryManager.list_internal_tools()` -- returns only `internal: true`
  servers.
- Unit test `StateInternalTool.run()` -- happy path and error path.
- Unit test `StateAI.run()` -- `route = "internal_tool"` sets `pending_internal_tool`
  and returns `next_state = "internal_tool"`.
- Integration test: send a query that requires web search; verify the graph visits
  `internal_tool` before `agent_reply` and the final reply references the search result.

---

## Open questions

- **Loop guard**: If the LLM repeatedly routes to `internal_tool` without making
  progress, the existing `MAX_ITER` on `Agent` should catch it. Verify this is wired
  up for the main chat graph (currently it is only enforced in subagents).
- **Result injection**: The tool result is currently returned as `StateOutput.content`.
  `StateAI` needs to receive it as a message in context. Confirm `StateHandler` appends
  `StateOutput.content` from every intermediate state to the context list before the
  next state runs. If not, a small change to `state_handler.py` will be needed.
- **Multiple internal tool calls per turn**: For now, one internal tool call per turn
  is sufficient. If the LLM needs to chain two searches, it will make a second turn.
  Revisit if this is too limiting in practice.
