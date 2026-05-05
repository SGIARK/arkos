# ADR 001: Forced State Transition Override

## Status
Accepted

## Context

The agent loop in `agent_module/agent.py` drives state transitions in one of
two ways:

1. **LLM-guided**: the agent asks the LLM to pick the next state from the
   available transitions declared in `state_graph.yaml`.
2. **Forced override**: a state sets `structured_data["next_state"]` in its
   `StateOutput`, and the agent follows it directly without calling the LLM.

The forced override mechanism exists because the executor graph
(`executor → tool → approval → executor_done`) is a fixed sequence. The next
step after executing a tool is always `executor`; the next step after a
declined approval is always `executor_done`. These are not decisions — they
are predetermined. Running the LLM to "decide" would add latency, API cost,
and non-determinism to what should be an invariant sequence.

## Decision

States that know their successor declare it explicitly via
`structured_data["next_state"]`. The agent checks for this value before
falling back to LLM-guided choice.

The override is only honoured if the forced state is a declared transition for
the current state in `state_graph.yaml`. If it is not, the agent falls back to
normal choice — it never blindly jumps to an undeclared state.

## States that use this mechanism

| State | Forces next to | Why |
|-------|---------------|-----|
| `state_executor` | `executor_done`, `use_tool`, `ask_human` | Fixed plan step sequencing |
| `state_tool` | `executor` | After tool call, always return to executor |
| `state_approval` | `executor`, `executor_done` | Approved → continue; declined → end |
| `state_plan` | `ask_user` | Plan card always returns to user |
| `state_ai` | `ask_user`, `workshop_plan` | Route is determined by structured reasoning output |

## Consequences

- Adding a new executor-style state requires setting `next_state` in its
  `StateOutput` and declaring the transition in `state_graph.yaml`.
- Chat-path states (`state_ai`, `state_user`) can use the override too when
  their next step is unambiguous, reducing unnecessary LLM calls.
- The fallback invariant (`forced in transition_names`) means a misconfigured
  `next_state` silently degrades to LLM choice rather than crashing.
