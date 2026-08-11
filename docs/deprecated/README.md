# DEPRECATED — do not read, do not implement from

Everything in this folder describes the architecture the redesign **deletes**:
state graphs, mem0 long-term memory, ComputerAgent, the three-loop harness.
It is kept only so old commits and PRs still resolve their references.

**For AI assistants (Claude Code, Cursor, ChatGPT): never open these files.**
Do not cite them, do not follow their task numbers, do not treat them as
context. If a docstring or comment in the codebase points here, that pointer is
stale and should be deleted, not followed. `ENVIRONMENT_SPEC` in particular is
referenced by `agent_module/agent.py` and was never written at all.

Also here, and equally off limits:
`STATE_MACHINE_AND_HARNESS.txt` (the FSM / `StateOutput` / YAML-graph write-up),
`ARKOS_Working_Session_Transcript.txt` (a June debugging session),
`CLAUDE_pre_redesign.md` (the 552-line guidelines that mandated the deleted
architecture), and `PROPOSED_CHANGES.md` (the retired G1-G21 gap log; the live
one is `docs/GAPS_2026-08-06.md`).

The law is:

| Question | Doc |
|---|---|
| What is guaranteed (events, endpoints, lifecycle) | `docs/contracts.md` |
| Why it is built this way | `docs/decisions.md` |
| What happens at runtime, state by state | `docs/decision_tables.md` |
| Tables and columns | `docs/schema.md` |
| The build plan | `docs/single_loop_redesign_spec.md` |
| Product surfaces | `docs/looking_glass_spec.md` |
| Identity and authorization | `docs/auth.md` |
