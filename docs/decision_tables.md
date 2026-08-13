# Decision Tables — the harness service manual

Every runtime decision as "state → action." Prose hides gaps; a table shows
blank cells. Each row is also a test case.

Scope: `runner.wake()`, `run_turn()`, and the appender. Contracts state the
invariants; this states the branching.

---

## 1. On crash / restart, and on loading a session

**Policy: only `running` fails; `idle` survives.** On startup every session still
marked `running` becomes `failed{interrupted}`. The sweep synthesizes the
`done{interrupted}` event first, so the transcript says why it stopped. **Nothing
is requeued** — a blind retry re-executes an unclosed side effect. Restarting is a
human act: the operator hits restart and the session takes the
`terminal -> running` transition, the same one that lets you keep talking to a
finished session. `idle`
conversations are untouched, so deploys do not kill open chats. The log is
append-only so history is intact; a human decides whether to restart. Any session
reopened later must still load cleanly, hence the repair below.

| Loading a session, tail of log | Status | Action |
|---|---|---|
| `done{...}` | terminal | nothing to do |
| `tool_call`, no result | `awaiting_approval` | parked, healthy — leave open; the respond event closes it |
| `tool_call`, no result | any other | nothing wrote a failure (the process died mid-call) — append `tool_result{ok:false, interrupted}`: "outcome unknown, verify before retrying" |
| `tool_result` / assistant text / user message | — | load as-is |
| empty | — | fresh session: system prompt + goal |

Why the repair is required, not optional: the message list is rebuilt from the
log on *every* load (nothing is held in memory). SGLang's chat template rejects
a request where a `tool_call` id has no matching tool message, so an unclosed
call makes that conversation permanently unloadable, not just once. Writing the
stand-in IS the "tool call failed" record nobody was alive to write.

## 2. End of a hop — what did the model produce?

| Model output | mode | Action |
|---|---|---|
| tool calls | any | execute (readonly parallel, mutating serial), append, loop |
| `finish_task` returns ok | unattended | `done{completed}` |
| `finish_task` errors, has bad args, or is at its attempt cap | unattended | closed as a failed `tool_result`, run CONTINUES; completion comes from the result, never the call |
| malformed tool args | any | `tool_result{invalid_args}`, one free repair round trip per turn (not charged as a hop), then charged like any hop |
| a tool fails `per_tool_attempts` times in a row | any | further calls closed `upstream_error` without dispatch; a success at any point clears the streak |
| empty completion (no text, no calls) | any | `done{model_error}`; looping on nothing just spends the budget |
| `finish_reason: length` with no calls | any | `done{context_overflow}` |
| `request_approval` / `ask` | any | park → `awaiting_approval`, exit loop, leave the tool_call open |
| `todo_write` | any | replace the current list (latest-wins); emit `todo` event; older todo results drop out of the view |
| text only | attended | `done{turn_end}` → `idle` (non-terminal: no `terminal_reason`, no `ended_at`) |
| text only | unattended | append, inject nudge, loop |
| text only, budget exhausted | unattended | `done{max_hops}` |

## 2b. Mode — attended vs unattended (one session, two phases)

| Event | Mode after | Effect |
|---|---|---|
| session created, human present | attended | turn-taking; ends its turn → `idle` |
| human approves the plan / says go | **unattended** | `finish_task` now required; budgets + wall-clock apply; runs without you |
| `done{...}` on an unattended run | attended | terminal status per the lifecycle table; `mode` flips back in the SAME update |
| human sends a message while unattended | unattended (unchanged) | steering: appends as a user event, read at the next hop |

"A human is the continuation" is a fact about the current mode, not a kind of
session. Same termination rule (D15), evaluated against `mode`.

## 3. Tool result — what came back?

| Envelope | Attempts on this tool | Action |
|---|---|---|
| `ok` | — | append, continue |
| error, retryable | < 3 | append, continue (the model decides whether to retry) |
| error, retryable | = 3 | append as permanent failure, continue (model must route around it) |
| `auth_required` | — | append with the setup URL in content, continue |
| `interrupted` | — | never returned by a tool; only synthesized on wake |

## 4. Model call failed

| Kind | Consecutive failures | Action |
|---|---|---|
| timeout / connect / rate_limit / server_error | < 3 | backoff, retry the hop (`model_retry`) |
| timeout / connect / rate_limit / server_error | = 3 | `done{model_error}` |
| bad_request / auth | any | `done{model_error}` immediately, no retry |
| context overflow | recovery not yet tried | apply ladder rung 0/1, retry the hop (`context_recovery`) |
| context overflow | ladder exhausted | `done{context_overflow}` |

## 5. Budget checks — before each hop

| Condition | Action |
|---|---|
| `hops_remaining` == `nudge_at_hops_remaining` | inject the finish nudge (once per run) |
| hops exhausted | `done{max_hops}` |
| wall clock exhausted (active segments only) | `done{wall_clock}` |
| view over `recovery_threshold` of input budget | apply ladder rung 1 preemptively |

## 6. Cancel

| Task state | Action |
|---|---|
| `running` | signal the loop; it closes its open tool_call, releases leases, exits via `done{cancelled}` |
| `pending` / `idle` / `awaiting_approval` | harness writes `cancelled` directly (no loop to signal) |
| terminal | no-op |

## 7. Resource leases (sandbox, browser)

| Condition | Action |
|---|---|
| first use of the resource in this run | `acquire(resource_key, session_id)` |
| already held by another session | stay `running`, emit `status{label:"waiting for {resource}"}`, wall clock paused; NOT a park (no ochre, no /attention row) |
| task parks on approval | release (a parked task is not acting) |
| task resumes | re-acquire (state persisted, so files/cookies are still there) |
| task terminal | release |

## 8. Append failure

| Condition | Action |
|---|---|
| `append()` fails | halt the run loudly with a terminal event; never continue unrecorded |
| DB unreachable (cannot even write terminal) | stderr; the run dies uncommitted, wake will find a dangling call and interrupt it |

---

## Blank cells to fill later

- Grace deadline if a signalled loop does not exit (deliberately unspecified;
  the conditional-UPDATE writer already prevents a zombie loop from overwriting
  `cancelled`).
- Lease wait timeout before a parked-on-resource task gives up.
