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
| `tool_call`, no result | any, `awaiting_approval` included | nothing wrote a failure (the process died mid-call) — append `tool_result{ok:false, interrupted}`: "outcome unknown, verify before retrying". A park closes its own call before parking, so in `awaiting_approval` this row means the process died between the two |
| `tool_result` / assistant text / user message | — | load as-is |
| empty | — | fresh session: system prompt + goal |

Why the repair is required, not optional: the message list is rebuilt from the
log on *every* load (nothing is held in memory). SGLang's chat template rejects
a request where a `tool_call` id has no matching tool message, so an unclosed
call makes that conversation permanently unloadable, not just once. Writing the
stand-in IS the "tool call failed" record nobody was alive to write.

**Settled 2026-08-16 (owner): a tool call is never left open — a park closes its
own call first.** This document used to say an open call in `awaiting_approval`
was "parked, healthy", which contradicted the transcript invariant at
`contracts.md:52` and, more damningly, the paragraph directly above: a session
that parks with its call open cannot be folded back into messages, so it is
unwakeable on the very next load — the park would brick the session it exists to
suspend. `request_approval` and `ask` therefore return a real result ("asked, awaiting
a human"), the call closes, and the session parks with a clean transcript. The
human's answer arrives later as a `user` event, which is what wakes the run; it
is not that call's return value. Consequence for Task 6: the respond endpoint
appends a `user` event and wakes at the cursor — it never back-fills a
`tool_result`, and there is no state in which a resume has to reconcile one.

## 2. End of a hop — what did the model produce?

| Model output | mode | Action |
|---|---|---|
| tool calls | any | execute (readonly parallel, mutating serial), append, loop |
| `finish_task` returns ok | unattended | `done{completed}` |
| `finish_task` errors, has bad args, or is at its attempt cap | unattended | closed as a failed `tool_result`, run CONTINUES; completion comes from the result, never the call |
| malformed tool args | any | `tool_result{invalid_args}`, one free repair round trip per turn (not charged as a hop), then charged like any hop |
| a tool fails `per_tool_attempts` times in a row | any | further calls closed `upstream_error` without dispatch; a success at any point clears the streak |
| empty completion (no text, no calls) | any | `done{stalled_progress}`; nothing errored and there is nothing to continue from, so looping on it just spends the budget |
| `finish_reason: length` with no calls | any | `done{context_overflow}` |
| `request_approval` / `ask` | any | close the tool_call with its result FIRST, then park → `awaiting_approval`, exit loop. The answer arrives later as a `user` event, not as this call's result |
| a human presses Stop | any | calls in flight close `cancelled_by_user`, the rest of the hop is refused the same way, and the turn holds at the next hop boundary on a `resume` row — no `done`, no mode flip, hops preserved. A terminal reached in the same hop wins instead |
| `propose_plan` | any | same park shape, kind `plan`, and the row carries the plan itself. The answer is not read by the model at all unless it is feedback: approve writes `plan.md` and wakes the session UNATTENDED, decline closes the park to `idle`, anything else is a REPLY — the human's `user` event plus an unrendered `user{source:system}` instruction — and wakes it attended to propose again |
| `todo_write` | any | replace the current list (latest-wins); emit `todo` event; older todo results drop out of the view |
| text only | attended | `done{turn_end}` → `idle` (non-terminal: no `terminal_reason`, no `ended_at`) |
| text only, 1st in a row | unattended | append, inject the CONTINUATION as `user{source:system}`, loop |
| text only, 2nd in a row | unattended | append, inject the finish nudge immediately, loop |
| text only, 3rd in a row | unattended | `done{stalled_progress}` — told to continue, told to finish, still only text |
| any tool call | unattended | the bare-text streak resets to zero |
| text only, budget exhausted | unattended | `done{max_hops}` |

## 2b. Mode — attended vs unattended (one session, two phases)

| Event | Mode after | Effect |
|---|---|---|
| session created, human present | attended | turn-taking; ends its turn → `idle` |
| human presses play | attended (UNCHANGED) | appends a `user{source:system}` handoff and starts an ordinary attended turn, whose job is to call `propose_plan`. It does not hand the session over |
| human approves a proposed plan | **unattended** | the ONE mode flip: `plan.md` is written from the approved args, the quota is checked, and mode + status move in the same conditional UPDATE. `finish_task` now required; budgets + wall-clock apply; runs without you |
| human declines a proposed plan | attended (unchanged) | the park closes → `idle`; nothing ran |
| human presses Stop on a run | unattended (UNCHANGED) | the hold: `running -> awaiting_approval` on kind `resume`. The plan stays approved and the hop budget carries, because no `done` was written |
| human resumes a stopped run | unattended (unchanged) | approve word, prose in the card, or prose in the COMPOSER — all three resume; prose lands as a `user` event the next hop reads |
| human cancels a stopped run | attended | the second press: `awaiting_approval -> cancelled`, the resume row closes, and the plan's approval is spent |
| `done{...}` on an unattended run | attended | terminal status per the lifecycle table; `mode` flips back in the SAME update |
| human sends a message while unattended | unattended (unchanged) | steering: appends as a user event, reaches the stream immediately, and is carried into the running turn at the top of the next hop (LG-1.8). Never mid-hop: the current tool call finishes first |

"A human is the continuation" is a fact about the current mode, not a kind of
session. Same termination rule (D15), evaluated against `mode`.

## 3. Tool result — what came back?

| Envelope | Attempts on this tool | Action |
|---|---|---|
| `ok` | — | append, continue |
| error, retryable | < 3 | append, continue (the model decides whether to retry) |
| error, retryable | = 3 | append as permanent failure, continue (model must route around it) |
| `auth_required` | — | append with the setup URL in content, continue |
| needs approval, none given | — | `invalid_args` naming `request_approval`; the gate cannot park from inside dispatch, so the model asks and the session parks on THAT call. Never "declined": nobody was asked |
| `interrupted` | — | never returned by a tool; only synthesized on wake |
| `cancelled_by_user` | — | never returned by a tool either; synthesized by the dispatch wrapper when a human presses Stop. Closes the call, and is the ONE failure kind that does NOT count toward the attempt cap |

A run that does not know what to do next PARKS — on a question, an approval or
a plan — rather than failing. There is no "confused" terminal: an unattended run
ends by `finish_task`, by parking, or by a named budget or stall reason.

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

## 7. Resource leases (browser, store folder) and the sandbox pool

The browser is leased per user (`browser:{user}`) and a store FOLDER is leased
per write claim (`folder:{user}:{name}`, 11.9), so two projects writing
different folders never wait on each other and two sessions writing the same
folder do. The sandbox is not leased: a box belongs to one session, and what is
shared is capacity.

| Condition | Action |
|---|---|
| first use of a leased resource in this run | `acquire(resource_key, session_id)` |
| already held by another session | stay `running`, emit `status{label:"waiting for {resource}"}`, wall clock paused; NOT a park (no ochre, no /attention row) |
| first use of the sandbox in this run | claim a slot in the user's pool; over `sandbox.max_concurrent_per_user`, wait the same way as a lease |
| task parks on approval | flush, release every lease, **pause** the box and keep its slot |
| task resumes | re-acquire the leases, resume the paused box (or rebuild it from the store, which is the record either way) |
| task terminal | flush, release every lease, kill the box and free its slot |
| slot expires unrenewed | reclaimed, and the box it names is killed |

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
