# Looking Glass — Product Spec

Companion to `docs/single_loop_redesign_spec.md` (build plan) and
`docs/contracts.md` (law: event vocabulary, endpoints, lifecycle).
Scope: the product surfaces — Looking Glass, Projects, Command Center.

**Status:** Not started | **Author:** John Wallace | **Last updated:** 2026-07-24

---

# Problem

A running session is invisible: it runs blind to terminal, then reports a summary
string on a row. Once a session goes unattended nothing reports back mid-flight,
so the human stays in the loop by OBSERVING and SUGGESTING — which makes a live
window load-bearing, not decoration. Long-horizon autonomy is
only livable with it.

---

# Design

## Sessions (recap, 3 lines)

A session = one conversation with its own log, budget, lifecycle row. It is
*attended* while you are turn-taking with it and *unattended* once you approve
the plan and it runs on its own; approving is the flip, and it returns to
attended when the run ends. **Nothing spawns** — the session you chat with is the
one that executes and the one you reopen here. Parallelism is the project grid:
five things at once is five projects.

## Looking Glass: two levels

**Level 1 — project grid (the landing, and the task list).** Rounded project
bubbles, status dot top-right = lifecycle rollup:

| Color | Meaning |
|---|---|
| green | working (`running`) |
| grey (neutral) | `idle` — alive, waiting for you |
| ochre | needs attention (`awaiting_approval` — approvals AND asks) |
| red | `failed` |
| gray (dim) | `completed` / `cancelled` |

The old "tasks" nav item is removed; this grid replaces it. Each new session gets
its own project unless explicitly pointed at an existing one — so the grid IS the
work list, and fan-out is several projects rather than several spawns.

**Level 2 — detail (the session window).** Streaming chat-style view of the
session's typed events: text streams token-by-token (`content`), tool rows
(`tool_call`), result cards with expand-via-`ref` (`tool_result`), spinner lines
(`status`), TODO panel (`todo`), budget meter (`budget`), status pill (`done`).
Composer at the bottom. **The right side is a canvas**, not a strip of metadata:
a pinned TODO list above a canvas that shows either working files or the **live
browser view** — one at a time, your choice. Collapse it and the stream takes
full width; choose the browser and it is full-height and readable, never a
thumbnail floating in a corner.
Back link to the grid. It IS the chat window — the same session, the same
component; the only difference is whether the model is currently driving itself
(unattended) or waiting on you (attended/idle).

## v1 interaction: observe + suggest (one-way)

- **Observe:** subscribe to any session's event stream, rendered live.
- **Suggest:** type into any session, exactly like chat. Appended as a user
  event, injected at the next hop; the model decides what to do with it.
  A suggestion, never a direct action.
- **In-window pause:** `needs_input` renders inline in the window and is
  answered there — not in a separate approval tray.
- **Deferred (two-way):** human tool takeover, cursor/screen control. Deferring
  it also defers every resource race (computer/browser/state) — v1 has a single
  actor per resource by construction.

## Projects

The durable container; base unit a session. A project owns working files
(durable, user-visible, uploadable; mounted into a task's sandbox at provision —
the filesystem/computer split) and, later, project memory (memory is currently
removed; see contracts.md). Project-per-session by default keeps the grid honest.

## Command Center: attention as projections

An approval/ask is a lifecycle state on a task — not owned by any surface. It
appears simultaneously wherever the task is visible: inline in its window, in
its project's attention list, and in the global Command Center. All three are
`GET /attention` at task/project/user scope; resolving anywhere writes the same
respond event and wakes the task at its cursor. Nothing is duplicated.

## The right panel: pinned TODO + one canvas

Three things wanted this space and they are not peers, so they do not get equal
billing:

- **TODO — pinned, always visible.** Five lines, always relevant during a run,
  glanceable. Hiding a checklist behind a tab is worse than showing it.
- **Canvas — one of two, chosen.** *Working files* (the project's durable files,
  droppable) or *live browser* (the CDP frame stream during a `browser_task`).
  Not both at once; you pick.
- **Project memory is NOT here.** Memory is removed (D8) — a tab for it would be
  speccing a deleted feature. It rejoins as a third canvas option if and when
  memory returns.

The whole panel collapses to zero (stream takes full width) and its state is
remembered per user. When a browser run starts, its `status` event carries the
stream URL: the browser option gains an activity dot and becomes *available* — it
does not steal focus or pop over the conversation. Choosing it is what makes it
visible, which is the difference between watching and being interrupted. Chosen,
it is full-height and legible; today's fixed 360px corner overlay
(`.browser-pane`, mounted by a permanently-open EventSource) is replaced by this.
Frames are ephemeral and never replayed from the log.

## Live updates

Per contracts.md: store-shape = wire-shape. Every surface = one snapshot request
+ one EventSource; reconnect replays from `Last-Event-ID`; no polling anywhere.
Browser video is a side-channel (frames are not events), announced by a `status`
event carrying the stream URL, mounted on demand into the canvas, cookie-authed
and ownership-checked, and torn down when the tab is closed or the run ends.

---

# Implementation

## Task LG-1: Looking Glass v1 — observe + suggest
**Done when:** project grid with rollup dots → detail window rendering the full
event vocabulary live over SSE; suggestions via `POST /sessions/{id}/messages`;
`needs_input` answered in-window; "tasks" nav removed. Strictly one-way.
**Touch:** `harness/api.py` (subscribe + suggest), frontend grid + window
**Priority:** P1 | **Effort:** 3d | **Blockers:** redesign Tasks 2, 4, 6
**Test:** grid shows correct colors per project; open a running task → tool
calls and text stream live; mid-run suggestion lands as a user event at the next
hop; an ask is answered in-window and the task continues; no code path lets the
human execute a tool.

**Two stream bugs land as this card's FIRST commit, 2026-08-17 (owner).** Both
are invisible until a browser is rendering the stream, and `curl` never hits
either, so they are fixed where their consumer arrives and tested against it.

1. **The SSE backlog is a single `LIMIT 1000` page** (`harness_module/api.py`,
   `_event_stream`, and the same read in the LAGGED re-join). Past 1000 events
   the reader sends the page, then advances `sent` to the next live event; the
   middle is never delivered, and `sent` only moves forward, so it cannot be
   recovered on that connection. This is the reconnect path, so a long session
   loses transcript on screen while the log itself is intact. Page until a read
   returns fewer rows than the limit, in both places.
2. **`lifecycle` events are appended but never published**
   (`harness_module/lifecycle.py`, `transition`). Every other append path calls
   `stream.publish`; this one does not, so the status pill moves only when a
   client reconnects. `transition()` should return the `StoredEvent` and let the
   caller publish it **after the transaction commits**. Publishing inside the
   transaction hands a subscriber a seq that a `Last-Event-ID` reader cannot yet
   fetch — the same class of bug the `append()` advisory lock closes.

**Both landed 2026-08-18, as this card's first commit.** The backlog reads in a
loop until a page comes back short, in both places (`api._backlog`), and
`transition()` returns the `StoredEvent` it appended and publishes it after the
transaction commits. The publish lives in `transition` rather than in each
caller: the bug was a caller not publishing, and five call sites each
remembering is the same bug waiting. Callers read the result in boolean context
as before — a `StoredEvent` is truthy, `None` is the lost race.

**The surface landed 2026-08-18, as this card's second commit.** `frontend/` is
four files now: `api.jsx` (sign-in, the endpoint table, one EventSource per open
session), `grid.jsx` (project bubbles with rollup dots, and what is waiting on
you at whatever scope is open), `window.jsx` (one session live — the full event
vocabulary, the composer, and the question answered where it was asked), and
`app.jsx` (sign in, then the grid or one window). `seed.jsx`, `components.jsx`
and `views.jsx` are deleted with the model they described; the old nav (desk /
tasks / watching / approvals / computer / chat) is gone, because a session is
the conversation and the run, so watching one and talking to it are one window.

Sign-in is real, not a placeholder: supabase-js `signInWithPassword`, the token
posted once to `/auth/session` and never stored (`persistSession: false`), the
cookie thereafter, and `DELETE /auth/session` on the way out. `auth.md`'s open
question 1 is settled with it — email+password at v1, no reset flow until there
are users to self-serve — and its rollout gate now has one unchecked item.

Three endpoints were added under this card because the surface could not be
built without them, each with its contracts row: `GET /projects/{id}/sessions`
(the grid could reach a project and never a session), `GET /attention` (specced
since Task 4, never routed), and `GET /auth/config` (the URL and publishable key
differ per deployment while the page is a checked-in file).

**Not verified in a browser.** The machine this was written on has no node and
could not fetch a CDN, so the JSX is unexecuted: delimiters balance and every
cross-file global resolves, and that is the whole of what was checked. The
supabase-js tag is also the one script on the page without an SRI hash, for the
same reason — the command to compute it is in `index.html`. Both want a human
with a browser before `/app` goes anywhere.

**Carded from the Task 4 review, 2026-08-17.** The frontend talks to an API that
no longer exists: `frontend/seed.jsx:156` calls `/auth/demo-login`, `:201-250`
call `/tasks` and `/computer/tasks`, and `frontend/components.jsx:217` opens
`/v1/browser/stream?user_id=…`. Separately, **nothing serves `/app`** —
`harness_module/api.py` mounts no static files and no `/app` route. That is not
cosmetic: `contracts.md:299-306` makes same-origin `/app` a deployment
requirement, because SameSite=Lax sends no cookie to a cross-site `EventSource`
and both SSE streams would 401 on every connect while the tests still passed.
Add both to this card's done-when: the grid and window read the endpoint table
in `contracts.md`, and `/app` is served same-origin with the API.

## Task LG-2: Projects + Command Center
**Done when:** `projects`/`project_files` tables; project-per-task default;
`GET /projects` rollup; file upload + sandbox mount; `GET /attention` at three
scopes, resolvable from any, same wake event.
**Touch:** migration, `harness/api.py`, frontend IA
**Priority:** P2 | **Effort:** 3-4d | **Blockers:** LG-1, redesign Tasks 6, 8
**Test:** upload a file, task reads it from its sandbox; its approval shows in
project list AND Command Center; resolving from either resumes at cursor.

### Build decomposition (tracking tasks)
Backend: per-session SSE stream · session snapshot · steer/pause-cancel/read_result
endpoints · projects overview + project-per-task · project files/upload.
Frontend: grid + detail shell · event-stream renderer · right panel (TODO,
files) · composer + EventSource wiring.

---

# Open Questions

1. ~~Does the chat window become a Looking Glass window?~~ Resolved: they are the
   same window on the same session (D5). Remaining detail: does the approve
   affordance render as a plan card inline, or as a button on the composer?
2. ~~File sync back: when a task writes a file in its sandbox, when does it
   become a project file, and what wins on conflict with a mid-run upload?~~
   **Dissolved by D27 (2026-08-18).** The question assumed files live in two
   places. They live in one: the store holds them, and the sandbox disk is a
   cache filled when a session takes its box and flushed before it gives it
   back. A file becomes a project file when the flush commits its tree, and a
   mid-run upload is written through to every live sandbox holding a
   materialized claim on that project (Task 8.6b: boxes are per session, so
   "the held sandbox" is a set) rather than racing them. See spec Tasks 8.1-8.9.
