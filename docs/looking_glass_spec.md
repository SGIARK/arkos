# Looking Glass — Product Spec

Companion to `docs/single_loop_redesign_spec.md` (build plan) and
`docs/contracts.md` (law: event vocabulary, endpoints, lifecycle).
Scope: the product surfaces — Looking Glass, Projects, Command Center.

**Status:** LG-1, LG-1.5/1.6/1.7 and LG-2 DONE · amended by Task 11.4 (2026-08-19) | **Author:** John Wallace | **Last updated:** 2026-08-19

**Amended by Task 11.4 (owner, 2026-08-19), three changes, all from the
`new_frontend/` design, which is ground truth where it and this spec disagree:**

1. **The rail says what it shows.** "looking glass" is now **projects** and
   "computer" is now **files** — finishing what LG-1.7's amendment already said
   in prose. The surface is unchanged; only its name in the nav is. Old hashes
   (`#computer`, `#looking glass`) still land where they used to.
2. **The browser is a POPOUT, not a fixed canvas.** Where "The right panel"
   below says the browser canvas is chosen and then full-height, read it as:
   the canvas holds a live thumbnail with the run's own status line, and
   clicking it opens the stream over the conversation at a size a page can
   actually be read at; escape puts it back. The reasoning in that section is
   unchanged and is in fact the argument for this — 300px was never "full-height
   and readable", it was a notification. It still does not steal focus.
3. **The composer carries the session's reach.** A `tools used/budget` chip sits
   left of the `ark>` prompt with the per-server panel behind it. Choosing what
   a session can reach belongs next to asking it for something; see Task 11.4 in
   `single_loop_redesign_spec.md`.

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

**Amended 2026-08-18 (owner): the landing is the CHAT, not the grid.** Opening
the app lands in the user's home session — the buddy, a standing ordinary
attended session auto-created on first login — because the product's identity
is a companion with memory, and identity is what greets you at the door.
Looking Glass (the grid below) is the **"Projects" tab**: the work surface
where chats/sessions live in their project bubbles, one click away. Level 2 is
unchanged and shared — the same window component renders the home chat and any
project session (D5). "The landing" in the paragraph below reads as "the
Projects tab" from here on.

**Level 1 — project grid (the Projects tab, and the task list).** Rounded
project bubbles, status dot top-right = lifecycle rollup:

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
  A suggestion, never a direct action. Carried into a turn already running at
  the top of its next hop (LG-1.8).
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
Frames are ephemeral and never replayed from the log. (**Amended by 11.4:** the
canvas holds a thumbnail and the full-size view is a popout over the
conversation — see the amendment at the top of this file. The stream URL is also
recovered from the session snapshot, so reloading the page mid-run no longer
loses the pane until the next step happens to fire.)

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

**Status: DONE 2026-08-18**, two commits (backend gaps + contracts rows
`66a1159`, surface `b5287be`). Sign-in is the real one (supabase-js
`signInWithPassword` → `/auth/session` → cookie; auth.md closed to its last
rollout-gate item). The surface is four files (`api.jsx`, `grid.jsx`,
`window.jsx`, `app.jsx`); `seed.jsx`, `components.jsx`, `views.jsx` deleted
with the dead model they rendered. **The JSX is unexecuted** — no browser has
rendered it; first human (or Chrome-driven) pass pending. Carried: the
supabase-js SRI hash (command in `index.html`).

## Task LG-1.5: The settings panel — resurrect, don't rewrite
**Why:** the `views.jsx` deletion took a complete `SettingsModal` whose popup
choreography was already right, and the model's own `auth_required` message
promises this panel exists ("authorize it from the connections panel in
Settings") — a live contradiction until this lands.
**Done when:** `SettingsModal` ported from
`git show b5287be^:frontend/views.jsx` (component at ~lines 238-360) onto the
new endpoints — `GET /connections`, `POST /connections/{server}/connect`,
`DELETE /connections/{server}` — keeping its hard-won behavior intact: popup
opened BEFORE the await so the user gesture is not lost, poll for popup close,
refresh on completion. Only the endpoint calls change. Sign-out lives here.
**Touch:** frontend | **P1, 0.5d** | **Blockers:** LG-1
**Test:** connect flow completes through the popup against a fake Smithery;
disconnect removes the row; an unconnected server's setup link opens the panel.

**Status: DONE 2026-08-18.** `frontend/settings.jsx`, ported rather than
rewritten: the popup opens synchronously inside the click handler, the watcher
polls until the server says connected or the popup closes, and the blocked-popup
fallback surfaces the setup url as a link. Only the endpoint calls changed. Two
things did not come across, both deliberately: the Slack DM section (no endpoint
exists for it, and inventing one would be building ahead of the backend) and the
backend-host field (there is one origin now, by construction). Sign-out lives
here; the entry point moved to the rail's foot in LG-1.6.

## Task LG-1.6: The rail — desk-level peripheral vision
**Why:** the old app's rail gave a standing view of what needed approval and
what was running; the LG-1 surface dropped it. This restores it on the new
endpoints — it is the persistent projection of Command Center (see Design),
scoped to v1.
**Done when:** a slim persistent sidebar across grid and window, two sections.
**"Needs you":** user-scoped `GET /attention` (approvals AND unanswered asks),
each item navigating to that session's window at the pending question.
**"Running":** the user's live sessions with project names, served by a new
`GET /sessions?status=running` (contracts row in the same commit — the nested
per-project list does not compose for a cross-project view). Refresh on
navigation and on any open session's stream events; NO polling, and no global
live feed — that is G27 (needs the hold-back window) and stays deferred:
slightly stale peripheral vision is acceptable, a polling loop is not. The
theme toggle and the settings entry point (LG-1.5) live at the rail's foot.
**Touch:** frontend, `api.py` + contracts (one row) | **P1, 0.5-1d** | **Blockers:** LG-1
**Test:** an unanswered ask appears in the rail from the grid AND from another
session's window, and clicking it lands at the question; a running session
appears while running and leaves on terminal; no network request repeats on a
timer.

**Status: DONE 2026-08-18.** `frontend/rail.jsx`, both sections, persistent
across tabs. `GET /sessions?status=running` is the one row added, with its
contracts entry in the same commit. No timer: the rail reloads on navigation, on
a tab switch, and when an open session's stream reports a `lifecycle` or `done`
— which is precisely what its two sections are made of. Everything else the
stream carries is ignored by it, so a chatty session does not turn the sidebar
into a polling loop by another name. The global live feed stays G27-deferred.

## Task LG-1.7: Chat-first landing — the buddy at the door
**Why (owner, 2026-08-18):** the product is a companion with memory; opening
the app should be walking up to your desk where ARK already is, not opening a
project tool. This reverses the spec's original "grid is the landing" (amended
above): the landing is the chat, Looking Glass is the "Projects" tab.
**Done when:** on sign-in the app opens the user's **home session** window —
an ordinary attended session auto-created on first login (`users.home_session_id`,
set once; no special-casing anywhere else: it may sit `idle` forever, per the
lifecycle). Top-level nav is two tabs, **Chat** (the home window, default) and
**Projects** (the grid); the rail (LG-1.6) persists across both. The window's
back link becomes the tab switch. Placeful creation is unchanged: composing in
the Projects tab creates sessions there; the home chat is where heavy work is
DISCUSSED and forked from — "fork this to a project" affordances stay future.
**Touch:** frontend routing, one migration column, `POST /auth/session`
(create-home-session on first login) | **P1, 0.5d** | **Blockers:** LG-1
**Test:** first login creates exactly one home session and lands in it; second
login lands in the same one; the Projects tab shows the grid with the home
session's project bubble present like any other; sign-out and back preserves
the home session.

**Status: DONE 2026-08-18.** Migration 0008 adds `users.home_session_id`;
`POST /auth/session` creates the session on the first-login path only, guarded by
a conditional UPDATE so two logins racing cannot make two. `GET /auth/me` carries
the id, because the page needs it on first render and no other request would.

Two consequences worth knowing, both found by the tests rather than by reading.
The home session's project appears in the grid — which the card asked for, and
which changed what `GET /projects` returns for a brand-new user from nothing to
one bubble. And the home session was silently spending the human's
`new_sessions_per_hour` allowance: **fixed**, the quota now excludes it, because
a limit on what a person asks for should not be spent by the server greeting
them.

Not built, and not asked for: any "fork this to a project" affordance. The home
chat is where heavy work is discussed; moving it stays future.

## Task LG-1.8: Steering reaches the turn it was typed into
**Why (found in use, 2026-08-18):** a message sent while a browser_task was
running — "clone it onto your computer" — appeared in the transcript, was never
seen by the model, and the run finished answering the previous question. The
promise is in three documents (contracts' endpoint table, the decision tables,
the Suggest bullet above) and in none of the code: `runner.fold` builds the
message list once, `run_turn` mutates it in place across hops, and nothing
re-reads the log while a turn is in flight.
**Done when:** between hops, the loop reads events appended since the last one
it saw and appends any `user` events to `messages` — the same place and shape as
the finish nudge, honouring the fold's existing rule that a user message landing
mid-tool-call waits for the results to close. A message arriving after the last
hop but before `done` is carried by the next turn rather than dropped.
**NOT this card:** interruption. Steering waits for the current hop; stopping a
run mid-step is `POST /cancel` and stays that way (owner, 2026-08-18).
**Touch:** `agent_module/loop.py`, `harness_module/runner.py` | **P1, 0.5d** | **Blockers:** none
**Test:** a message posted while a turn is running appears in the model's
messages on the following hop, after the open tool call's result and before the
next completion; two messages arrive in order; one posted after the last hop is
read by the next turn.

**Status: DONE 2026-08-18.** `run_turn` takes a `steer` callback and calls it
once per hop; `runner._steering` closes over the fold's `last_seq`, reads
forward, and hands back only what a human typed — the loop keeps no knowledge of
the log, which is the harness's. Reading from the fold's cursor rather than the
turn's start means a message that landed between the fold and the first hop is
carried too, and advancing past everything read means nothing arrives twice.
Six tests, including the one that has been missing since `f65039b`: the promise
entered contracts in the first law commit and no version of `loop.py` had ever
read the log.

## Task LG-2: Projects + Command Center
**Done when:** `projects`/`project_files` tables; project-per-task default;
`GET /projects` rollup; file upload + sandbox mount; `GET /attention` at three
scopes, resolvable from any, same wake event.
**Touch:** migration, `harness/api.py`, frontend IA
**Priority:** P2 | **Effort:** 3-4d | **Blockers:** LG-1, redesign Tasks 6, 8
**Test:** upload a file, task reads it from its sandbox; its approval shows in
project list AND Command Center; resolving from either resumes at cursor.

**Status: DONE 2026-08-18.** Most of this card was already standing when it
came up, built underneath it by the store work: `projects`/`project_files`
(migration 0 + 0001), project-per-session by default (`POST /sessions` makes one
when none is named), the `GET /projects` rollup, and upload + sandbox mount
(Tasks 8.4 and 8.7 — an upload lands in the store and is written through to
every live box holding a materialized claim). What was actually missing was
smaller than the card, and it is worth saying so rather than restating the card
as if it had all been built here.

Built now: **`GET /attention` at the third scope.** It had user and project;
`session_id` completes it, and the window asks at its own scope instead of
fetching the account and filtering in the browser. One row, three projections,
answered from any of them — pinned by a test that opens one question, sees it at
all three scopes, resolves it from the window, and watches it leave all three.

And the **right panel**: a pinned TODO above one canvas, files or browser, your
choice, collapsible with the choice remembered per browser. The plan moved out
of the transcript into the panel where it stays glanceable rather than scrolling
away. Files are listed from the tree, uploaded by drop or picker, and a live
browser run gives the Browser tab an activity dot — available, never stealing
focus. Task 9's frame pane moved in here, which is where the design always said
it went.

Not built: project memory as a third canvas (D8 — a tab for a feature that is
not in the product), and the human tool takeover the card defers by design.
Still unrendered: no browser has run any of this.

**Deferred (owner, 2026-08-18): the computer view stays per-project.** It lists
the STORE — `GET /projects/{id}/files` — so its dropdown offers projects and
nothing else, and anything the box holds outside a claimed mount is invisible to
it. That is the whole durable filesystem, so it is the right default. Letting it
walk the sandbox disk itself is a later card: it needs HTTP over the box
(`list_dir` and `read_file` exist as model tools, not as endpoints), and it
would show a filesystem that dies with the session beside one that does not,
which wants a deliberate design rather than a second dropdown entry.

**Half of that came due in Task 11.4 (2026-08-19): the endpoints exist.**
`GET /sessions/{id}/fs` and `GET /sessions/{id}/fs/file` read the live box,
ownership-checked and never booting one — a parked or reaped session is a 404.
The other half is deliberately still open: nothing points the files view at
them, because the question of showing a disk that dies beside a store that does
not is exactly the design pass this paragraph asked for, and the 11.4 design,
broad as it is, does not answer it.

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
