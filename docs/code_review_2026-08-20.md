# Code review — bloat, route overlap, frontend consolidation

**Date:** 2026-08-20 · **Method:** three parallel review agents over a staged snapshot
(api layer · backend modules · frontend), findings spot-checked before inclusion.
**Snapshot caveat:** taken while the coding session was actively landing 11.7 —
`api.py`, `approvals.py`, `runner.py` were minutes old. **Not reviewed** (missing
from the agents' snapshot): `tool_module/smithery.py`, `tool_module/session_tools.py`,
`tool_module/tools/control.py`, `frontend/components.jsx`, `frontend/api.jsx`.
Re-run over those before calling the review complete.

**Verdicts.** Routes: NOT bloated — 31 routes, 30 contract-documented; overlaps are
deliberate projections. Backend: disciplined core carrying ~4% pure duplication and
~250 dead lines; runner.py and store.py each absorbed three jobs. Frontend: no
rampant copy-paste; one contracts violation and a scattered shared-primitive layer;
~10% consolidatable. Total Python ~10.5k lines vs the spec's stated 4.5k success
criterion — the target is stale or a trim card is owed.

---

## P0 — fix now

1. **`api.py:1356` — composer prose silently declines a gated call.** (verified)
   `_answer_by_message` refuses only `kind == "approval"`. A message typed into a
   session parked on a `kind == "call"` row (11.7) falls through and answers the
   call row; anything but the approve word reads as declined. "sounds good, go
   ahead" = silent decline without the human seeing the call. Fix: refuse `call`
   alongside `approval` in the guard (or share one answer helper with
   `respond_to_approval`).

2. **`views.jsx:775-789` — 2s `setInterval` polling `api.connections()` during
   OAuth.** Contracts forbids polling anywhere. Fix: one-shot recheck on popup
   close plus `focus`/`visibilitychange`, or a `connection` event on the stream.

3. **`lifecycle.py:179` — `close_dangling` without publish.** The append-then-
   publish pattern is copy-pasted at runner.py:497, :734, :1241 and api.py:770,
   and this fifth site forgets the publish — same bug class 11.4's first commit
   fixed for `transition`. Fix: `publish_all` helper in `stream.py`, used by all
   five sites.

## P1 — duplication worth collapsing (backend)

- `_uuid` coercion defined **12×** in 3 inconsistent flavors (runner.py:1373,
  store.py:938, workspace.py:414, approvals.py:194, lifecycle.py:190,
  session_log.py:267, system_log.py:148, leases.py:84, sandbox/manager.py:74,
  connections.py:52, tools/world.py:34, api.py:1476). One home (e.g. `db/ids.py`),
  ~45 lines back.
- `_cfg(key, default)` defined **8×** (runner, store, loop, system_log, registry,
  browser/tool, tools/memory, api). Give `config.get` a `default=` param, ~30 lines.
- `workspace.py:510` `write_through` inlines the query `_live_boxes` wraps.
- `runner.py:640` `_result_event` re-implements loop.py's `_settle`/`_cap_view`
  normalisation — export one `result_event_from_envelope`, keeps paths from drifting.
- Declined/interrupted sentences duplicated across loop.py:382 / session_log.py:182
  and runner.py:605 / envelope.py:155 — single constants.
- `connections.py:70-173` — every function written twice (shared vs user tables);
  a table selector collapses each to one statement, ~40 lines.
- `runner.py:820` `_park_calls` duplicates the gate rework's `_calls` — likely
  mid-flight leftover; delete one.
- api.py copy-paste: file-payload shaping (609-625 vs 1257-1261), inline project-
  ownership SQL bypassing `_owned_project` (331, 1385), session wire-shape built
  3× (398, 421, 490), `updated_at` touch inlined 3× vs `lifecycle.touch_project`.

## P1 — dead code (backend, ~250 lines)

- `store.py:599-738` snapshots section (Snapshot, snapshot/list/restore/prune) —
  zero callers in-tree. Confirm no out-of-tree script, then delete (~140 lines).
- `store.py:588` `diff_tree` + `TreeDiff` — flush diffs inline; delete (~28).
- `store.py:862` `read_notes` + `Note` — no caller; delete or wire the compactor.
- `leases.py:56,75` `release`/`holder` — only `release_all` is used (~18).
- `envelope.py:54` `ResultEnvelope.retryable` + `_RETRYABLE` — written everywhere,
  read nowhere (verify smithery.py first).
- Small: `stream.py:64 subscriber_count`, `browser/stream.py:70 watching`,
  `sandbox/manager.py:365 awake`, `events.py:150 is_terminal` + `:37
  TERMINAL_REASONS`, `workspace.py:76-79` unread `Materialized` fields,
  `workspace.py:129 lease_keys` (or have runner.py:932 call it instead of
  re-deriving the rule).

## P1 — frontend consolidation (~10%)

- `api.attention()` fetched **4×** per pulse (app.jsx:77, views.jsx:33, :130,
  lookingglass.jsx:33) — fetch once in App, pass down.
- Snapshot-on-pulse effect copied 3× (views.jsx:29, :128, lookingglass.jsx:29) —
  `useSnapshot(fetcher, [pulse])` hook.
- List-row JSX copied 3× (views.jsx:80, :106, lookingglass.jsx:96) — one row
  component.
- File-tree module (~130 lines, views.jsx:169-296) consumed by lookingglass.jsx —
  belongs in components.jsx; same for `PageHead`, `fileSize`, upload loop
  (views.jsx:394 vs lookingglass.jsx:497), Set-toggle (views.jsx:328 vs
  lookingglass.jsx:476), escape-key effect (×3), auto-scroll effect (×2).
- Modal/scrim scaffolding 3 ways; `@keyframes pop` defined twice (styles.css:519,
  :777 — second silently overrides the first).
- Five status-dot implementations; `Dot` has two prop APIs (`kind=` vs `status=`).
- CSS: five `.who` rules, five pill/chip variants, 8 mini-button rules → utility
  classes; ~25 lines likely dead (`.lg-detail`, `.lg-composer .attach` confirmed).
  Note: styles.css:1-8 declares lines 1-670 a verbatim design-export section —
  pruning inside it trades sync hygiene for bytes; owner's call.

## P2 — structure and convention

- `runner.py` (1,376 lines) = fold builder + driver/gate + `_Sink`; split into
  `fold.py` / `sink.py`. `store.py` (941) = blobs client + tree + dead snapshots +
  memory; split after deleting snapshots.
- `jwt_utils.py:108` — blocking JWKS fetch (urllib) inside async request path;
  wrap in `asyncio.to_thread` or pre-warm at startup.
- `runner.py:625` `dispatch_granted` pokes `_Sink` privates from module level;
  stale comment at :826 ("so `abort` knows" — abort never reads it) — mid-flight
  gate-rework residue, tidy when 11.7 settles.
- Second-gated-call-in-hop branch (runner.py:893) reuses kind `"approval_required"`
  and leans on `_park` state; a distinct kind would remove the coupling.
- Missing return hints: runner.py:766, sandbox/tools.py:29, several `__init__`s.
- `GET /health` has no contracts row (the only route/contracts drift).
- Frontend file headers stale: views.jsx still says "computer", lookingglass.jsx
  still says "looking glass"; ~330 of lookingglass.jsx's 634 lines are
  SessionTools/FilesCanvas/BrowserCanvas — candidates for their own files.

## Follow-ups

1. Fix P0.1 before any real gated-call use (it undermines 11.7's whole premise).
2. Re-run review over smithery.py, components.jsx, api.jsx.
3. Decide: refresh the 4.5k-line target in the spec, or card a trim pass
   (the P1 lists above are ~600-700 lines of it).
