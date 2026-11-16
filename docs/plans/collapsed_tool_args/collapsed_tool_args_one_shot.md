# Collapsed tool arguments are clipped

Cycle: collapsed_tool_args; Mode: one-shot; Record/index: [this record](./collapsed_tool_args_one_shot.md) · [plans index](../README.md)
Debug mode: disabled
Status: planning; next permitted action: commit this plan, then implement

## Original request

> can you please help me figure out the cause of this visual issue: https://zenmemes.com/images/07160b22e5b4dc7c5dd27cb2e7ef360d5171f2599b9bd7fc1614d5c263e26f3a.png

After the diagnosis below, the user wrote:

> can you please follow railspike one-shot for this? and do it on a branch, please

## Authorization

- On 2026-11-16 the user asked for this fix as a Railspike one-shot on a branch.
- The fix had been stated just before: remove the height cap on the collapsed row in `frontend/styles.css`.
- Exclusions: no push, no pull request and no merge. Those need a separate request.

## Outcome and work type

- Work type: application change, a bug fix in the chat frontend.
- Reported mismatch: in a chat, a tool call such as `browser_task` shows its arguments on one collapsed line. The bottom of that line is cut off, so letters like "g" and "p" lose their tails.
- Cause: `.ev-tool .args.collapsed` caps the row at 20px. The page uses border-box sizing, so 16px of padding sits inside those 20px. One 11px line at 1.65 line height needs about 18px, so only the top 12px shows.
- Allowed files: `frontend/styles.css`, `frontend/index.html` for the cache version only, and this plan folder.

## Approach and risks

- Delete `max-height: 20px` from `.ev-tool .args.collapsed`. `white-space: nowrap` already holds the row to one line. `overflow: hidden` and `text-overflow: ellipsis` still end long arguments with "…".
- Raise the `?v=52` version on every asset in `frontend/index.html` to `?v=53`. Recent frontend commits raise all of them together. Without it, a browser can keep the old stylesheet.
- Risk: a long argument string might wrap or grow the row. Impact: the collapsed row would take several lines. Early check: render a long argument in a real browser and measure the row. Mitigation: keep an explicit one-line cap if the check fails.
- Risk: the expanded row could change. Expanded rows never had this cap, so the check only confirms they still wrap.

## Reuse decisions

- `frontend/components.jsx` renders the row in the `tool_call` case and toggles the `collapsed` class. It stays unchanged.
- `.ev-gated .args` in `frontend/styles.css` styles the approval card's arguments. It has no `collapsed` state and stays unchanged.
- No new markup, class or helper is needed.

## Completion conditions

- Ordinary entry: a chat where the agent calls a tool with arguments.
- End-to-end result: the collapsed row shows its whole line with even padding above and below. A click still expands and collapses it.
- Recovery: a user who still sees clipping reloads the page. The new `?v=53` fetches the new stylesheet.
- Handoff: the fix is committed on `fix/collapsed-tool-args-clipping`. Pushing and merging wait for the user.

## Concepts touched

None. The project has no concept inventory, and this fix does not call for one.

## Verification and recovery plan

- Render the real `frontend/styles.css` with the row's real markup in headless Chromium through Playwright. Do this before and after the change.
- Check that the collapsed row's content fits its box: `scrollHeight` should not exceed `clientHeight`. Use a short argument and a long one.
- Check that a long argument stays on one line and ends with an ellipsis.
- Check that the expanded row still wraps.
- Save screenshots of each case as evidence.
- Run the CI checks that `CLAUDE.md` requires: `ruff check .`, `ruff format --check .`, the scoped `mypy` command, and `pytest tests/ -q --timeout=120 -m "not integration"` against the local test database. They cover Python only, so they confirm the branch stays green rather than test this fix.
- Inapplicable: a full app run needs the backend, model credentials and a live tool call. The isolated render uses the same stylesheet and markup, so it reproduces the reported row.
- Not checked: other browsers. The rule uses only standard CSS.

Starting revision and branch: `bcd327c` (`origin/main`) on `fix/collapsed-tool-args-clipping`
Planning commit: pending

## Dated execution history

- 2026-11-16: Diagnosed the clipping from the screenshot and from `frontend/styles.css` lines 66, 774 and 775. The cap dates from the first commit, `bc75e81`.

## Outcome

Pending.

## Verification

Pending.

## Final review

Pending.

## Commit, merge and publication status

Planning commit pending. Nothing pushed.

*— Claude · 2026-11-16 UTC · Model: claude-opus-5-5*
