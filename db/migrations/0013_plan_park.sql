-- An unattended run starts from an approved plan (Task 11.8.5).
--
-- The play button used to flip the mode itself, handing the model a transcript
-- rather than a task. The 2026-08-20 Marketplace run went unattended with the
-- model's own unanswered question as the last event and burned its budget
-- greeting nobody. Now there is exactly one door: `propose_plan` parks the
-- session on a row of kind `plan`, and approving that row is what flips the
-- mode.
--
-- `plan` reuses the `call` machinery's columns rather than growing its own,
-- because it is the same shape of promise: `tool_args` carries the artifact the
-- human actually read, and approving binds to THAT — the harness writes those
-- args to plan.md and the run starts from the file. Consent binds to the plan,
-- never to prose about it, for the same reason it binds to the call.
--
-- No version column. Each `propose_plan` is a new row, and a row's version is
-- its position in the session's plan history: the versions ARE the rows, and a
-- counter that disagreed with them would be a second source of truth. A newer
-- plan supersedes the open one (`answer = 'superseded'`) so a session never has
-- two live plans.
--
-- That last invariant is enforced in code (`approvals.supersede_plans`), NOT
-- here. `idx_approvals_one_open_per_call` is keyed on (session_id,
-- tool_call_id), and two proposals always carry different tool_call_ids, so the
-- database would happily hold both. It is not tightened to one open plan per
-- session because the same index is what permits exactly one open `call` across
-- a park, and narrowing it would collapse two rules into one.

ALTER TABLE approvals DROP CONSTRAINT IF EXISTS approvals_kind_check;
ALTER TABLE approvals ADD CONSTRAINT approvals_kind_check
    CHECK (kind IN ('approval', 'ask', 'call', 'plan'));

-- The card reads a session's whole plan history to number the versions and to
-- diff the newest against the one before it.
CREATE INDEX IF NOT EXISTS idx_approvals_plans
    ON approvals (session_id, created_at)
    WHERE kind = 'plan';
