-- Stop is not cancel (Task 11.8.6).
--
-- The only control on a run was `POST /cancel`: `task.cancel()` on the whole
-- turn, `done{cancelled}`, terminal status, mode flipped back to attended. So
-- stopping one slow step spent an approved plan — which happened on the plan
-- gate's first day of use, 2026-08-20.
--
-- Stop now holds the run at a hop boundary instead of killing it. The hold is a
-- row of kind `resume`: leases released, box hibernated, `running ->
-- awaiting_approval`, and the MODE UNCHANGED, because a park is not a terminal
-- and the plan's standing approval has to survive it.
--
-- `resume` belongs to no tool. Its `tool_call_id` is a synthetic `stop_*` id
-- with no matching `tool_call` event in the log, which is exactly right: there
-- is no open call across this park, so `close_dangling` has nothing to find and
-- the transcript invariant is untouched. The column stays NOT NULL and the
-- partial unique index keeps doing its job — one open row per id.
--
-- `tool_name` and `tool_args` are null here. A resume row carries no artifact:
-- what it holds is a run, and the run is in the session.

ALTER TABLE approvals DROP CONSTRAINT IF EXISTS approvals_kind_check;
ALTER TABLE approvals ADD CONSTRAINT approvals_kind_check
    CHECK (kind IN ('approval', 'ask', 'call', 'plan', 'resume'));
