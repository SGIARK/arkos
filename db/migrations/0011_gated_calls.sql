-- Consent binds to the CALL, not to a sentence about it (Task 11.7).
--
-- Until now a gated tool call was refused inside dispatch with a message telling
-- the model to describe its intent via `request_approval`. The human then
-- approved that DESCRIPTION, and nothing tied the approval to the call that
-- followed — the model could describe one thing and call another. Worse, the
-- gate never read this table, so the grant it promised could never be found and
-- every gated call looped forever.
--
-- Now the turn parks on the gated call itself, with the call left OPEN, and the
-- row below carries what will actually run.
--
-- `tool_name` / `tool_args`: the real call, so the human approves the thing that
-- executes and the resumed run has it without asking the model again. Null for
-- the two prose kinds, which have no call of their own.
--
-- `consumed_at`: the latch that makes "exactly once" true. A granted call is
-- executed by whichever wake claims it, and claiming is a conditional UPDATE on
-- `consumed_at IS NULL` — the same pattern `answer()` uses, for the same reason.
-- Concurrent wakes therefore admit exactly one executor. A row that is consumed
-- while its call is still open means the process died mid-flight: the outcome is
-- unknown, so the repair closes the call as interrupted and never re-runs it.
-- Sending a message twice is worse than not knowing whether it sent once.
--
-- The partial unique index on (session_id, tool_call_id) already guarantees one
-- open row per call, which is what keeps "exactly one open tool call across a
-- park" enforceable in the database rather than only in the runner.

ALTER TABLE approvals DROP CONSTRAINT IF EXISTS approvals_kind_check;
ALTER TABLE approvals ADD CONSTRAINT approvals_kind_check
    CHECK (kind IN ('approval', 'ask', 'call'));

ALTER TABLE approvals ADD COLUMN IF NOT EXISTS tool_name   TEXT;
ALTER TABLE approvals ADD COLUMN IF NOT EXISTS tool_args   JSONB;
ALTER TABLE approvals ADD COLUMN IF NOT EXISTS consumed_at TIMESTAMPTZ;

-- The resume looks for exactly this: a gated call, answered, not yet claimed.
CREATE INDEX IF NOT EXISTS idx_approvals_grantable
    ON approvals (session_id)
    WHERE kind = 'call' AND answered_at IS NOT NULL AND consumed_at IS NULL;
