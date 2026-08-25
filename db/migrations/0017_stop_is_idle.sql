-- A stop is not a park, so it has no approvals row (Task 11.8.7).
--
-- 11.8.6 held a stopped run on a row of kind `resume` with three answers. But a
-- stop is not a question and a held run is not waiting on consent: the plan it
-- was approved from already stands. 11.8.7 lands the stop as
-- `done{stopped}` -> idle with the mode KEPT, and resuming is then the code
-- that already existed — an idle session starts on a message or a plain start,
-- unattended because the mode never moved. No row to answer, no arm in
-- `respond`, no exemption in the composer's 409.
--
-- Any row still open is closed rather than dropped. It is a real thing that
-- happened to a real session and the transcript keeps its own record; what must
-- not survive is a row nothing can answer, sitting in `/attention` forever and
-- turning a project's dot ochre for a run that is idle and resumable.

UPDATE approvals
   SET answer = 'superseded', answered_at = now()
 WHERE kind = 'resume' AND answered_at IS NULL;

-- A stopped session was left `awaiting_approval` by the old hold. With its row
-- closed it would sit there with nothing to wait for, so it lands where 11.8.7
-- puts a stop: idle, mode untouched, resumable by a message or a start.
UPDATE sessions
   SET status = 'idle'
 WHERE status = 'awaiting_approval'
   AND NOT EXISTS (
       SELECT 1 FROM approvals a
        WHERE a.session_id = sessions.id AND a.answered_at IS NULL
       );

ALTER TABLE approvals DROP CONSTRAINT IF EXISTS approvals_kind_check;
ALTER TABLE approvals ADD CONSTRAINT approvals_kind_check
    CHECK (kind IN ('approval', 'ask', 'call', 'plan'));
