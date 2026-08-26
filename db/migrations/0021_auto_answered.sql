-- Autopilot answers its own gates, and the row says so (Task 11.11.2).
--
-- Autopilot used to park on every gated call, so a human clicked through a run
-- that was supposed to be unattended — which defeats the word. It answers them
-- itself now, EXCEPT the destructive ones (`tools.destructive` in config.yaml),
-- which still park for a person.
--
-- The audit trail is the point of this column. An auto-answered call is a real
-- approvals row with a real answer and a real timestamp — identical to a manual
-- one in every respect except who answered it — and without somewhere to record
-- that, "the human approved this" and "the harness approved this" would be the
-- same fact in the table. They are not the same fact, and a person reading the
-- history months later has to be able to tell them apart.
--
-- NULL means a human. Not a default of 'human': the column is new, every
-- existing row was answered by a person, and backfilling a value onto rows that
-- predate the distinction would assert something this migration cannot know.
-- Absent means "nobody recorded otherwise", which is exactly true of them.

BEGIN;

ALTER TABLE approvals ADD COLUMN answered_by TEXT;

COMMENT ON COLUMN approvals.answered_by IS
    'Who answered: NULL for a human, ''auto'' for autopilot answering its own non-destructive gate.';

COMMIT;
