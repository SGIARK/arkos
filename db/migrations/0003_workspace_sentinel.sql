-- The sentinel: a flush may only commit against a workspace that proves it was
-- materialized.
--
-- `materialize` writes a nonce into the box and records it here; `flush` refuses
-- to commit unless the two agree. Without it, a box that died between the two
-- comes back empty, the sweep finds nothing, and the commit replaces the
-- project's tree with no rows at all — a deletion of everything, recorded as a
-- clean flush.

ALTER TABLE session_sandboxes ADD COLUMN IF NOT EXISTS workspace_nonce TEXT;
