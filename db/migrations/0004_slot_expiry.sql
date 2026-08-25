-- Slots expire, as the lease they replaced did.
--
-- A row is written before the box boots and dropped when it is reaped, so a
-- process that dies mid-run leaves a row holding capacity no session is using.
-- `expires_at` is renewed on every call into the box; once it passes, the slot
-- is reclaimable and the box it names is killed with it.

ALTER TABLE session_sandboxes
    ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ NOT NULL DEFAULT now() + interval '15 minutes';

-- The reclaim sweep reads this, and the cap counts under it.
CREATE INDEX IF NOT EXISTS idx_session_sandboxes_expiry ON session_sandboxes (expires_at);
