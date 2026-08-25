-- Per-session sandboxes: the box follows the session, capped per user.
--
-- The sandbox disk is a cache of the store (D27), so a box is not a user's
-- computer and there is nothing durable in it for a per-user row to name. One
-- row per session is both the e2b handle and the pool slot: the row count for a
-- user is what `sandbox.max_concurrent_per_user` caps, so the cap is taken and
-- released by the same writes that create and reap the box.
--
-- The old rows are dropped, not migrated: a per-user handle names a box no
-- session owns. Reap live sandboxes BEFORE running this, or they keep billing
-- until their own idle timeout.

DROP TABLE IF EXISTS user_sandboxes CASCADE;

CREATE TABLE IF NOT EXISTS session_sandboxes (
    session_id    UUID        PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    user_id       UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    sandbox_id    TEXT,                   -- e2b handle; NULL between the slot and the boot
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The cap counts a user's slots on every claim.
CREATE INDEX IF NOT EXISTS idx_session_sandboxes_user ON session_sandboxes (user_id);

-- A box belongs to one session, so there is nothing to serialize: the sandbox
-- is capacity now, and `resource_leases` holds only 'browser:{user}' and
-- 'project:{id}'.
DELETE FROM resource_leases WHERE resource_key LIKE 'sandbox:%';
