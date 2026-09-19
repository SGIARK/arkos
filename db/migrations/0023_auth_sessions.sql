-- A stolen cookie dies with the password (Task 12.2.5).
--
-- The session cookie is a self-signed JWT, so until now the server had no way
-- to take one back: signing out cleared the cookie in the browser doing the
-- signing out, and nothing else. A cookie lifted from a shared machine stayed
-- good for its full seven days, and the one thing a person does on noticing —
-- change the password — did not touch it. Recovery that cannot evict an
-- attacker is not recovery.
--
-- One row per live cookie, keyed by the `jti` the cookie carries. A request is
-- authenticated only if its jti is still here, so revoking is a DELETE:
-- per-session on sign-out, per-user on a password change.
--
-- ON DELETE CASCADE from users, because a deleted account must not leave live
-- sessions behind. `expires_at` mirrors the cookie's own exp so the sweeper has
-- something to prune by; the cookie is still refused on its own expiry, so a
-- row outliving its prune is dead weight rather than a hole.

BEGIN;

CREATE TABLE auth_sessions (
    jti         UUID        PRIMARY KEY,
    user_id     UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    issued_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at  TIMESTAMPTZ NOT NULL
);

-- Revocation is by user, and it is the whole point of the table.
CREATE INDEX idx_auth_sessions_user ON auth_sessions (user_id);
CREATE INDEX idx_auth_sessions_expiry ON auth_sessions (expires_at);

COMMENT ON TABLE auth_sessions IS
    'One row per live session cookie. Present = valid; a DELETE signs it out. '
    'Cookies minted before this table existed carry no jti and are refused.';

COMMIT;
