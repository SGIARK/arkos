-- The memory region: the user's, not any project's (D29).
--
-- Its own table rather than a reserved project, because memory is keyed by user
-- and a project tree is keyed by project: putting it in `project_files` would
-- mean inventing a project that is not one. Nothing here decides whether memory
-- may ever be mounted — that is D30, open. Today no claim can name this table,
-- and settling D30 towards a read-only mount costs one additive migration
-- (a region column on `session_claims`), not a rebuild.
--
-- Paths are relative to `{user}/memory/`: `MEMORY.md` for the curated core,
-- `notes/<stamp>-<rand>.md` for one appended note.

CREATE TABLE IF NOT EXISTS memory_files (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    path          TEXT        NOT NULL,
    content_hash  TEXT        NOT NULL,
    size          BIGINT      NOT NULL,
    mtime         TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per path per user: the region is a map, as the project tree is.
CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_files_path ON memory_files (user_id, path);
