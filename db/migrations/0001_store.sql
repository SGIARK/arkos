-- The store: bytes in object storage, the tree in Postgres (D27).
--
-- project_files stops describing a file in some other system and becomes the
-- tree itself. `storage_path` named a Supabase Storage layout nothing ever
-- wrote; `content_hash` replaces it, and the bytes live under
-- blobs/{hh}/{sha256} in the bucket named by config `store.bucket`.
--
-- Safe to run twice, and safe on a database carrying Task-8-era rows: those
-- rows have no path, so the backfill gives them one from `name`.

-- ------------------------------------------------------- the tree columns --
ALTER TABLE project_files ADD COLUMN IF NOT EXISTS path         TEXT;
ALTER TABLE project_files ADD COLUMN IF NOT EXISTS content_hash TEXT;
ALTER TABLE project_files ADD COLUMN IF NOT EXISTS size         BIGINT;
ALTER TABLE project_files ADD COLUMN IF NOT EXISTS mtime        TIMESTAMPTZ;

-- Rows written before the store existed carry `name` and `size_bytes`.
UPDATE project_files SET path  = COALESCE(path, name)             WHERE path  IS NULL;
UPDATE project_files SET size  = COALESCE(size, size_bytes)       WHERE size  IS NULL;
UPDATE project_files SET mtime = COALESCE(mtime, created_at)      WHERE mtime IS NULL;

ALTER TABLE project_files ALTER COLUMN path  SET NOT NULL;
ALTER TABLE project_files ALTER COLUMN size  SET NOT NULL;
ALTER TABLE project_files ALTER COLUMN mtime SET NOT NULL;
ALTER TABLE project_files ALTER COLUMN mtime SET DEFAULT now();

-- One row per path per project: the tree is a map, not a log.
CREATE UNIQUE INDEX IF NOT EXISTS idx_project_files_path ON project_files (project_id, path);

-- Superseded: `name` and `size_bytes` are the path's tail and `size`, and
-- `storage_path` pointed at a backend that was never built.
ALTER TABLE project_files DROP COLUMN IF EXISTS storage_path;
ALTER TABLE project_files DROP COLUMN IF EXISTS name;
ALTER TABLE project_files DROP COLUMN IF EXISTS size_bytes;

-- ------------------------------------------------------------- the claims --
-- What a session may see and what it locks (D29). Declared at creation; the
-- sole source of both lease acquisition and sandbox contents.
CREATE TABLE IF NOT EXISTS session_claims (
    session_id  UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    project_id  UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    subpath     TEXT NOT NULL DEFAULT '/',
    mode        TEXT NOT NULL CHECK (mode IN ('read', 'write')),
    PRIMARY KEY (session_id, project_id, subpath)
);

CREATE INDEX IF NOT EXISTS idx_session_claims_project ON session_claims (project_id);
