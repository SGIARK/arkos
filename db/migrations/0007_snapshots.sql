-- Snapshots: a saved copy of a project's tree rows.
--
-- Cheap because the bytes are content-addressed and immutable (D27): a snapshot
-- copies rows, never blobs, and restoring one points the tree back at blobs that
-- were never going anywhere. A hundred snapshots of a hundred-megabyte project
-- cost a hundred sets of rows.
--
-- This is why nothing deletes a blob. A blob GC would have to walk every
-- snapshot as well as every tree, so until one exists with that walk in it,
-- an orphan blob is storage we are choosing to spend.

CREATE TABLE IF NOT EXISTS project_snapshots (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id  UUID        NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    label       TEXT,
    taken_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_project_snapshots_project
    ON project_snapshots (project_id, taken_at DESC);

-- The tree as it stood, in the same shape `project_files` holds it.
CREATE TABLE IF NOT EXISTS snapshot_files (
    snapshot_id   UUID        NOT NULL REFERENCES project_snapshots(id) ON DELETE CASCADE,
    path          TEXT        NOT NULL,
    content_hash  TEXT        NOT NULL,
    size          BIGINT      NOT NULL,
    mtime         TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (snapshot_id, path)
);
