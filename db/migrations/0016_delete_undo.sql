-- Deleting from the store, and taking it back.
--
-- A delete is a ROW delete. Blobs are content-addressed and immutable and are
-- never collected (blob GC is still nobody's card), so the bytes of a deleted
-- file are all still there — which is what makes undo exact rather than a
-- best effort: the same content comes back under the same id, not a copy.
--
-- The removed rows go to `deleted_files` rather than staying in `files` behind
-- a `deleted_at` flag. A flag would put a soft-deleted row in the way of every
-- query that reads the tree, of `put_file`'s ON CONFLICT (user_id, path), and
-- of the unique index that makes folder names unique per user. A tombstone
-- table leaves `files` holding live rows and only live rows, which is the
-- property every other statement in the system already relies on.
--
-- `batch` is one delete GESTURE: a person pointed at one row and it took a file
-- or a whole subtree with it. Undo restores a batch, so it restores exactly
-- what that click removed.

CREATE TABLE IF NOT EXISTS deleted_files (
    -- The row's OWN id, kept so undo restores the file rather than replacing
    -- it: a surface holding a `file_id` across an undo still resolves.
    id           UUID PRIMARY KEY,
    user_id      UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    path         TEXT NOT NULL,
    content_hash TEXT,
    size         BIGINT NOT NULL,
    mtime        TIMESTAMPTZ NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL,
    batch        UUID NOT NULL,
    deleted_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_deleted_files_batch ON deleted_files (user_id, batch);
CREATE INDEX IF NOT EXISTS idx_deleted_files_when ON deleted_files (user_id, deleted_at DESC);

-- The links the same gesture had to drop.
--
-- A folder exists exactly as long as a file exists under it, so deleting the
-- last file under one takes the FOLDER with it — and a project linking a folder
-- that is not there is the dangling link this schema went out of its way to
-- avoid. The links go with the files and come back with them, in the same
-- batch, because they were removed by the same click and for the same reason.
CREATE TABLE IF NOT EXISTS deleted_links (
    batch      UUID NOT NULL,
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    folder     TEXT NOT NULL,
    PRIMARY KEY (batch, project_id, folder)
);
