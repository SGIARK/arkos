-- The store is ONE flat namespace per user, and a folder is a top-level path
-- segment in it (Task 11.9).
--
-- `project_files` made a project the OWNER of a directory: the tree was keyed
-- by project id, the mount name came from `projects.slug`, and the files view
-- grouped its headers by project TITLE — so a rename renamed the filesystem's
-- headers while the schema promised the folder never moved. Two ideas were
-- fused that were never the same idea. A folder is where bytes live; a project
-- is a piece of work that reads and writes some of them.
--
-- After this migration:
--   * `files (user_id, path)` is the whole store. The folder is
--     `split_part(path, '/', 1)` — DERIVED, never a row, unique per user by
--     construction, and it exists exactly as long as a file exists under it.
--   * `project_folders (project_id, folder)` is the LINK. A project owns no
--     folder; it links any number of them, and deleting a project deletes its
--     links and nothing else. Files are never orphaned because they were never
--     owned.
--   * `session_claims` names `(folder, subpath, mode)`. A write claim leases
--     `folder:{user_id}:{name}`, so two projects writing DIFFERENT folders
--     never contend and the same folder still serializes.
--
-- BACKFILL. Every existing row lands at `{user_id, slug || '/' || path}`,
-- which is exactly where it was already mounted (`~/projects/<slug>/<path>`),
-- so nobody's file moves in the only sense a file can move — the bytes are
-- content-addressed blobs and this migration does not touch one. `projects.slug`
-- is unique per user, so the prefixing cannot collide. Row ids are preserved:
-- nothing references them across tables, but a surface holding a `file_id` from
-- before the deploy keeps working.
--
-- Do this PRE-LAUNCH. It is at its cheapest while there is no live user data.
--
-- Two clean-ups ride along at the end, because they are the same idea applied
-- to rows that already exist: a link is only written for a folder that is
-- really there, and the home session's shadow project is unmade.

CREATE TABLE IF NOT EXISTS files (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    path         TEXT NOT NULL,          -- 'triage/receipts/q3.pdf'; the folder
                                         -- is the first segment and is derived
    content_hash TEXT,                   -- sha256; the blob is at
                                         -- {prefix}/blobs/{hh}/{sha256}
    size         BIGINT NOT NULL,
    mtime        TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_files_path ON files (user_id, path);

-- The folder listing (`GET /folders`, the modal's checklist, `+ link`) reads
-- the first segment of every path this user has.
CREATE INDEX IF NOT EXISTS idx_files_folder ON files (user_id, split_part(path, '/', 1));

INSERT INTO files (id, user_id, path, content_hash, size, mtime, created_at)
SELECT f.id, p.user_id, p.slug || '/' || f.path, f.content_hash, f.size, f.mtime, f.created_at
  FROM project_files f
  JOIN projects p ON p.id = f.project_id
ON CONFLICT DO NOTHING;

-- The links. No `mode`: links ship write-only (per-folder read mode is claims'
-- to express and is not this card).
CREATE TABLE IF NOT EXISTS project_folders (
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    folder     TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (project_id, folder)
);

CREATE INDEX IF NOT EXISTS idx_project_folders_folder ON project_folders (folder);

-- Every project that existed held exactly one directory: its slug. Linking it
-- is what makes this migration a no-op for anything already running.
--
-- ONLY where that directory actually has files in it. A folder exists exactly
-- as long as a file exists under it, so linking the slug of an empty project
-- would put a link to a folder that IS NOT THERE — an empty pane, a claim that
-- mounts nothing, and a name that never appears in the folder picker because
-- the picker is derived from the files. A project with nothing in it links
-- nothing until someone links it something, which is what `+ link` is for.
INSERT INTO project_folders (project_id, folder)
SELECT p.id, p.slug
  FROM projects p
 WHERE EXISTS (
       SELECT 1 FROM files f
        WHERE f.user_id = p.user_id AND split_part(f.path, '/', 1) = p.slug
       )
ON CONFLICT DO NOTHING;

-- Claims name a FOLDER. The old column named a project, which was the same
-- thing only because a project held one directory.
ALTER TABLE session_claims ADD COLUMN IF NOT EXISTS folder TEXT;

UPDATE session_claims c
   SET folder = p.slug
  FROM projects p
 WHERE p.id = c.project_id AND c.folder IS NULL;

-- A claim whose project vanished has no folder to name and nothing to mount.
DELETE FROM session_claims WHERE folder IS NULL;

ALTER TABLE session_claims ALTER COLUMN folder SET NOT NULL;
ALTER TABLE session_claims DROP CONSTRAINT IF EXISTS session_claims_pkey;
ALTER TABLE session_claims DROP COLUMN IF EXISTS project_id;
ALTER TABLE session_claims ADD PRIMARY KEY (session_id, folder, subpath);

CREATE INDEX IF NOT EXISTS idx_session_claims_folder ON session_claims (folder);

-- The order the claims were GIVEN, which for the default case is the order the
-- project linked its folders in. It decides one thing and decides it exactly:
-- an approved `plan.md` lands in the FIRST folder the session writes. An
-- explicit column rather than a timestamp, because "first" must not depend on
-- how close together two inserts landed; alphabetical would have made it depend
-- on the folder's name, which is nobody's choice about where work goes.
-- Backfilled to 0: rows written before this have one folder each.
ALTER TABLE session_claims ADD COLUMN IF NOT EXISTS ord INT NOT NULL DEFAULT 0;

-- Leases rekey from `project:{id}` to `folder:{user_id}:{name}`. A lease is
-- live state with an expiry, not a record, so the old keys are dropped rather
-- than translated: whoever held one is holding it under a key nothing asks for
-- any more, and the next acquire is the correct one.
DELETE FROM resource_leases WHERE resource_key LIKE 'project:%';

-- Snapshots, rekeyed rather than parked. `snapshot_files.path` was
-- project-relative in exactly the way `project_files.path` was, so it gets the
-- same prefix and keeps meaning what it meant. The CAPABILITY is still absent —
-- `snapshot_project`/`restore_snapshot` were removed in 11.7.5 and nothing in
-- the tree reads these tables — but a dormant table holding paths from a
-- namespace that no longer exists is how a future restore writes a tree nobody
-- can find. `project_snapshots.project_id` still names a project that still
-- exists, so a restore would resolve the folder through `project_folders`.
UPDATE snapshot_files sf
   SET path = p.slug || '/' || sf.path
  FROM project_snapshots s
  JOIN projects p ON p.id = s.project_id
 WHERE s.id = sf.snapshot_id
   AND sf.path NOT LIKE p.slug || '/%';

-- The tree is the user's now. Dropping this rather than leaving it is the point
-- of the card: a second table that still says a project owns files is a second
-- path for code to grow back into.
DROP TABLE IF EXISTS project_files;

-- The home session's shadow project is UNMADE, not cleaned up.
--
-- It existed for one reason: a session needed a directory, and a project was
-- the only thing that could hold one. Nothing holds a directory now — folders
-- are the store's and projects link them — so the reason is gone and the row
-- goes with it. `Chat ▸ Chat` in the grid was never a piece of work; it was the
-- shape of a dependency that no longer exists.
--
-- Narrow on purpose. A project is unmade only when it is the home session's
-- own, holds no files, and no OTHER session ever ran in it. Anything else is
-- someone's work and is left exactly where it is. The session's `project_id` is
-- cleared first because the foreign key is NO ACTION, and clearing it is the
-- point rather than a step: the home chat has no project.
-- Three plain statements sharing this migration's transaction, rather than one
-- clever one: a data-modifying CTE would have to be forced to run by a dummy
-- predicate, and a clean-up that deletes rows is the wrong place to be subtle.
CREATE TEMP TABLE shadow_projects ON COMMIT DROP AS
SELECT s.id AS session_id, p.id AS project_id
  FROM users u
  JOIN sessions s ON s.id = u.home_session_id
  JOIN projects p ON p.id = s.project_id
 WHERE NOT EXISTS (
       SELECT 1 FROM files f
        WHERE f.user_id = p.user_id AND split_part(f.path, '/', 1) = p.slug
       )
   AND NOT EXISTS (
       SELECT 1 FROM sessions o WHERE o.project_id = p.id AND o.id <> s.id
       );

UPDATE sessions SET project_id = NULL
 WHERE id IN (SELECT session_id FROM shadow_projects);

DELETE FROM projects WHERE id IN (SELECT project_id FROM shadow_projects);

-- `projects.slug` survives ONLY as the default NAME for the folder a project
-- created with no links makes for itself. It is no longer "the folder", so the
-- uniqueness rule that described a flat `~/projects/` namespace no longer
-- describes anything: folder names are unique per user because they are
-- segments of a unique path, and that is enforced by idx_files_path. Keeping
-- an index that asserts a rule about a namespace it does not own is how a stale
-- invariant survives a rename.
DROP INDEX IF EXISTS idx_projects_slug;
