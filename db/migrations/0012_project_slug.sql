-- A project's folder name is its own column, fixed at creation.
--
-- `workspace.claims_for` derived it from `projects.title` on EVERY turn:
--
--     slug=store.slug(r["title"], str(r["project_id"])[:8])
--
-- so renaming a project moved `~/projects/<slug>/` under the agent at its next
-- materialize — while the system prompt promises that path is the only durable
-- one it has. Every absolute path the model had learned silently stopped
-- resolving, and nothing said so. A title is a label; it was being used as an
-- identity.
--
-- The two are divorced from here on. A rename touches `title` alone. Moving the
-- folder to follow a rename would be offering to break the agent's paths as a
-- convenience — if the drift ever matters, the answer is to SHOW the folder
-- name, not to move it.
--
-- BACKFILL. The expression below is `store.slug` in SQL — lowercase, non
-- alphanumerics to hyphens, trimmed, capped at 48, falling back to the first 8
-- characters of the id. It reproduces exactly what each project's folder is
-- called on disk today, so applying this migration moves nobody's directory.
--
-- The row_number suffix handles what was already a live collision: two projects
-- with the same title both derived the same slug and therefore both claimed
-- `~/projects/<slug>/`. Oldest keeps the bare name; the others get `-2`, `-3`.
-- The unique index then makes that state unreachable rather than unlikely.

ALTER TABLE projects ADD COLUMN IF NOT EXISTS slug TEXT;

WITH derived AS (
    SELECT
        id,
        COALESCE(
            NULLIF(LEFT(TRIM(BOTH '-' FROM REGEXP_REPLACE(LOWER(title), '[^a-z0-9]+', '-', 'g')), 48), ''),
            LEFT(id::text, 8)
        ) AS base,
        user_id,
        created_at
    FROM projects
    WHERE slug IS NULL
),
numbered AS (
    SELECT id, base, ROW_NUMBER() OVER (PARTITION BY user_id, base ORDER BY created_at, id) AS n
    FROM derived
)
UPDATE projects p
   SET slug = CASE WHEN n.n = 1 THEN n.base ELSE n.base || '-' || n.n END
  FROM numbered n
 WHERE p.id = n.id;

ALTER TABLE projects ALTER COLUMN slug SET NOT NULL;

-- A DEFAULT, not just NOT NULL. Without one, every INSERT anywhere — including
-- code not yet written — has to know this column exists, and the one that
-- forgets fails at runtime rather than at review. The fallback is unique by
-- construction, so an insert that does not care about the folder name still
-- gets a folder nothing else can claim. `api._new_project` sets it properly.
ALTER TABLE projects ALTER COLUMN slug SET DEFAULT ('project-' || substr(gen_random_uuid()::text, 1, 8));

-- One folder per name per user. `~/projects/` is a flat namespace and two
-- projects cannot share a directory.
CREATE UNIQUE INDEX IF NOT EXISTS idx_projects_slug ON projects (user_id, slug);
