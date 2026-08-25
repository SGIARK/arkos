-- The snapshot tables go (Task 11.8.8).
--
-- 11.7.5 removed the capability: `snapshot_project`, `restore_snapshot`,
-- `list_snapshots` and `prune_snapshots` had no caller in the tree, and the
-- script the config comment named had never existed. 0015 then dutifully
-- rekeyed the dormant rows into the user namespace, which is the moment it
-- became clear what was being maintained — a correctly-migrated table for a
-- feature that does not exist.
--
-- Pre-launch the tables are empty, so this costs nothing. The DESIGN was sound
-- and git remembers the DDL and the code both: a snapshot is tree rows, which
-- is only true because blobs are immutable and never mutated. When snapshots
-- come back they get designed against the folder store natively rather than
-- inheriting a shape drawn for `project_files`, which no longer exists either.
--
-- `snapshot_files` first: it references `project_snapshots`.

DROP TABLE IF EXISTS snapshot_files;
DROP TABLE IF EXISTS project_snapshots;
