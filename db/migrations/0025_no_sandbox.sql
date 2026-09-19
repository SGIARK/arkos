-- This build has no sandbox, so the table that tracked one is dropped.
--
-- `session_sandboxes` held one row per session: the box handle, the pool slot
-- and the workspace nonce a flush checked itself against. Nothing creates a box
-- here and nothing transfers a tree into one, so every one of those columns
-- names a thing that no longer exists.
--
-- The earlier migrations that built it are left as they were: they are the
-- record of what ran, not a description of the schema today.

DROP TABLE IF EXISTS session_sandboxes CASCADE;
