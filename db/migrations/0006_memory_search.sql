-- Memory becomes searchable: the text, where the query runs.
--
-- `search_memory` is a Postgres full-text query, so the words have to be in the
-- row. The blob stays the record of the bytes, as it is for every other file in
-- the store; `body` is the same text kept next to the index. They cannot drift:
-- a note is written once and never edited, and the core is replaced by one
-- statement under the advisory lock that guards it.
--
-- No vectors, no embeddings, no per-turn retrieval: the model searches when it
-- decides to, and reads what comes back.

ALTER TABLE memory_files ADD COLUMN IF NOT EXISTS body TEXT NOT NULL DEFAULT '';

-- The default exists only to add the column to rows that predate it. A row
-- written from here on carries its own text.
ALTER TABLE memory_files ALTER COLUMN body DROP DEFAULT;

-- Generated, so the index can never describe text the row no longer holds.
ALTER TABLE memory_files
    ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', body)) STORED;

CREATE INDEX IF NOT EXISTS idx_memory_files_tsv ON memory_files USING GIN (tsv);
