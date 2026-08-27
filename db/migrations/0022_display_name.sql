-- What the buddy should call you (Task 12.1).
--
-- Sign-up asks for a name, because an assistant that has to address someone and
-- has only their email address addresses them as their email address. The name
-- is not an identity: `users.id` is the Supabase sub and `email` is the login.
-- This is a label, and the only thing that reads it is prose.
--
-- It arrives as Supabase user_metadata on the access token — `name` for a
-- password sign-up, which is what our own form writes, and `full_name` for
-- Google, which is what their OIDC profile carries. `POST /auth/session` reads
-- whichever is there. It is NOT read from the request body: the token is signed
-- and a body is not, and this column would otherwise be settable by anyone who
-- could reach the endpoint with a valid session.
--
-- NULLABLE, with no default and no backfill. Every user who predates this
-- column signed up when nothing asked for a name, so there is no name to write:
-- inventing one from the local part of their email would assert something this
-- migration cannot know, and "Nathaniel" and "nathaniel+buddy" are not the same
-- claim. NULL means nobody has said, and the caller falls back to the email —
-- which is exactly what it did before this column existed.
--
-- The upsert in `POST /auth/session` COALESCEs it the same way `email` is
-- handled, so a token that carries no metadata never blanks a name already set.
-- That matters for the mixed case: signing up with a password, then later
-- signing in through Google with the same address, must not erase what the
-- person typed.

BEGIN;

ALTER TABLE users ADD COLUMN display_name TEXT;

COMMENT ON COLUMN users.display_name IS
    'What the buddy calls this person. From Supabase user_metadata (name, or '
    'full_name from Google) at sign-in. NULL means nobody has said; fall back '
    'to email. A label, never an identity.';

COMMIT;
