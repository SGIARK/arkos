-- users.id IS a Supabase account, enforced (Task 12.2.5.5).
--
-- `schema.md` has always said `users.id` = the Supabase auth `sub`. Nothing
-- enforced it, and a stated invariant with no constraint behind it drifts: the
-- live database reached 775 rows of which 3 had a matching account. The rest
-- came from self-signed test tokens, which verify fine and upsert a real row for
-- a `sub` Supabase has never heard of.
--
-- The ongoing hazard was worse than the tidiness. Deleting an account in the
-- Supabase dashboard left our row and everything under it — sessions, files,
-- memory, connections — retained forever with no owner able to reach it.
--
-- THE CASCADE CHAIN HAD TWO BREAKS. `projects.user_id` and
-- `user_connections.user_id` were ON DELETE NO ACTION, so deleting a user did
-- not cascade, it FAILED. Both become CASCADE here, or the guarantee this
-- migration exists to make would stop dead one hop in.
--
-- CASCADE IS THE BACKSTOP, NOT THE PATH. Dropping a `user_connections` row does
-- not revoke anything at Composio — the app's own disconnect deletes the grant
-- there FIRST and then forgets the row, and that stays the normal flow. A
-- cascade reaching these rows means an account was deleted out from under us,
-- and it is better to lose the row than to keep a dangling one. Reconciling
-- grants that outlive their row is its own card.

BEGIN;

-- A stub for databases that are not Supabase — the test database has no `auth`
-- schema, and the FK has to resolve there too or test and production differ in
-- exactly the way that let this drift happen. On Supabase this is a no-op.
DO $$
BEGIN
    IF to_regclass('auth.users') IS NULL THEN
        CREATE SCHEMA IF NOT EXISTS auth;
        CREATE TABLE auth.users (id UUID PRIMARY KEY);
        COMMENT ON TABLE auth.users IS
            'Stub for non-Supabase databases so users_id_is_a_supabase_account resolves. '
            'On Supabase this is GoTrue''s own table and this migration leaves it alone.';

        -- Only on the stub: keep the FK satisfiable for a harness that mints its
        -- own tokens. On Supabase the account genuinely exists first, so nothing
        -- like this is wanted or created.
        CREATE FUNCTION auth.ensure_stub_account() RETURNS TRIGGER AS $fn$
        BEGIN
            INSERT INTO auth.users (id) VALUES (NEW.id) ON CONFLICT DO NOTHING;
            RETURN NEW;
        END;
        $fn$ LANGUAGE plpgsql;

        CREATE TRIGGER users_stub_account
            BEFORE INSERT ON public.users
            FOR EACH ROW EXECUTE FUNCTION auth.ensure_stub_account();
    END IF;
END $$;

-- The two breaks in the chain.
ALTER TABLE projects DROP CONSTRAINT IF EXISTS projects_user_id_fkey;
ALTER TABLE projects
    ADD CONSTRAINT projects_user_id_fkey
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;

ALTER TABLE user_connections DROP CONSTRAINT IF EXISTS user_connections_user_id_fkey;
ALTER TABLE user_connections
    ADD CONSTRAINT user_connections_user_id_fkey
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;

-- The invariant itself.
ALTER TABLE users
    ADD CONSTRAINT users_id_is_a_supabase_account
    FOREIGN KEY (id) REFERENCES auth.users(id) ON DELETE CASCADE;

COMMENT ON CONSTRAINT users_id_is_a_supabase_account ON users IS
    'users.id IS the Supabase auth sub. Deleting the account deletes this row '
    'and everything under it.';

COMMIT;
