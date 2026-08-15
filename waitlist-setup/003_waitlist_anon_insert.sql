-- 003_waitlist_anon_insert.sql
-- Run AFTER 001_waitlist.sql (which creates the table and enables RLS).
-- This file lives outside vercel-landing/ on purpose: it must never be part of
-- the deployed static site.
--
-- WHY THIS DEVIATES FROM 001:
-- 001 revoked anon entirely, because the original threat model assumed the anon
-- key might be exposed in the BROWSER. On Vercel the key is held server-side in
-- the function (SUPABASE_WAITLIST_KEY) and never shipped to the client, so the
-- browser still cannot touch Postgres.
--
-- We grant anon INSERT-only rather than using the service_role key because
-- service_role bypasses RLS completely — a leaked service_role key would expose
-- users, token_store, conversation_context and memories. An insert-only anon
-- role cannot read the waitlist, cannot read anything else, and its worst case
-- is spam rows. That preserves the least-privilege intent of 002's
-- waitlist_writer role using the only auth PostgREST understands.
--
-- NOTE: this is the correct grant only because there is NO select policy below.
-- Never add one. Read the list as the owner, manually, per 002's comment.

grant usage  on schema public   to anon;
grant insert on public.waitlist to anon;

drop policy if exists waitlist_anon_insert on public.waitlist;
create policy waitlist_anon_insert on public.waitlist
  for insert to anon
  with check (true);

-- Deliberately absent: any select/update/delete policy for anon.
-- Verify the lockdown after running this (should return zero rows / permission
-- denied when executed with the anon key, e.g. via the REST endpoint):
--   select * from public.waitlist;
