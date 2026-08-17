-- 001_waitlist.sql
-- Public-intake table for the landing-page waitlist.
-- Lives in the SAME database as sensitive ARKOS tables (users, token_store,
-- conversation_context, memories), so it is locked down hard:
--   * RLS enabled, and NO policy for anon/authenticated  -> the Supabase anon
--     key (even if it leaked) cannot read or write this table via PostgREST.
--   * All grants revoked from anon/authenticated.
-- Writes happen ONLY through the dedicated low-privilege role in 002 that the
-- Vercel intake function uses. The browser never talks to Postgres.
--
-- Idempotent: safe to re-run.

create extension if not exists pgcrypto;  -- gen_random_uuid()

create table if not exists public.waitlist (
  id         uuid primary key default gen_random_uuid(),
  email      text not null unique,          -- store normalized (lowercased) email
  source     text not null default 'landing',
  user_agent text,
  created_at timestamptz not null default now()
);

alter table public.waitlist enable row level security;

-- Owner-side reads (see 002's admin read path) must not be silently filtered by
-- a future policy change, but writes from waitlist_writer must be. Postgres
-- exempts table owners from RLS unless FORCE is set, so leave FORCE off: the
-- owner reads everything, waitlist_writer is governed by the insert policy.

-- Belt and suspenders: strip any inherited PostgREST access.
revoke all on public.waitlist from anon, authenticated;
