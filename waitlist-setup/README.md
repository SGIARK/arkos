# Waitlist setup

How the trybuddynow.com landing page stores signups in Supabase Postgres.

## Shape

```
browser  --POST /api/waitlist-->  Vercel function  --INSERT-->  Supabase Postgres
                                  (waitlist_writer, insert-only)
```

The browser never talks to Postgres and never holds a credential. The function
holds one connection string for a role that can `INSERT` into `public.waitlist`
and do nothing else — it cannot read the list back, and cannot see any other
table.

## First-time setup

### 1. Create the table

Run `001_waitlist.sql` as the `postgres` owner (Supabase dashboard → SQL Editor).
Idempotent, safe to re-run.

### 2. Create the writer role

```bash
cp 002_waitlist_role.sql.example 002_waitlist_role.sql
# replace REPLACE_WITH_A_LONG_RANDOM_SECRET (both occurrences) with a real secret
```

Generate one that needs no percent-encoding — alphanumeric only. A password with
`@`, `%`, `$` or `*` in it must be percent-encoded inside a connection URI, which
is a reliable source of "password authentication failed" at 2am:

```bash
python3 -c "import secrets,string;a=string.ascii_letters+string.digits;print(''.join(secrets.choice(a) for _ in range(40)))"
```

Run the file as the `postgres` owner. `002_waitlist_role.sql` is gitignored —
keep it out of the repo.

### 3. Set the Vercel environment variable

Project → Settings → Environment Variables, for all environments:

| Name | Value |
|---|---|
| `WAITLIST_DATABASE_URL` | `postgresql://waitlist_writer.<PROJECT_REF>:<SECRET>@<POOLER_HOST>:6543/postgres` |

Two things that are easy to get wrong:

- **Use the pooler host, not `db.<ref>.supabase.co`.** The direct host resolves to
  IPv6 only, and Vercel functions have no IPv6 egress — the direct string works
  from your laptop and fails on deploy. Find the pooler host under Project →
  Connect → Transaction pooler.
- **The pooler username is `role.projectref`**, not `role`. Supavisor uses the
  suffix to route to the right tenant.

Then redeploy. Vercel installs `pg` from `vercel-landing/package.json` — there is
no build step.

### 4. Optional: failure alerts

If a signup cannot be stored, the visitor sees "try again later" and can retry,
but a lost signup has no second chance. Set all three to get an email when an
insert fails:

| Name | Value |
|---|---|
| `RESEND_API_KEY` | Resend API key |
| `ALERT_EMAIL_FROM` | a sender on a domain verified in Resend |
| `ALERT_EMAIL_TO` | where alerts go |

Throttled to one email per 15 minutes per instance (`ALERT_COOLDOWN_MINUTES`), so
an outage cannot flood the inbox. Unset means alerting is skipped and the
endpoint works normally.

## Reading the list

`waitlist_writer` cannot read the table, by design. Read it as the owner:

```sql
select email, created_at from public.waitlist order by created_at desc;
```

## Rotating the secret

Re-run `002_waitlist_role.sql` with a new secret (the `alter role` branch handles
an existing role), then update `WAITLIST_DATABASE_URL` in Vercel and redeploy.

## Notes for whoever touches this next

- **The insert is `on conflict do nothing` with no conflict target.** Naming the
  target — `on conflict (email) do nothing` — makes Postgres inspect the arbiter
  index, which needs `SELECT`, which this role deliberately does not have. The
  targeted form fails with `permission denied for table waitlist` on every
  duplicate signup while fresh signups keep working, so it breaks quietly.
- **The Supabase root CA is pinned in `api/waitlist.js`.** The pooler presents a
  cert from Supabase's own CA, which is not in Node's default trust store. The
  common workaround is `rejectUnauthorized: false`, which leaves the connection
  encrypted but impersonatable. The pinned cert expires **April 2031**.
- **The endpoint returns 204 for both a new signup and a duplicate**, and never
  reads back, so it cannot be used to test whether an address is on the list.
- **A previous revision granted `anon` INSERT** so the function could write via
  PostgREST. That is gone. If you re-enable PostgREST access, remember it exposes
  an insert path on the public internet that the pooler route does not.
