# Landing page deploy — ark.mit.edu/landing

Serve the marketing/waitlist page at `https://ark.mit.edu/landing`, capture
emails into a new Supabase (Postgres) table, and take the authenticated `/app`
offline for the public launch.

**Design in one line:** Caddy serves the static page; the browser posts the
email to a tiny loopback service that holds a *low-privilege* DB role; the
browser never sees a database key. This matters because the waitlist table lives
in the same database as sensitive ARKOS data (OAuth tokens, users, memories).

```
browser ──HTTPS──> Caddy ──/landing──────> static index.html
                    │
                    └──/landing/api/*──> 127.0.0.1:8899 (waitlist svc)
                                              │ role: waitlist_writer (INSERT-only)
                                              ▼
                                         Supabase Postgres: public.waitlist
```

## Package contents

| Path | What it is |
|---|---|
| `migrations/001_waitlist.sql` | Creates `public.waitlist`, enables RLS, revokes anon/authenticated. |
| `migrations/002_waitlist_role.sql.example` | Creates the INSERT-only `waitlist_writer` role + policy. Copy, set a secret, run. |
| `caddy/Caddyfile` | Serves `/landing`, proxies `/landing/api/*`, **disables `/app`**, security headers + CSP. |
| `service/waitlist.py` | ~90-line FastAPI intake: honeypot, per-IP rate limit, email validation, no read-back. |
| `service/requirements.txt` / `.env.example` / `waitlist.service` | Deps, config template, hardened systemd unit. |
| `site/index.html` | Pre-built landing page (source `Landing.html` + honeypot + real submit). |
| `scripts/build-site.py` | Rebuilds `site/index.html` from `Landing.html` (don't hand-maintain two copies). |
| `scripts/deploy.sh` / `smoke-test.sh` | Publish + reload Caddy; post-deploy checks. |

---

## Auth / security review (READ THIS)

Opening a public, unauthenticated write path into a database that also holds
secrets is the risky part. The threats and how this package handles them:

1. **Never expose a Supabase key in the browser.** The anon/service key would be
   visible to anyone viewing source. Because this DB also holds `token_store`
   (OAuth tokens), `users`, `conversation_context`, and `memories`, a leaked key
   plus any missing RLS anywhere = full compromise. **Mitigation:** the key never
   leaves the server; the browser only sees `/landing/api/waitlist`. If you ever
   choose direct-to-Supabase instead, use a **separate Supabase project** for the
   waitlist, never the one holding ARKOS data.

2. **Least privilege at the DB.** The service authenticates as `waitlist_writer`,
   which has `INSERT` on `waitlist` and nothing else — no `SELECT` (can't read the
   list), no access to other tables. Even a fully compromised intake service
   cannot read emails or reach ARKOS data. (migration 002)

3. **No read-back / no enumeration.** The endpoint returns `204` on success *and*
   on duplicate (`ON CONFLICT DO NOTHING`), and there is no GET. An attacker can't
   probe whether an email is already registered, and the list is never served.

4. **RLS on, no anon policy.** Even if the anon key is exposed elsewhere, PostgREST
   can't touch `waitlist`: RLS is enabled with no policy for anon/authenticated and
   grants revoked. (migration 001)

5. **Bot spam / cost abuse.** A public POST is a spam magnet. **Mitigations:**
   hidden honeypot field (`company_website`), per-IP token-bucket limit in the
   service (default 5/60s), optional edge `rate_limit` in Caddy, and a body capped
   by the pydantic model. Add Cloudflare Turnstile/hCaptcha if abuse appears.

6. **Input validation + injection.** Email is validated and length-capped
   server-side (never trust the client `required`), normalized to lowercase, and
   inserted via a parameterized query ($1/$2) — no string-built SQL.

7. **Lock CORS + same-origin.** CORS allows only `https://ark.mit.edu` and POST;
   the page and API are same-origin so cross-site calls aren't needed. `form-action
   'self'` and `frame-ancestors 'none'` in the CSP block form hijack + clickjacking.

8. **TLS + headers.** Caddy issues/renews HTTPS automatically; HSTS, `nosniff`,
   `X-Frame-Options: DENY`, `Referrer-Policy`, and a tight CSP are set. The form
   only ever posts over HTTPS.

9. **THE BIG ONE — taking down `/app` is not just a Caddy edit.** Removing the
   `/app` route stops serving the UI, but the ARKOS FastAPI backend exposes
   `/tasks`, `/computer`, `/v1/chat/completions`, `/services` with **no public auth
   boundary**. If that process is still listening on a public interface, the
   internet can run agent tasks, drive the sandbox/browser, and read files.
   **You must also** stop the ARKOS service (`systemctl stop ark`) or bind it to
   `127.0.0.1` only, and confirm nothing proxies to it. Verify:
   `curl -I https://ark.mit.edu/app` → 404, and from off-box `nmap`/`curl` the
   app port is refused.

10. **Secrets hygiene.** `.env` is `chmod 600`, owned by the `arkwait` service
    user, never committed. The systemd unit runs unprivileged with
    `NoNewPrivileges`, `ProtectSystem=strict`, `ProtectHome`, empty capability set.

11. **PII.** Emails are personal data: don't log them in plaintext access logs,
    keep the "no spam" promise on the page, and be ready to delete on request.

Residual risks to accept or address later: distributed bot spam (needs a CAPTCHA),
and the honeypot being defeated by sophisticated bots (add Turnstile).

---

## Deploy steps

### 0. Prereqs
- DNS: `ark.mit.edu` → the server. Caddy installed. Python 3.11+ on the server.
- The Supabase Postgres connection info (project → Database → Connection string).

### 1. Database
```bash
# as the DB owner / via Supabase SQL editor or psql:
psql "$SUPABASE_OWNER_URL" -f migrations/001_waitlist.sql
cp migrations/002_waitlist_role.sql.example migrations/002_waitlist_role.sql
# edit 002: set a long random password for waitlist_writer
psql "$SUPABASE_OWNER_URL" -f migrations/002_waitlist_role.sql
```

### 2. Waitlist service
```bash
sudo useradd -r -s /usr/sbin/nologin arkwait || true
sudo mkdir -p /opt/ark-waitlist && sudo chown arkwait:arkwait /opt/ark-waitlist
sudo -u arkwait python3 -m venv /opt/ark-waitlist/venv
sudo -u arkwait /opt/ark-waitlist/venv/bin/pip install -r service/requirements.txt
sudo install -o arkwait -g arkwait -m 0644 service/waitlist.py /opt/ark-waitlist/waitlist.py
sudo install -o arkwait -g arkwait -m 0600 service/.env.example /opt/ark-waitlist/.env
# edit /opt/ark-waitlist/.env: WAITLIST_DB_URL uses waitlist_writer + the pooler host
sudo install -m 0644 service/waitlist.service /etc/systemd/system/ark-waitlist.service
sudo systemctl daemon-reload && sudo systemctl enable --now ark-waitlist
curl -fsS http://127.0.0.1:8899/landing/api/health   # {"ok":true}
```

### 3. Take down /app
```bash
sudo systemctl stop ark          # or reconfigure ARKOS to bind 127.0.0.1 only
# confirm the app port is not publicly reachable before continuing
```

### 4. Publish page + Caddy
```bash
REPO_LANDING=/path/to/landing ./scripts/deploy.sh
```

### 5. Verify
```bash
./scripts/smoke-test.sh
# and confirm the row landed (run as owner, NOT via the app):
psql "$SUPABASE_OWNER_URL" -c "select count(*) from public.waitlist;"
```

## Rollback
- Restore `/app`: re-enable the `handle /app*` block in the Caddyfile, restart the
  ARKOS service, `caddy reload`. Only do this once `/app` has a real auth boundary.
- Remove waitlist: `systemctl disable --now ark-waitlist`; the table can stay.
