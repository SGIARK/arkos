# Auth Spec

Companion to `docs/contracts.md` (which states the invariants the system relies
on) and the redesign spec. Scope: identity, authorization, trust boundaries.
**Gate:** `/app` does not go public until everything in "Rollout gate" is done.

**Status:** Built, except the rollout gate's last item | **Author:** John Wallace |
**Last updated:** 2026-08-18

---

# The problem, plainly

Token *verification* works today (signed JWTs, checked on every request, queries
scoped to the token's user). Token *issuance* is broken: `POST /auth/demo-login`
hands a valid token to anyone who types a username — no password, no email.
Anyone who can reach the server can become any user, including their connected
OAuth services (Google, Linear...) through the agent. Plus: `ARK_DEMO_MODE`
allows tokenless header impersonation, and CORS is `*`.

---

# Identity: Supabase Auth (GoTrue) — plugin, not hand-rolled

Already in the stack (same Supabase project). It owns signup, login, password
reset, and OAuth-social login if we ever want it.

- **Frontend:** logs in with `supabase-js`; posts that JWT ONCE to
  `/auth/session`; thereafter holds nothing — the cookie does the work. **Built
  in LG-1** (`frontend/api.jsx`): `signInWithPassword`, the token used once and
  never stored, `persistSession: false` so no second copy lives in localStorage,
  and `GET /auth/config` serving the project URL and publishable key because
  both differ per deployment while the page is a checked-in file.
- **Backend:** `jwt_utils.py` stops minting tokens and instead VERIFIES
  Supabase's JWT. Settled 2026-08-18 by checking the live project: it signs
  with an **ES256 key published at `/auth/v1/.well-known/jwks.json`**, so
  verification fetches that key rather than sharing a secret. The HS256
  `SUPABASE_JWT_SECRET` path remains for a project still signing with one, and
  the algorithm named in a token only selects among the mechanisms actually
  configured — an unlisted `alg` is refused, never trusted. `sub` claim → our
  `users` row
  (create-on-first-login, keyed by Supabase user id; username becomes profile
  data, not identity).
- **Deleted:** `demo-login`, `issue_token`, `ARK_DEMO_MODE` and the `X-User-ID`
  fallback. CORS pinned to the real origin with `allow_credentials=True` (see below).

# The session cookie (and why SSE needs nothing special)

`EventSource` cannot send an `Authorization` header, so a bearer design needs a
workaround for live streams. We avoid it entirely: after verifying Supabase's
JWT once, **we set our own session cookie** — httpOnly, Secure, SameSite=Lax.
The browser attaches it to every request including `EventSource`, so streams
authenticate with no token in the URL and `Last-Event-ID` reconnect works out of
the box. httpOnly also means XSS cannot read the session at all.

Bearer-only existed because wildcard CORS (`allow_origins=["*"]`, a demo
convenience for `file://` frontends) forbids credentials. In production the
frontend is same-origin, so that constraint is gone.

**Cost:** CSRF. Handled by SameSite=Lax plus an `Origin` check on every
POST/PATCH/DELETE. Non-browser clients (CLI, mobile) would need bearer
alongside — a small addition when one appears, not a redesign.

**Work (all done, 2026-08-18).** ~~pin CORS to the real origin +
`allow_credentials=True`~~ (Task 4); ~~read the token from a cookie in
`jwt_utils`~~; ~~set on login, clear on logout~~; ~~origin-check mutations~~;
~~delete localStorage token handling and `authHeaders()` from the frontend~~ —
closed by LG-1, which deleted `seed.jsx` with them and signs in through
supabase-js instead. `POST /auth/stream-token` was never built, as intended.

# Authorization invariants (what contracts.md relies on)

- Every query/subscribe/command is scoped to the verified user. No endpoint
  accepts a user id from headers, body, or query (the OAuth callback's
  `fallback_user_id` path is removed).
- Ownership checked on: sessions (snapshot, events, messages, cancel),
  projects + files, approvals (respond), blob refs (`read_result` — refs are
  unguessable UUIDs AND ownership-checked; unguessable alone is not access
  control), the browser frame stream (`GET /sessions/{id}/browser/frames`).
- Quotas: max concurrently running sessions per user (default 5), max new
  sessions per hour (default 20), upload size caps. Enforced in the
  harness at command time; exceeding returns the standard error shape.
- Grants and approvals are never shared across sessions (peer isolation).

# Trust boundaries (prompt injection + secrets)

Three classes of untrusted input reach the model: user suggestions (steering),
web content via `browser_task`, and MCP tool results. Stance:

- The model may be manipulated by any of them; the mitigation is NOT trusting
  the model's judgment but the tool boundary: write-effect tools carry
  `requires_approval` where consequential, budgets bound the blast radius, and
  the manifest bounds capability. A hijacked session can waste its budget; it
  cannot exceed its manifest.
- No secrets in event payloads: tool args are redacted against a denylist
  (tokens, passwords, connection strings) BEFORE `append()` — the log is
  durable and rendered in the UI. Credentials stay behind the Smithery
  vault+proxy and never enter the brain or sandbox (contracts.md).
- Sandbox env carries no credentials (verified in redesign Task 8).
- **A per-user MCP connection may never be written to `shared_connections`.**
  That table has no user column, and `_load(None)` hands every row to every
  user, so one person's OAuth grant landing there is a grant handed to the whole
  install. The gate is `requires_auth` in config: `false` writes shared, `true`
  writes `user_connections` and REFUSES when there is no `user_id` rather than
  falling back. The schema cannot express this (nothing links the two tables),
  so it is enforced only by `_owner_for()` and pinned by
  `test_a_per_user_server_without_a_user_is_refused_not_written_as_shared`.
  Anything that ever calls `connect()` with a fallback or anonymous user id
  re-opens this.
- **The persistent browser profile is credential-equivalent at rest.** Because
  the browser is leased per user and keeps its profile (D18), that directory
  holds live logged-in session cookies for whatever the user signed into. It is
  as sensitive as a password store: encrypted at rest, never copied between
  users, never surfaced in events, logs, or `tool_result` payloads, and deleted
  with the account. This is new sensitive state the pre-lease design did not
  have — a fresh-context-per-call browser had nothing to steal.

# Rollout gate (all boxes before /app is public)

Everything on this list is built except the last, so the gate now gates one
thing: public exposure.

1. ~~Supabase Auth wired; demo-login + ARK_DEMO_MODE deleted from the codebase.~~ Done — issuance is Supabase's, verification is `jwt_utils`, and LG-1 is the client.
2. ~~CORS pinned + credentials allowed; session cookie set on login; origin check on mutations.~~ Done in Task 4; `/app` is served same-origin so SameSite=Lax carries the cookie to `EventSource`.
3. ~~Ownership checks on every resource above; `test_authz_scoping` green.~~ Done.
4. ~~Quotas enforced.~~ Done.
5. ~~Redaction in the appender.~~ Done.
6. **Off-box verification: app port unreachable except via Caddy; landing's
   `steps/` hardening still in place.** The remaining item, and it is not a code
   change — it is checked from outside the box or it is not checked.

# Conformance

- `test_authz_scoping` — user A cannot read/steer/subscribe/resolve user B's
  anything (sessions, refs, files, approvals, streams).
- `test_token_issuance` — no code path issues a token without Supabase
  verification; grep-level check that demo-login/ARK_DEMO_MODE are gone.
- `test_cookie_session` — no cookie or a foreign cookie is rejected on every endpoint including both SSE streams; mutations reject a bad `Origin`.
- `test_a_per_user_server_without_a_user_is_refused_not_written_as_shared` — a
  `requires_auth` server never produces a `shared_connections` row.
- `test_one_users_connection_is_never_visible_to_another`,
  `test_a_tool_another_user_connected_is_not_callable` — MCP connections and
  their tools are scoped to their owner.

# Open questions

1. ~~Email+password vs magic-link-only at launch?~~ **Settled 2026-08-18
   (owner): email+password at v1, and no reset flow initially** — accounts are
   made and reset from the Supabase dashboard until there are real users to
   self-serve. Magic link stays available as one more GoTrue call whenever it
   earns its place; nothing in LG-1's sign-in view forecloses it.
2. Do we need refresh-token handling in the frontend beyond what supabase-js
   does automatically? (Likely no — verify token expiry UX under a long
   Looking Glass session.)
3. Team/multi-user projects are out of scope here — single-owner everything
   until a sharing spec exists.
