# arkos — MIT Launch Readiness Plan (Legal, Trust & User Guidance)

Status: DRAFT v2 · Branch: `docs/launch-legal-spec` · Written against `main` @ `a522ff4`

> **Where the work is tracked now (v2, updated 2026-10-02).** This plan is the background:
> findings, risks, data inventory, content outlines, verified facts. **Execution**
> is in [`specs/README.md`](specs/README.md): nine `ready/` specs any agent can build
> now (deletion is one spec, `ready/02-deletion.md`) and two `drafting/` specs written with placeholders.
> **Decisions that block launch** live only in [`DECISIONS.md`](DECISIONS.md); the rest ship with defaults. §2 and §6 below are
> kept for context and point there.

This document is two things at once:

1. **A specification** of every file, page, piece of in-product copy, and backend
   capability arkos needs before it opens to the MIT student community, so that
   users are informed, consent is captured, promises are enforceable, and the
   operator's liability is limited.
2. **An implementation plan** that an AI coding agent (or a human) can execute
   PR by PR, with file paths, acceptance criteria, and tests.

> **Not legal advice.** Every legal text produced from this plan is a *draft for
> counsel review*. Nothing in section 9 (Go/No-Go) may be checked off until a
> lawyer — MIT's Office of the General Counsel if arkos is an MIT-operated
> service, otherwise the operator's own counsel — has reviewed the Terms, the
> Privacy Policy, and the Acceptable Use Policy.

---

## Table of contents

0. [Instructions for the executing agent](#0-instructions-for-the-executing-agent)
1. [Current state (audit findings)](#1-current-state-audit-findings)
2. [Decisions register (owner must answer)](#2-decisions-register-owner-must-answer)
3. [Risk register — what exposes the operator, and the mitigation](#3-risk-register)
4. [Data inventory, subprocessors, retention](#4-data-inventory-subprocessors-retention)
5. [Target architecture for legal & guidance content](#5-target-architecture)
6. [Implementation plan — PR by PR](#6-implementation-plan--pr-by-pr)
7. [Content specifications for each document](#7-content-specifications)
8. [Non-code workstream (owner / counsel / consoles)](#8-non-code-workstream)
9. [Go / No-Go launch checklist](#9-go--no-go-launch-checklist)
10. [Appendix: verified product facts for copywriting](#10-appendix-verified-product-facts)

---

## 0. Instructions for the executing agent

Read this section fully before touching code.

### 0.1 Ground rules

- **Coding standards:** `.github/CLAUDE.md` and `CLAUDE.md` at the repo root are
  authoritative. Key rules: functions under 40 lines, type hints on every
  signature, no bare `except`, no `print()`, no `TODO` in PRs, `async def` for
  anything that awaits, no blocking IO on async paths.
- **Branch names:** `type/short-description` (`feat/`, `fix/`, `docs/`, `chore/`,
  `test/`, `refactor/`). Each PR in section 6 names its branch.
- **PR size:** under 400 changed lines of *code*. Long legal/help HTML is content,
  but still split it as section 6 does (one or two documents per PR).
- **PR body:** follow `.github/pull_request_template.md`.
- **CI (run all four locally before claiming green):**
  ```bash
  ruff check .
  ruff format --check .
  mypy --follow-imports=silent --ignore-missing-imports --disable-error-code=arg-type \
      tool_module/tools/ tool_module/browser/tool.py tool_module/sandbox/tools.py
  DB_URL=postgresql://test:test@localhost:5432/test pytest tests/ -q --timeout=120 -m "not integration"
  ```
  Tests need a local Postgres with `python db/migrate.py` applied.
- **Tests mirror modules:** `harness_module/account.py` → `tests/test_account.py`.
  Name tests `test_[function]_[scenario]`. Mock every external dependency
  (Supabase admin API, Composio, e2b, OpenAI).
- **UI changes must be seen in a browser.** Run the API
  (`python -m uvicorn harness_module.api:app --port 1121`) and open
  `http://localhost:1121/app`. Legal and help pages are static and can also be
  checked with any static server pointed at `frontend/`.
- **Frontend cache-busting:** `frontend/index.html` loads every script and
  stylesheet with `?v=N` (currently `49`). Any PR that changes a `.jsx` or
  `.css` file MUST bump `N` on every tag in `index.html`.

### 0.2 Placeholders — never invent facts

Legal and help text will reference facts the owner has not decided yet (section 2).
**Do not guess them.** Use these exact tokens, and nothing else, for unknowns:

| Token | Meaning | Default if owner says "use defaults" |
|---|---|---|
| `{{OPERATOR_LEGAL_NAME}}` | The legal person/entity operating arkos | — (must be answered) |
| `{{OPERATOR_POSTAL_ADDRESS}}` | Mailing address for legal notices | — |
| `{{EFFECTIVE_DATE}}` | Date the document takes effect | launch date |
| `{{SUPPORT_EMAIL}}` | General help | — |
| `{{PRIVACY_EMAIL}}` | Privacy requests (access, deletion) | — |
| `{{SECURITY_EMAIL}}` | Vulnerability reports | — |
| `{{LEGAL_EMAIL}}` | Legal notices, DMCA | — |
| `{{GOVERNING_LAW}}` | Governing law & venue | Commonwealth of Massachusetts; courts of Suffolk or Middlesex County |
| `{{MIN_AGE}}` | Minimum age | 18 |
| `{{LLM_PROVIDER}}` | Model provider at launch | OpenAI, L.L.C. |
| `{{PUBLIC_URL}}` | Production origin | — |
| `{{SOURCE_URL}}` | AGPL source repository | `https://github.com/SGIARK/arkos` |
| `{{MIT_RELATIONSHIP_STATEMENT}}` | See decision D2 | see D2 |
| `{{HOSTING_PROVIDER}}` | Who runs the API and proxy | — (verify, not a decision) |

PR 16 adds a check that fails if any `{{` token remains in `frontend/legal/` or
`frontend/help/`. That check is a launch blocker, not a CI blocker, until launch.

### 0.3 Stop and ask the owner when

- A decision in section 2 is still open and the PR you are on depends on it
  *behaviorally* (e.g. the legal text needs D1). Copy can use placeholders;
  code that enforces a policy cannot.
- You find a user-owned table or storage location not listed in section 4 —
  update section 4 in the same PR and flag it.
- A deletion or migration step could destroy data in a non-test database.

### 0.4 Keep this document current

Each PR that completes a section-6 item edits this file: tick its checkbox in
section 6 and note any deviation in a "Deviations" line under that PR.

---

## 1. Current state (audit findings)

Audited on `main` @ `a522ff4` (2026-09-24).

### 1.1 What a visitor and a user see

| Surface | File | What exists | Gap |
|---|---|---|---|
| Sign-in / sign-up / reset | `frontend/views.jsx` (`Login` ~L571, `MailSent`, `ResetPassword`, `AuthAside` ~L954) | Pitch copy, email+password, Google sign-in | No Terms/Privacy links, no consent, no eligibility or age statement, no footer |
| App shell | `frontend/app.jsx` | Rail nav: desk · approvals · files · projects; theme toggle; settings | No help/guide, no legal links, no beta disclaimer, no "about" |
| Settings | `frontend/views.jsx` `SettingsModal` ~L374 | Connections (connect/disconnect with scopes), signed-in email, sign out | No delete-account, no data export, no legal/about section, no source link |
| Composer / session | `frontend/lookingglass.jsx` | Chat composer, session detail, tools toggles, files, live browser canvas | No AI-accuracy disclaimer, no explanation of what the agent can do |
| Approvals | `frontend/components.jsx` `ApprovalCard` ~L39 | Approve/decline gated tool calls | Copy does not say the user is authorizing the action as themselves |
| Files | `frontend/views.jsx` `ComputerView` | Upload / tree | No guidance on what must not be uploaded |
| Help / docs | — | **None** | Entire help center missing |
| Legal pages | — | **None** | Terms, Privacy, AUP, cookies, subprocessors, DMCA, security, accessibility, licenses all missing |

### 1.2 Backend facts that affect legal promises

- **Anyone with an email or Google account can sign up.** `POST /auth/session`
  (`harness_module/api.py` ~L287) upserts any verified Supabase user; there is no
  domain gate and no age attestation.
- **No consent is recorded.** The `users` table (`db/migrations/0000_migration_0.sql`
  ~L61, plus later columns) has no field for terms or privacy acceptance.
- **No account deletion endpoint; no data export endpoint; no per-session delete.**
  Routes are listed in `harness_module/api.py`; only `DELETE /auth/session`
  (sign-out), `DELETE /files`, and `DELETE /connections/{server}` exist.
- **Account deletion would fail today at the database level.**
  `sessions.user_id` and `user_sandboxes.user_id` (`0000_migration_0.sql` ~L90,
  ~L184) are `REFERENCES users(id)` with **no** `ON DELETE CASCADE`, and no later
  migration changes them. `0024_users_are_supabase_accounts.sql` fixed `projects`
  and `user_connections` but its comment ("the cascade chain had two breaks") missed
  these. `sessions.project_id → projects(id)` also has no cascade, so deleting a
  project with sessions fails too. Deleting a Supabase auth user (which cascades to
  `users`) will raise a foreign key violation for any user who has ever signed in
  (every user has a home session).
- **File bytes are global and content-addressed.** `harness_module/blobs.py` stores
  bytes at `{prefix}/blobs/{hh}/{sha256}` with no owner. The `Blobs` protocol has
  `put`, `get`, `missing` — **no `delete`**. Deleting a user's files today deletes
  tree rows only; bytes persist forever. A deletion promise requires a
  reference-counted blob delete.
- **Composio holds OAuth grants** for Gmail, Outlook, GitHub, Linear, Notion,
  Google Calendar, Google Drive. `0024`'s comment is explicit: a cascade on
  `user_connections` does **not** revoke the grant at Composio. Account deletion
  must call the app's existing disconnect path per connection first.
- **Retention that exists:** `system_log.retain_days: 30` (pruned at startup);
  session cookie 7 days (`auth.session_ttl_s`). **Undefined (kept forever):**
  sessions and their events, uploaded files, agent memory (`memory_files`),
  `deleted_files` undo records, approvals, result blobs.
- **Outbound third parties** (from `config_module/config.yaml` and `README.md`):
  OpenAI (`llm.base_url` default `api.openai.com`, `gpt-4.1-mini`), Supabase
  (auth + Postgres + storage bucket `arkos`), e2b (sandbox VMs), Browserless
  (headless browser, screencast frames), Composio (MCP gateway + OAuth grants),
  SerpAPI (web search on an app-level key). The frontend also loads React, ReactDOM,
  Babel and supabase-js from **unpkg.com** and fonts from **Google Fonts**, which
  exposes every visitor's IP address to those CDNs.
- **Agent autonomy:** unattended sessions run up to 15 hops / 30 minutes
  (`budgets.unattended`). Gated calls park for a human; the `tools.destructive`
  list (send email, forward, trash, share Drive file, merge PR, etc.) always parks,
  even in autopilot. `approvals.attended_auto_approve` is deliberately off.
- **Quotas:** 5 unattended sessions per user, 20 new sessions per hour, 25 MB per
  upload, 30 auth attempts per minute per IP, 5 concurrent sandboxes per user.
- **Password policy:** minimum 8 characters (`MIN_PASSWORD` in `views.jsx`). No
  MFA.
- **Security headers:** none set by the app (no CSP, no `X-Frame-Options` /
  `frame-ancestors`, no `Referrer-Policy`). The approvals UI is clickjackable in
  principle.
- **Licensing:** code is **AGPL-3.0** (`LICENSE`). AGPL §13 requires offering the
  corresponding source to users interacting over a network — the UI has no source
  link. `.github/CLA.md` is marked "Draft, pending legal review".
- **Accessibility:** several interactive controls are `<span onClick>` (e.g.
  `auth-forgot`, `auth-link` in `views.jsx`) and are not keyboard-reachable.

---

## 2. Decisions register (owner must answer)

> **Superseded by [`DECISIONS.md`](DECISIONS.md)**, which is the only place a
> decision is tracked or resolved. It adds D13–D19, pulled out of the specs, plus
> status, owner, and what each one unblocks. Since 2026-10-02 it lists only the
> decisions that block launch. The table below is the v1 snapshot,
> kept for the reasoning column. Don't resolve anything here.

| ID | Decision | Why it matters | Recommended default |
|---|---|---|---|
| D1 | Who is the operator (person, student group, LLC, MIT lab)? | The Terms are a contract with this party; an individual operator carries personal liability | Form an LLC (or operate under an MIT unit if D2 = official) **before** launch; see 8.1 |
| D2 | Relationship to MIT: official MIT service, MIT-lab research project, or independent student project? | Changes which policies govern (MIT IS&T/OGC vs. own), whether "MIT" may appear in branding, and whether FERPA applies | Independent project: state "arkos is not operated, sponsored or endorsed by MIT" and follow MIT's name-use policy |
| D3 | LLM provider at launch: OpenAI (hosted) or the self-hosted "MIT profile" (SGLang)? | Determines whether prompts/files leave to a third-party model provider — the largest privacy disclosure | Launch on OpenAI API with zero-data-retention requested (8.4); disclose clearly; update policy if switched |
| ~~D4~~ | ~~Eligibility: restrict sign-up to `@mit.edu`?~~ | ~~Scope of user base, cost control, audience-specific policy~~ | **Resolved 2026-09-25 (owner): no.** Sign-up stays open to any confirmed email or Google account. The launch is *aimed* at the MIT community, but not restricted to it. The eligibility-gate PR (6) is withdrawn; cost and abuse exposure moves to R11 and §8.9 |
| D5 | Minimum age | COPPA (<13) and contract capacity (<18); some MIT first-years are 17 | 18+, with a self-attestation checkbox |
| D6 | Pricing | Consumer-protection and refund terms if paid | Free beta; Terms reserve the right to introduce pricing with notice |
| D7 | Retention schedule (section 4.3) | The Privacy Policy must state it and the code must honor it | Adopt section 4.3 as written |
| D8 | Will user data be used for research, evaluation, or model training? | MIT research on human subjects requires COUHES (IRB) review; training use needs explicit opt-in | No research or training use at launch; any future use is opt-in with a new policy version |
| D9 | Who on the team may read user transcripts/files, and when? | Must be disclosed; insider access is a top breach vector | Only named operators, only for security/abuse/debugging a user-reported issue, logged (8.6) |
| D10 | Dispute resolution: arbitration clause + class-action waiver, or courts? | Liability exposure vs. student goodwill; enforceability varies | Counsel decides; default draft uses courts in `{{GOVERNING_LAW}}` with informal-resolution step first |
| D11 | Self-host frontend third-party assets (fonts, React, supabase-js)? | Removes IP disclosure to unpkg/Google; reduces supply-chain risk | Yes (PR 14) |
| D12 | Inactive-account policy | Data minimization | Accounts inactive 12 months are deleted after 30 days' email notice; all beta data deleted within 90 days if the service shuts down |

---

## 3. Risk register

Each risk maps to the PRs (section 6) and non-code items (section 8) that
mitigate it.

| # | Risk / exposure | Mitigation | Where |
|---|---|---|---|
| R1 | No enforceable agreement with users → no liability limits, no warranty disclaimer, no right to suspend | Terms of Service with clickwrap consent recorded per version | PR 3, PR 5 |
| R2 | Collecting personal data with no privacy notice (FTC Act §5 "unfair/deceptive"; Google OAuth policy violation) | Privacy Policy; accurate subprocessor list; honored retention and deletion | PR 2, PR 4, PR 7–9 |
| R3 | Privacy Policy promises the code can't keep (deletion, retention) → deceptive-practice exposure | Build deletion, export, retention jobs **before** publishing those promises | PR 7, 8, 9, 10, 15 |
| R4 | Agent sends an email / shares a file / merges a PR the user regrets | Terms allocate responsibility to the approving user; approval UI states it plainly; destructive list always parks; AUP | PR 3, PR 11 |
| R5 | AI output is wrong and relied upon (academic, medical, legal, financial) | Disclaimers in Terms and in-product under composer; no professional advice clause | PR 3, PR 11 |
| R6 | Academic integrity violations by students using arkos | AUP prohibits violating MIT academic integrity policies; help center page on responsible use | PR 3, PR 12 |
| R7 | Users upload regulated data (SSNs, health records, export-controlled or CUI research data, other students' education records) | AUP prohibited-data list; upload-area warning; Terms disclaim suitability | PR 3, PR 11 |
| R8 | Agent's browser used to scrape or access sites without authorization (CFAA, site ToS) | AUP; Terms make user responsible for sites the agent visits on their instruction; right to suspend | PR 3 |
| R9 | Copyright infringement in uploaded/produced content | DMCA policy + registered agent → §512 safe harbor | PR 4, 8.3 |
| R10 | Minors sign up (COPPA; contract capacity) | Age attestation at consent; Terms minimum age; policy section on children | PR 5, PR 2 |
| R11 | Open sign-up (D4) lets anyone in: runaway model/sandbox/search spend, throwaway abuse accounts | Existing per-user quotas (`quotas.*`, `sandbox.max_concurrent_per_user`) and required email confirmation; AUP "one account per person"; provider-side spend caps and alerts, because per-user quotas don't cap the total | §8.9, PR 3 |
| R12 | Data breach → Massachusetts 93H notification duties; 201 CMR 17.00 requires a written security program (WISP) | WISP, incident response runbook, security headers, `security.txt`, access policy | PR 13, PR 15, 8.6 |
| R13 | Account deletion fails silently or leaves orphaned data/grants | FK cascade fix; orphan test; Composio revocation; blob refcount delete | PR 7, PR 8 |
| R14 | Third-party OAuth app verification / policy (Google API Services User Data Policy, Limited Use; Microsoft publisher) | Privacy Policy Limited Use statement; consent screen links; verify Composio managed-app status | PR 2, 8.5 |
| R15 | AGPL §13 non-compliance (no source offer to network users) | Source link + running version in UI and on legal pages | PR 4, PR 1 |
| R16 | Implied MIT endorsement (trademark / MIT name-use policy) | Relationship statement (D2) in footer and Terms; no MIT seal/logo | PR 1, PR 3, 8.2 |
| R17 | Clickjacking the approval buttons | `frame-ancestors 'none'` / `X-Frame-Options: DENY` | PR 13 |
| R18 | Visitor IPs leaked to CDNs without disclosure | Self-host assets, or disclose | PR 14 / PR 2 |
| R19 | Inaccessible UI (MIT community expectations; ADA exposure) | Accessibility statement + fix keyboard traps in auth flows | PR 4, PR 12 |
| R20 | Using user data for research without IRB | D8 = no; COUHES before any change | 8.7 |
| R21 | Personal liability of the individual operator | Entity formation + insurance (tech E&O, cyber) | 8.1 |
| R22 | Service shuts down or loses data | Terms: no SLA, "back up your files", export tool, shutdown notice commitment | PR 3, PR 9 |
| R23 | Policy changes without notice | Versioned documents; re-consent on material change; changelog | PR 5, PR 1 |

---

## 4. Data inventory, subprocessors, retention

The Privacy Policy (PR 2) and the help page "Your data" (PR 12) are written
**from this section**. If code disagrees with this section, code or section must
change in the same PR.

### 4.1 Data inventory

| Category | Examples | Where stored | Source |
|---|---|---|---|
| Account | user id (Supabase `sub`), email, display name, password hash (Supabase only), Google profile name if Google sign-in | Supabase Auth; `users` | Sign-up |
| Auth sessions | session id (jti), expiry | `auth_sessions`; cookie `arkos_session` (HttpOnly, 7 days) | Sign-in |
| Consent records (new) | document, version, timestamp | `legal_acceptances` (PR 5) | Consent screen |
| Projects & sessions | titles, goals, mode, status | `projects`, `sessions` | User |
| Conversation content | prompts, agent replies, tool calls & results, todos, plans | `session_events`, `result_blobs`, blob store | User + agent |
| Approvals | the exact tool call proposed, the user's answer, timestamps | `approvals` | Agent + user |
| Files | uploaded files, files the agent writes | `files` tree rows + blob store (Supabase Storage bucket `arkos`); copied into e2b sandbox during runs | User + agent |
| Agent memory | `MEMORY.md` core and notes about the user | `memory_files` | Agent |
| Connected-account data | emails, calendar events, Drive files, GitHub/Linear/Notion content the agent reads while doing a task | Transiently in the model context; persisted where it lands in `session_events`/blobs | Composio → providers |
| Connection metadata | which toolkits are connected, tool lists | `user_connections`, `session_tools` | User |
| Browser activity | URLs visited, page content, screencast frames | Browserless container (transient); results in session log | Agent |
| Web searches | queries | SerpAPI; session log | Agent |
| Operational logs | errors, timings, benching events, request metadata | `system_events` (rows over 30 days dropped at each server start) | System |
| Client-side storage | settings in `localStorage`: theme (`arkos-theme`), files/browser tab (`arkos-canvas`); no Supabase token is stored (`persistSession: false`) | Browser | Frontend |
| Network metadata | IP address, user agent | Hosting logs / reverse proxy (Caddy), Supabase, CDNs | Every request |

### 4.2 Subprocessors (verify each before publishing)

| Provider | Purpose | Data | Notes for the agent / owner |
|---|---|---|---|
| Supabase | Authentication, Postgres database, file storage | All categories above | Record region of the project |
| `{{LLM_PROVIDER}}` (OpenAI by default) | Language model inference | Conversation content, file excerpts, connected-account data in context, memory | OpenAI API: not used for training by default; abuse-monitoring retention up to 30 days unless zero-retention is approved (8.4). Omit if D3 = self-hosted |
| e2b | Per-session sandbox VM (shell, files) | Files in claimed folders, commands, outputs | Boxes time out after 300 s idle |
| Browserless | Headless browser the agent drives | Sites visited, page content, screenshots | Self-hosted container or cloud — record which |
| Composio | OAuth gateway and tool execution for Gmail, Outlook, GitHub, Linear, Notion, Google Calendar, Google Drive | OAuth grants; the requests and responses of connected-account tools | Users grant access to Composio's managed OAuth apps |
| SerpAPI | Web search | Search queries | App-level key; queries not linked to user identity by SerpAPI, but are sent |
| Hosting provider (`{{HOSTING_PROVIDER}}`) | Runs the API and reverse proxy | All traffic | Add token to 0.2 table when known |
| Email sender (Supabase Auth SMTP or custom) | Confirmation and reset emails | Email address | Configure custom SMTP with branded sender (8.5) |
| unpkg.com, Google Fonts | Static assets | Visitor IP, user agent | Remove after PR 14 |
| PostHog (EU), via browser-use's built-in analytics | Library usage statistics | Browser task text, visited URLs, final result | Switched off by ready 10 Task 1 (`ANONYMIZED_TELEMETRY`, `BROWSER_USE_CLOUD_SYNC` = false); not a subprocessor once merged |

### 4.3 Retention schedule (proposed — decision D7)

| Data | Retention | Mechanism |
|---|---|---|
| Account, projects, sessions, events, files, memory | Until the user deletes them or deletes the account; inactive accounts kept (the sweep is deferred, post-launch) | PR 8, PR 10 |
| Deleted files (undo records in `deleted_files` / `deleted_links`) | 30 days, then purged along with unreferenced blobs | PR 10 purge job |
| Blobs | Deleted when no `files`, `result_blobs`, or other row references the hash | PR 8 refcount delete, PR 10 sweep |
| Auth sessions | Until expiry (7 days) or sign-out; swept by existing tick | Existing |
| System log | 30 days | Existing (`system_log.retain_days`) |
| Consent records | Life of account + 3 years (evidence of agreement) | Not cascaded — see PR 5 note |
| Composio grants | Until the user disconnects or deletes the account | Existing disconnect; PR 8 |
| Backups | Supabase backup window (record the actual value, typically 7–30 days) | Disclose in policy |
| After shutdown | All user data deleted within 90 days of service end, after 30 days' notice and export window | Commitment in Terms/Policy |

---

## 5. Target architecture

### 5.1 Where things live

```
frontend/
  legal/                    public, no sign-in, no JS required
    legal.css               shared styles for legal + help pages (reuse tokens from ../styles.css)
    index.html              "Legal" hub linking every document
    terms.html              Terms of Service
    privacy.html            Privacy Policy
    acceptable-use.html     Acceptable Use Policy
    ai-and-agents.html      How the agent acts on your behalf (disclosure)
    subprocessors.html      Third-party services
    cookies.html            Cookies & local storage
    copyright.html          DMCA / copyright policy
    security.html           Security overview + responsible disclosure
    accessibility.html      Accessibility statement
    licenses.html           AGPL source offer + third-party notices
    changelog.html          Version history of the legal documents
  help/                     public help center
    index.html
    getting-started.html
    projects-and-sessions.html
    approvals-and-autopilot.html
    connecting-accounts.html
    files-and-memory.html
    limits.html
    your-data.html
    responsible-use.html
    faq.html
    troubleshooting.html
  (existing app files)
docs/launch/
  LAUNCH_READINESS_PLAN.md  this file
  internal/
    WISP.md                 written information security program (201 CMR 17.00)
    INCIDENT_RESPONSE.md    breach runbook (M.G.L. c. 93H)
    DATA_ACCESS_POLICY.md   who can see user data, and how it is logged
    RETENTION.md            the schedule in 4.3 as operated
    LAUNCH_CHECKLIST.md     copy of section 9, tracked
SECURITY.md                 repo-level vulnerability reporting
```

The existing mount in `harness_module/api.py` (`app.mount("/app", _Frontend(...))`)
already serves `frontend/legal/*.html` at `/app/legal/*.html` with `no-store`.
PR 1 adds short, stable public URLs (needed for the Google OAuth consent screen and
for emails): `/terms`, `/privacy`, `/acceptable-use`, `/help`, `/legal`,
and `/.well-known/security.txt`.

### 5.2 Versioning model

- A `legal:` block in `config_module/config.yaml` is the single source of truth:
  ```yaml
  legal:
    terms_version: "2026-10-01"
    privacy_version: "2026-10-01"
    aup_version: "2026-10-01"        # AUP is incorporated into Terms; its version rides along
    min_age: 18
    source_url: "https://github.com/SGIARK/arkos"
  ```
- Each legal HTML page carries `<meta name="arkos-doc-version" content="...">`
  which a test (PR 5) asserts equals the config value, so the text users read and
  the version they accept cannot drift.
- A **material** change bumps the version → every user is shown the consent gate
  again on next load. A non-material change (typo) does not bump the version but
  is noted in `changelog.html`.

### 5.3 Consent flow

```
sign in (email or Google)
   → GET /auth/me returns consent: {required: bool, terms_version, privacy_version, accepted: {...}}
   → if required: <ConsentGate> (blocking full-screen)
        [ ] I am {{MIN_AGE}} or older
        [ ] I agree to the Terms of Service (incl. Acceptable Use Policy)
        [ ] I have read the Privacy Policy
        (links open in new tab)  [continue]  [sign out]
   → POST /auth/consent {terms_version, privacy_version, age_confirmed: true}
   → app loads
```

Server-side enforcement: a `ConsentedUser` dependency (wrapping `current_user`)
guards every endpoint that **creates work or ingests data** — `POST /sessions`,
`POST /sessions/{id}/messages`, `POST /files`, `POST /connections/{server}/connect`,
`POST /approvals/{id}/respond`, `POST /sessions/{id}/approve`, `POST /projects`.
It returns `403 consent_required`. Read endpoints, sign-out, export and delete stay
available so a user who refuses new terms can still leave with their data.

The sign-up form additionally shows "By creating an account you agree to the
Terms and Privacy Policy" with links — belt-and-braces, but the gate is the record.

---

## 6. Implementation plan — PR by PR

> **Superseded by [`specs/README.md`](specs/README.md)** for execution order. The
> PR list below is the v1 breakdown and stays as reference for content detail. Where
> it and a spec disagree, the spec wins: the specs were grounded later and fixed
> several details (see each spec's Technical Background).

Order matters: **promises ship after the capabilities that keep them.** PRs 7–10
(deletion, export, retention) must merge before PR 2's Privacy Policy is marked
final, even though PR 2 can be drafted in parallel.

Dependency graph:

```
PR 0 (this plan)
PR 1 scaffold ──┬─ PR 2 privacy ─┐
                ├─ PR 3 terms+AUP+AI
                ├─ PR 4 other legal pages
                └─ PR 12 help center
PR 5 consent (needs PR 1, PR 3 versions)
PR 6 withdrawn (D4 resolved: open sign-up)
PR 7 cascade fix ── PR 8 account deletion ── PR 9 export
                                   └──────── PR 10 session delete + retention jobs
PR 11 in-product disclosures (needs PR 1)
PR 13 security headers + security.txt
PR 14 vendor frontend assets (D11)
PR 15 internal policies
PR 16 launch gate (placeholder check, final QA)
```

---

### PR 0 — `docs/launch-legal-spec` — this plan
- [x] `docs/launch/LAUNCH_READINESS_PLAN.md`

---

### PR 1 — `feat/legal-pages-scaffold`

**Goal:** the skeleton every later content PR drops into, plus the links that make
it reachable.

Files:
- `frontend/legal/legal.css` — readable long-form layout: max-width ~72ch, 16px
  side gutter, headings, tables, anchors, print styles, light/dark via the same
  `data-theme` tokens as `frontend/styles.css` (link `../styles.css` then
  `legal.css`; do not duplicate tokens). Must work with JS disabled.
- `frontend/legal/_template.html` — reference skeleton: header with arkos mark
  and link back to `/app`, `<meta name="arkos-doc-version">`, "Effective
  {{EFFECTIVE_DATE}} · Version X", auto table of contents (static list), body,
  footer with the standard legal footer links + `{{MIT_RELATIONSHIP_STATEMENT}}`.
- `frontend/legal/index.html` — the hub listing every document with one-line
  descriptions.
- Stub pages for every file in 5.1 `legal/` with a title and "Draft — content in
  PR N" so links never 404.
- `frontend/help/index.html` stub.
- `config_module/config.yaml` — add the `legal:` block from 5.2.
- `harness_module/api.py` — redirect routes (307 → `/app/legal/*.html` etc.) for
  `/terms`, `/privacy`, `/acceptable-use`, `/legal`, `/help`. Keep each handler
  one line; register them **before** the static mount.
- `frontend/components.jsx` — `LegalLinks({ compact })` component: Terms ·
  Privacy · Acceptable Use · Help · Source. Links use `target="_blank"
  rel="noopener"` inside the app.
- `frontend/views.jsx` — render `LegalLinks` at the bottom of `auth-right` in
  `Login`, `MailSent`, `ResetPassword`; add an "about & legal" section to
  `SettingsModal` with the links, the running version, and "arkos is free
  software under the AGPL-3.0 — get the source" linking `legal.source_url`.
- `frontend/app.jsx` — add `help` link in the rail `foot` (opens `/help` in a
  new tab; do not add it to `NAV`).
- `GET /auth/config` (public) — add `legal: {terms_version, privacy_version,
  min_age, source_url}` and `build: os.environ.get("ARKOS_BUILD_SHA")`.
- Bump `?v=` in `frontend/index.html`.

Tests (`tests/test_api.py`):
- `test_legal_redirects_point_at_pages` — each short URL returns 307 to an
  existing file under `frontend/legal` or `frontend/help`.
- `test_auth_config_exposes_legal_versions`.

Acceptance: from a signed-out browser, every footer link resolves; from Settings,
the source link and version are visible; pages readable on a 375px-wide viewport.

---

### PR 2 — `docs/privacy-policy`

**Goal:** `frontend/legal/privacy.html` complete per section 7.2, written from
section 4. Also `frontend/legal/cookies.html` (7.6) and `subprocessors.html` (7.5)
since they are the policy's annexes.

Rules for the agent:
- Every statement of fact must be traceable to section 4 or section 10. If you
  need a fact that isn't there, verify it in code and add it to section 10 first.
- Use placeholders for undecided items; do not soften or omit a disclosure because
  a decision is open — write both branches in a clearly marked
  `<!-- DECISION D3: … -->` comment and render the default.
- Plain-language summary box at the top ("the short version"), 6–8 bullets.

Acceptance: counsel-review-ready draft; version meta matches config; no statement
contradicts section 4.

---

### PR 3 — `docs/terms-aup-ai-disclosure`

Files: `terms.html` (7.1), `acceptable-use.html` (7.3), `ai-and-agents.html` (7.4).

Acceptance: counsel-review-ready drafts; Terms incorporate AUP and Privacy Policy
by reference; limitation-of-liability and warranty-disclaimer sections in
conspicuous (bold/caps) text as is customary.

---

### PR 4 — `docs/legal-annex-pages`

Files: `copyright.html` (7.7), `security.html` (7.8), `accessibility.html` (7.9),
`licenses.html` (7.10), `changelog.html` (initial entry), `SECURITY.md` at repo root
(mirrors 7.8's reporting section).

For `licenses.html`, list each third-party asset the frontend loads with its
license: React & ReactDOM (MIT), @babel/standalone (MIT), @supabase/supabase-js
(MIT), JetBrains Mono (SIL OFL 1.1). Verify each license at the pinned version.
For Python dependencies, generate a list from `requirements.txt` with
`pip-licenses --format=markdown` (dev-only tool, do not add to requirements) and
include it as a table. Include the AGPL source offer from 7.10.

---

### PR 5 — `feat/legal-consent`

Files:
- `db/migrations/0025_legal_acceptances.sql`
  ```sql
  CREATE TABLE legal_acceptances (
      id            BIGSERIAL   PRIMARY KEY,
      user_id       UUID        NOT NULL,          -- deliberately NOT a cascading FK: see note
      document      TEXT        NOT NULL CHECK (document IN ('terms', 'privacy')),
      version       TEXT        NOT NULL,
      age_confirmed BOOLEAN     NOT NULL,
      accepted_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
      UNIQUE (user_id, document, version)
  );
  CREATE INDEX legal_acceptances_user ON legal_acceptances (user_id);
  ```
  Note: consent records are evidence and are retained after account deletion per
  4.3 (no personal content; just id, version, time). If counsel disagrees, change
  to `REFERENCES users(id) ON DELETE CASCADE`. Do **not** store IP or user agent.
- `harness_module/consent.py` — `current_versions() -> dict`,
  `async accepted(user_id) -> dict[str, str]`, `async needs_consent(user_id) -> bool`,
  `async record(user_id, terms_version, privacy_version, age_confirmed) -> None`
  (rejects versions that are not current with `409 stale_version`).
- `harness_module/api.py`:
  - `GET /auth/me` gains `consent: {required, terms_version, privacy_version}`.
  - `POST /auth/consent` body `{terms_version, privacy_version, age_confirmed}`;
    `age_confirmed` must be `true` → else `422`.
  - `ConsentedUser = Depends(consented_user)`; apply to the endpoints listed in 5.3.
- `frontend/api.jsx` — `api.consent(...)`.
- `frontend/views.jsx` — `ConsentGate({ me, onAccepted, onSignOut })`; reuse the
  auth layout (`AuthAside` + `auth-card`). Three checkboxes, continue disabled
  until all ticked. On a version bump the copy reads "We've updated our Terms /
  Privacy Policy — see what changed" linking `changelog.html`.
- `frontend/app.jsx` — after `user` is set, if `user.consent.required` render
  `ConsentGate` instead of the app. Handle `403 consent_required` in `onError` by
  re-fetching `me()`.
- Sign-up form: add the "By creating an account…" line with links.

Tests (`tests/test_consent.py` + `tests/test_api.py`), max 5 per function:
- `test_needs_consent_true_for_new_user`
- `test_record_rejects_stale_version`
- `test_consent_endpoint_requires_age_confirmation`
- `test_create_session_refused_without_consent` (403 `consent_required`)
- `test_version_bump_requires_consent_again`
- `test_legal_page_version_meta_matches_config` — parses each legal HTML file and
  compares `arkos-doc-version` to `legal:` config.

Existing tests that call guarded endpoints will now fail with 403 — add a fixture
in `tests/conftest.py` that records consent for the test user, and use it where
those tests set up a user. Do not weaken the guard to make tests pass.

---

### ~~PR 6 — `feat/eligibility-gate`~~ — withdrawn

D4 was resolved on 2026-09-25: sign-up is **not** restricted to `@mit.edu`. There
is no domain gate, no `auth.allowed_email_domains` key, and no Google `hd` hint.
Eligibility is only the minimum age, which is self-attested in the consent
gate (PR 5). Supabase's existing email confirmation is the only check on the
address. Cost and abuse exposure from open sign-up is handled in R11 and §8.9.
Do not reintroduce a domain gate without a new owner decision.

---

### PR 7 — `fix/user-delete-cascade`

**Goal:** deleting a `users` row removes everything under it, with no FK error.

- `db/migrations/0026_user_delete_cascade.sql`, single transaction:
  - `sessions.user_id` → `ON DELETE CASCADE`
  - `user_sandboxes.user_id` → `ON DELETE CASCADE`
  - Audit **every** FK that transitively references `users`, `projects`, or
    `sessions` (query `information_schema.referential_constraints` in the test
    DB) and fix any other `NO ACTION` link in the chain. List each change in the
    migration header comment. Known suspects: `sessions.project_id`,
    `session_events.session_id`, `approvals.session_id`, `session_tools`,
    `session_claims`, `session_sandboxes`, `result_blobs`, `project_files`,
    `project_folders`.
  - Do **not** change `legal_acceptances` (intentionally unlinked).
- Tests (`tests/test_store_schema.py`, which already inspects schema):
  - `test_user_delete_leaves_no_orphans` — create a user, then one row in every
    table that has a `user_id` column or chains to one (discover the list from
    `information_schema` so a future table is covered automatically), delete the
    user, assert zero rows remain for that user in each.
  - `test_every_user_chain_fk_cascades` — assert no FK on the chain has
    `delete_rule = 'NO ACTION'`.
- Correct the misleading comment in the header of the new migration (do not edit
  `0024`, migrations are append-only).

---

### PR 8 — `feat/account-deletion`

**Goal:** `DELETE /auth/me` removes the account and all its data, revokes all
third-party grants, and is safe to retry.

- `harness_module/blobs.py` — add `delete(content_hash)` to the `Blobs` protocol
  and both backends (filesystem: unlink; Supabase: storage `DELETE`). Treat
  "already absent" as success.
- `harness_module/account.py` (new), each step idempotent and logged to
  `system_log`:
  1. Mark the user as deleting (e.g. `users.deleting_at`; add in migration
     `0027_account_deleting.sql`) so new sessions/sign-ins are refused mid-way.
  2. Cancel every running or parked session (reuse the existing cancel path in
     `runner`/`lifecycle`; find it via `POST /sessions/{id}/cancel`).
  3. Release sandboxes (reuse `tool_module/sandbox/manager.py`).
  4. For each `user_connections` row, run the same revocation the
     `DELETE /connections/{server}` handler runs (Composio grant delete first).
     If Composio fails, stop and return `502` — do not delete the rows and orphan
     the grant.
  5. Collect the user's blob hashes (from `files`, `result_blobs`, `memory_files`
     if blob-backed, `deleted_files`), delete the rows, then for each hash delete
     the blob **only if no remaining row anywhere references it** (content
     addressing means two users can share a hash).
  6. Delete the Supabase auth user via the Admin API
     (`DELETE {SUPABASE_URL}/auth/v1/admin/users/{id}` with the service key) —
     the `users_id_is_a_supabase_account` FK cascades the rest. On a non-Supabase
     DB (tests), delete from `users` directly.
  7. Delete all `auth_sessions` and clear the cookie in the response.
- `harness_module/api.py` — `DELETE /auth/me` body `{"confirm_email": "..."}`
  which must equal the account email (case-insensitive) → else `422`. Returns
  `204` and clears the cookie. Not guarded by `ConsentedUser`.
- `frontend/views.jsx` `SettingsModal` — "delete account" in a danger zone:
  explains what is deleted and that it can't be undone, recommends export first
  (PR 9), requires typing the email, then signs out to the login screen with a
  notice "your account has been deleted".

Tests (`tests/test_account.py`), mock Composio, e2b, Supabase admin:
- `test_delete_account_removes_all_rows`
- `test_delete_account_keeps_blob_shared_with_other_user`
- `test_delete_account_stops_when_composio_revoke_fails`
- `test_delete_account_is_idempotent`
- `test_delete_me_requires_matching_email` (in `test_api.py`)

---

### PR 9 — `feat/data-export`

- `harness_module/account.py` `export_archive(user_id) -> AsyncIterator[bytes]`:
  a streamed ZIP containing
  - `account.json` (id, email, display name, created_at, consent records)
  - `projects.json`, `sessions.json`
  - `sessions/{session_id}.jsonl` (events, oldest first)
  - `approvals.json`
  - `memory/…` (memory core and notes as files)
  - `files/<path>` (the bytes of every current file)
  - `connections.json` (toolkit names and connected dates only — **never** tokens)
  - `README.txt` explaining the format
  Stream with `zipfile` in a thread or a streaming zip helper; do not buffer the
  whole archive in memory. Enforce a size ceiling (config
  `quotas.export_max_mb`, default 2048) with a clear error if exceeded.
- `GET /auth/me/export` → `application/zip`,
  `Content-Disposition: attachment; filename="arkos-export-YYYY-MM-DD.zip"`.
  Rate-limit to 1 per 10 minutes per user. Not consent-guarded.
- Settings: "download my data" button.

Tests: archive contains each section for a seeded user; no token-like fields in
`connections.json`; second request within the window is `429`.

---

### PR 10 — `feat/session-delete-and-retention`

- `DELETE /sessions/{session_id}` — deletes the session and its events,
  approvals, tools, claims; refuses (`409`) if running (user must stop first); the
  home session may be deleted (next sign-in recreates it, per `auth_me` docstring).
  Frontend: delete action in session detail with confirm.
- `DELETE /projects/{project_id}` if absent — same pattern.
- Retention jobs in `harness_module/lifecycle.py` (or wherever the existing
  startup/tick jobs live — `system_log` prune is the pattern to copy):
  - purge `deleted_files` / `deleted_links` older than
    `retention.deleted_files_days` (30);
  - sweep unreferenced blobs (same refcount rule as PR 8), batched;
  - (inactive-account sweep: after launch, `specs/ready/02-deletion.md` Task 6).
- `config_module/config.yaml` `retention:` block.

Tests: purge removes only rows past the window; blob sweep keeps referenced
hashes; running session delete refused.

---

### PR 11 — `feat/in-product-disclosures`

Copy changes only; each is one line or a small component. Use the exact strings
below unless the owner edits them.

| Where | File | Copy |
|---|---|---|
| Beta banner (dismissible per browser via `localStorage` key `arkos-beta-ack`) | `app.jsx` | "arkos is in beta. It can make mistakes and the service may change. Keep copies of anything important. [Learn more](/help)" |
| Under every composer | `lookingglass.jsx` `Composer` | "arkos can be wrong. It acts on your connected accounts only with your approval — check before you approve." |
| Approval card, above buttons | `components.jsx` `ApprovalCard` | "Approving lets arkos do this **as you**, using your connected account. It may not be reversible." For tools in `tools.destructive`, add: "This action can't be undone from arkos." (Expose the destructive list via `GET /auth/config` or the approval payload — check which is already available before adding.) |
| Before first connect of any toolkit | `views.jsx` `SettingsModal.connect` | Interstitial: "Connecting {name} lets arkos read and act in your {name} account through Composio, our integration provider, whenever a session you start has {name} turned on. Actions that send, share, delete or merge always ask you first. You can disconnect anytime here. [How connections work](/app/help/connecting-accounts.html)" → Continue / Cancel. Remember acknowledgment per toolkit in `localStorage`. |
| Files view header / upload zone | `views.jsx` `ComputerView` | "Don't upload passwords, government ID numbers, health records, other students' records, or export-controlled research data. [What not to upload](/app/help/responsible-use.html)" |
| Sign-up form | `views.jsx` `Login` | Consent line (PR 5) |
| Autopilot / unattended toggle (wherever mode flips) | `lookingglass.jsx` | "In autopilot, arkos keeps working without you for up to 30 minutes. It still stops and asks before anything in the destructive list." |

Also fix keyboard accessibility of `<span onClick>` controls touched here (convert
to `<button type="button">` styled as links).

---

### PR 12 — `docs/help-center`

Write each page in `frontend/help/` from section 10 facts only. Share
`legal/legal.css`. Every page ends with "Still stuck? {{SUPPORT_EMAIL}}".
Page specs in 7.11. Link the help center from: rail foot (PR 1), `Empty`
states in desk/projects (one "new here? start with the guide" link), and the
beta banner.

---

### PR 13 — `feat/security-headers`

- Middleware in `harness_module/api.py` adding to every response:
  - `Content-Security-Policy` — start in `Content-Security-Policy-Report-Only`
    for one release, then enforce. Allow only the origins the frontend actually
    uses (self; unpkg and Google Fonts until PR 14; the Supabase project URL for
    `connect-src`; Composio for the OAuth popup is a navigation, not a fetch).
    `frame-ancestors 'none'`. Note `@babel/standalone` needs `'unsafe-eval'`
    until the frontend is prebuilt — document that as a known gap.
  - `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`,
    `X-Content-Type-Options: nosniff`,
    `Permissions-Policy: camera=(), microphone=(), geolocation=()`.
  - `Strict-Transport-Security` only when `auth.cookie_secure` is true.
- `GET /.well-known/security.txt` (RFC 9116): `Contact: mailto:{{SECURITY_EMAIL}}`,
  `Expires`, `Policy: {{PUBLIC_URL}}/app/legal/security.html`,
  `Preferred-Languages: en`.
- Tests: headers present on `/app/` and on an API route; security.txt served.
- Verify in the browser that the app still loads and signs in with the CSP in
  report-only mode; read the console for violations and fix the policy.

---

### PR 14 — `chore/vendor-frontend-assets` (D11)

- Download the exact pinned versions currently in `frontend/index.html` into
  `frontend/vendor/` (react, react-dom, babel standalone, supabase-js) and the
  JetBrains Mono woff2 files into `frontend/vendor/fonts/` with the OFL license
  file. Verify each download's SHA-384 against the existing `integrity`
  attributes before committing.
- Point `index.html` and `legal.css` at the local copies; keep `integrity`.
- Remove unpkg and Google Fonts from the CSP (PR 13) and from the subprocessor
  list (PR 2); add a changelog entry.

---

### PR 15 — `docs/internal-policies`

Create `docs/launch/internal/` files per 7.12. These are operator documents, not
public pages. Include the retention schedule as operated and the launch checklist.

---

### PR 16 — `chore/launch-gate`

- `scripts/check_launch_ready.py` — exits non-zero if:
  - any `{{…}}` token remains under `frontend/legal/`, `frontend/help/`,
    `SECURITY.md`;
  - any legal page still contains "Draft";
  - `legal:` versions in config don't match page metas;
  - `auth.cookie_secure` is false.
- Add it to the Go/No-Go checklist (section 9), not to CI.
- Full manual QA pass (section 9.3) with screenshots attached to the PR.

---

## 7. Content specifications

Tone for all public pages: plain English, second person, short sentences, the
same lowercase-friendly voice as the app but **sentence case in legal documents**
for clarity. Each legal page opens with a "Short version" box that is explicitly
non-binding ("this summary isn't a substitute for the full text").

### 7.1 Terms of Service (`terms.html`)

1. **Agreement** — who the parties are (`{{OPERATOR_LEGAL_NAME}}`), acceptance by
   clickwrap, incorporation of the Privacy Policy and AUP.
2. **Eligibility** — `{{MIN_AGE}}`+; a confirmed email address; one account
   per person; accurate information. Open to anyone, not only MIT (D4).
3. **Beta service** — provided for evaluation; features may change or end; no
   SLA; operator may suspend or discontinue with notice where practicable
   (30 days for shutdown, with export window).
4. **Your account** — credential security; responsibility for activity under
   the account; notify of unauthorized use.
5. **What arkos does on your behalf** — the agent acts using tools and accounts
   you connect; **you are responsible for actions you approve** and for actions
   in categories you allow to run without approval; the operator is not a party
   to your dealings with third parties; the destructive-action safeguard is a
   convenience, not a guarantee.
6. **Connected services** — your use of Gmail, Outlook, GitHub, etc. remains
   subject to their terms; Composio acts as integration provider; you authorize
   access you grant and can revoke it.
7. **Your content** — you own your inputs and outputs (to the extent the law
   allows); you grant the operator a limited license to host, process, transmit
   and display your content **solely to provide the service**; no training or
   research use (per D8); you represent you have rights to what you upload.
8. **AI output** — may be inaccurate, incomplete, or offensive; not professional
   (legal, medical, financial, academic) advice; you are responsible for
   evaluating output before relying on or submitting it; compliance with your
   course and MIT academic integrity policies is your responsibility.
9. **Acceptable use** — incorporated AUP; violations may lead to suspension.
10. **Fees** — free during beta; notice before any charges (D6).
11. **Open source** — the software is AGPL-3.0; these Terms govern the hosted
    service, not your rights under the AGPL to the code.
12. **Intellectual property** — the service, name and marks; feedback license.
13. **Third-party services** — no responsibility for third-party services, sites
    the agent visits, or their content.
14. **Suspension and termination** — by you (delete account anytime); by the
    operator for AUP breach, legal risk, or abuse; effect on data (export window
    except where prohibited by law or abuse).
15. **Disclaimer of warranties** — "AS IS" and "AS AVAILABLE", conspicuous.
16. **Limitation of liability** — exclusion of indirect/consequential damages;
    aggregate cap (greater of amounts paid in 12 months or US$50); carve-outs
    required by law; conspicuous.
17. **Indemnification** — for your content, your approved actions, AUP breach,
    violation of third-party rights.
18. **Disputes** — informal resolution first (30 days via `{{LEGAL_EMAIL}}`);
    then per D10; governing law `{{GOVERNING_LAW}}`.
19. **Changes to these Terms** — versioned; material changes require re-acceptance
    in-product; changelog.
20. **Relationship to MIT** — `{{MIT_RELATIONSHIP_STATEMENT}}`.
21. **General** — entire agreement, severability, no waiver, assignment, notices,
    export control compliance (users must not use the service in violation of US
    export laws or from embargoed jurisdictions).
22. **Contact**.

### 7.2 Privacy Policy (`privacy.html`)

1. Short version box.
2. Who we are and how to contact us (`{{PRIVACY_EMAIL}}`, address).
3. **What we collect** — each category from 4.1, in plain words, grouped:
   information you give us; information arkos creates while working for you;
   information from services you connect; technical information.
4. **How we use it** — to provide the service (run your sessions, store files,
   remember context you asked it to); security and abuse prevention; support;
   operational debugging; communicating service changes. **Not** for advertising,
   selling, model training, or research (D8).
5. **Connected accounts & Google/Microsoft data** — what arkos can access once
   connected; only while a session with that connector is enabled; **Google API
   Services User Data Policy / Limited Use** statement verbatim-in-substance:
   use of information received from Google APIs adheres to that policy including
   the Limited Use requirements; no transfer to others except to provide the
   service, for security, to comply with law, or with consent; no human reading
   except with consent, for security, or as required by law.
6. **AI processing** — content you send (and content the agent reads from files,
   web pages and connected accounts) is sent to `{{LLM_PROVIDER}}` to generate
   responses; provider's retention terms; D3 branch.
7. **Who we share with** — subprocessors (link to page), legal requirements,
   business transfer, with your direction (e.g. emails the agent sends for you).
   No sale or "sharing" for cross-context advertising.
8. **Human access** — per D9.
9. **Retention** — table from 4.3.
10. **Your choices and rights** — access/export (self-serve), correction, deletion
    (self-serve account and session delete), disconnecting services, withdrawing
    consent, contacting us; response within 30 days; verification method.
    Brief sections for users in the EEA/UK (legal bases: contract, legitimate
    interests; right to complain to a supervisory authority) and California —
    keep, as MIT has international and out-of-state students.
11. **Security** — measures (encryption in transit, HttpOnly cookies, approval
    gates, sandboxed execution, access controls, logging); no system is perfectly
    secure; breach notification as required by law.
12. **Children** — not for anyone under `{{MIN_AGE}}`; deletion if discovered.
13. **Students and FERPA** — arkos is not an MIT education-records system (per D2);
    don't upload other students' education records; if D2 = official MIT service,
    counsel replaces this section.
14. **International transfers** — data processed in the United States.
15. **Cookies & local storage** — summary + link to cookies page.
16. **Changes** — versioning and notice.
17. **Contact**.

### 7.3 Acceptable Use Policy (`acceptable-use.html`)

You may not use arkos (or direct its agent) to:
- violate law, or MIT policies including the Academic Integrity policy and MIT's
  rules on use of IT resources (submitting AI work as your own where a course
  forbids it is your responsibility to avoid);
- access accounts, systems or data without authorization; bypass paywalls,
  CAPTCHAs, rate limits or technical protections; scrape in violation of a site's
  terms;
- send spam, bulk unsolicited messages, phishing, or impersonate others;
- harass, threaten, dox, or collect personal data about others without a lawful
  basis;
- create, upload or distribute malware, or run crypto-mining, DoS, or
  port-scanning from the sandbox or browser;
- infringe intellectual property;
- upload **prohibited data**: government ID numbers (SSN, passport), financial
  account numbers, health information (PHI), biometric data, other people's
  passwords, other students' education records, export-controlled data
  (ITAR/EAR), Controlled Unclassified Information, or data covered by a research
  data use agreement or IRB protocol that doesn't permit third-party AI
  processing;
- generate CSAM or sexual content involving minors; content that facilitates
  violence or self-harm;
- probe or attack arkos itself outside the responsible-disclosure policy;
- evade quotas, share accounts, or resell access;
- use outputs to develop competing foundation models where the model provider's
  terms prohibit it.
Also: enforcement (warning, suspension, termination, reporting to authorities
or to MIT where required), and how to report abuse (`{{SUPPORT_EMAIL}}`).

### 7.4 How arkos acts on your behalf (`ai-and-agents.html`)

- What an agent is here; attended vs. autopilot (unattended) sessions and their
  limits (6 steps / 5 min attended; 15 steps / 30 min unattended).
- Tools it has: sandbox shell and files, web search, browser, memory, and any
  connectors you enable per session.
- The approval model: gated calls park; you see the exact action; approving runs
  that exact call once; the always-ask list (render it from `tools.destructive`
  as human-readable names).
- Memory: what it remembers across sessions and how to view/edit/delete it.
- Limits of safeguards: approvals depend on you reading them; the agent can
  misunderstand instructions or content on web pages (prompt injection) — don't
  connect accounts you aren't comfortable having an assistant act in.
- How to stop a run.

### 7.5 Subprocessors (`subprocessors.html`)

Table from 4.2 (name, purpose, data, location, link to their privacy policy),
"last updated" date, commitment to update before adding a new one.

### 7.6 Cookies & local storage (`cookies.html`)

`arkos_session` (strictly necessary, HttpOnly, 7 days; the only sign-in state, since
Supabase runs with `persistSession: false`); `arkos-theme`, `arkos-canvas`,
`arkos-cookie-ack`, `arkos-beta-ack`, connect acknowledgments (preferences); no analytics or advertising cookies; third-party
CDN requests until PR 14. No consent banner is needed while only strictly
necessary/preference storage is used — state that, and require a banner if
analytics are ever added.

### 7.7 Copyright / DMCA (`copyright.html`)

Notice contact `{{OPERATOR_LEGAL_NAME}}`, `{{LEGAL_EMAIL}}`, postal address (the designated-agent registration in §8.3 is a checklist task); the
six §512(c)(3) notice elements; counter-notice procedure; repeat-infringer
termination policy.

### 7.8 Security (`security.html`) and `SECURITY.md`

Overview of measures (don't over-promise); how to report (`{{SECURITY_EMAIL}}`),
what to include, response targets (acknowledge 3 business days), scope/out of
scope, good-faith safe-harbor language, no bug bounty at this time.

### 7.9 Accessibility (`accessibility.html`)

Commitment to WCAG 2.1 AA, known gaps (list any from the PR 11/12 audit),
contact for accommodations or alternative formats.

### 7.10 Licenses & source (`licenses.html`)

AGPL-3.0 notice; "Source code for the version running here: `{{SOURCE_URL}}` at
commit `<ARKOS_BUILD_SHA>`" (render the sha client-side from `/auth/config`);
NOTICE text; third-party license table (PR 4).

### 7.11 Help center pages

| Page | Must cover |
|---|---|
| `index.html` | What arkos is, who it's for (beta, MIT), links to every page |
| `getting-started.html` | Sign up (email or Google, confirm email), first chat in the home session, create a project, attach folders, run a first task, approve a step |
| `projects-and-sessions.html` | Projects, sessions, home chat, attended vs autopilot, stop/resume, deleting a session |
| `approvals-and-autopilot.html` | Approvals view and pending pill, what a gated call is, always-ask actions, declining, what happens while waiting |
| `connecting-accounts.html` | The seven connectors, what "grants …" scopes mean, per-session tool toggles, tool-count cap (128 total), shared sign-ins (disconnecting one may disconnect siblings), how to disconnect, Composio's role |
| `files-and-memory.html` | Files view, folders, uploads (25 MB), undo delete (30 days), how files reach the sandbox (`~/store/<folder>/`), agent memory and how to review it |
| `limits.html` | Quotas from section 10, budgets per session, why a run might stop |
| `your-data.html` | Plain-language privacy summary, export, delete session, delete account, retention |
| `responsible-use.html` | Academic integrity, prohibited data, checking AI output, prompt-injection awareness |
| `faq.html` | 10–15 Q&As (is it free, does it train on my data, can it send email without me, what if it makes a mistake, who can see my stuff, how do I leave…) |
| `troubleshooting.html` | Confirmation email not arriving (spam, 60 s resend window), reset link expired, 403 on actions (consent not accepted), connector popup blocked, session stuck waiting |

### 7.12 Internal documents (`docs/launch/internal/`)

- `WISP.md` — 201 CMR 17.00 elements: responsible person, risk assessment,
  employee (team) policies, access controls and least privilege, encryption in
  transit (and at rest where provided by Supabase), vendor oversight (section 4.2),
  monitoring, annual review, training, disposal.
- `INCIDENT_RESPONSE.md` — roles, severity levels, containment steps for this
  stack (rotate `ARKOS_SESSION_SECRET` signs everyone out; revoke Supabase service
  key; revoke Composio key; kill e2b boxes), evidence preservation, M.G.L. c. 93H
  notification (affected MA residents, Attorney General, Office of Consumer
  Affairs and Business Regulation) "as soon as practicable and without
  unreasonable delay", other-state and GDPR 72-hour considerations, user
  communication template, post-mortem.
- `DATA_ACCESS_POLICY.md` — per D9: named people with production DB/storage
  access, when access is allowed, logging of each access, no copying data to
  personal devices, offboarding.
- `RETENTION.md` — section 4.3 as operated, with the job names that enforce each row.
- `LAUNCH_CHECKLIST.md` — section 9, to be ticked with names and dates.

---

## 8. Non-code workstream

The owner (and counsel) must complete these; the agent can draft emails/forms
but must not submit anything.

### 8.1 Entity, liability, insurance
- Decide D1. If individuals are operating, form an LLC (Massachusetts or
  Delaware) before launch so Terms bind the entity, not a person.
- Obtain quotes for technology errors & omissions and cyber liability insurance.
- Get the CLA (`.github/CLA.md`, marked draft) reviewed, since it defines "the
  Owner" who can relicense contributions.

### 8.2 MIT relationship (D2)
- If independent: confirm no MIT funds, equipment, or staff time create an MIT
  ownership claim (MIT's IP policy applies to significant use of MIT resources);
  follow MIT's name-use policy — don't use MIT logos/seal; "for the MIT
  community" is descriptive, "MIT's arkos" is not.
- If an MIT lab/official service: engage OGC and IS&T; MIT's policies and data
  classification then govern, and parts of this plan change.
- If the self-hosted "MIT profile" (SGLang on GPUs) is used, confirm where those
  GPUs live and under whose policies.

### 8.3 Copyright
- Register a DMCA designated agent with the US Copyright Office (online, small fee,
  renew every 3 years).

### 8.4 Model provider
- If OpenAI: accept the API business terms under the operator entity; apply for
  zero data retention if eligible; confirm the org's data-controls settings (no
  training on API data).

### 8.5 Console configuration
- **Supabase:** custom SMTP with a branded sender domain (SPF/DKIM/DMARC);
  email templates include links to Terms and Privacy; enable leaked-password
  protection; raise minimum password length to match the app; keep email
  confirmation ON (with open sign-up it is the only check on an address); record project region and
  backup retention for section 4.
- **Google Cloud OAuth consent screen (Supabase Google sign-in):** app name,
  logo, homepage `{{PUBLIC_URL}}`, privacy URL `{{PUBLIC_URL}}/privacy`, terms URL
  `{{PUBLIC_URL}}/terms`, authorized domain; submit for verification if required.
- **Composio:** confirm the managed OAuth apps' verification status for Gmail
  (restricted scopes) and Microsoft; note in the privacy policy that users will
  see Composio's app name on consent screens; review Composio's DPA/terms.
- **e2b, Browserless, SerpAPI:** accept terms under the entity; record data
  retention of each for section 4.2; sign DPAs where offered.
- **Contact inboxes:** create support@, privacy@, security@, legal@ on a domain
  the operator controls, with at least two people receiving each.

### 8.6 Operations
- Restrict production DB and storage access to named people (D9); enable
  Supabase audit logs; turn on MFA for every operator account (GitHub, Supabase,
  OpenAI, Composio, e2b, hosting, domain registrar).
- Rotate every secret that has ever been in a shared chat or `.env` on a laptop.

### 8.7 Research (D8)
- If anyone wants to study usage for a paper or class project: COUHES review
  **before** collecting, plus a separate opt-in consent. Not part of launch.

### 8.8 Counsel review
- Terms, Privacy Policy, AUP, AI disclosure, DMCA page, and the consent-gate copy.
  Record reviewer and date in `changelog.html`.

### 8.9 Spend and abuse limits (open sign-up, D4)
- Per-user quotas already exist (`quotas.new_sessions_per_hour: 20`,
  `quotas.max_unattended_sessions: 5`, `sandbox.max_concurrent_per_user: 5`,
  `quotas.auth_attempts_per_minute: 30` per IP). They bound one account, not the
  total. With sign-up open to anyone, set a hard monthly spend cap and a 50%
  alert on OpenAI, e2b and SerpAPI before launch.
- Watch sign-up volume for the first two weeks. If throwaway accounts appear,
  the fallbacks are CAPTCHA on Supabase Auth or an invite code, each a new owner
  decision. The domain gate is not one of them (D4).

---

## 9. Go / No-Go launch checklist

### 9.1 Blockers (all must be true)
- [ ] D1–D5, D7–D9 answered; `{{…}}` tokens filled (`scripts/check_launch_ready.py` passes)
- [ ] Counsel reviewed Terms, Privacy, AUP, AI disclosure (8.8)
- [ ] PRs 1, 2, 3, 5, 7, 8, 9, 10, 11, 13 merged and deployed (PR 6 withdrawn)
- [ ] Spend caps and alerts set on OpenAI, e2b and SerpAPI (§8.9)
- [ ] Deletion tested end-to-end on staging with a real Supabase user and a real
      Composio connection (grant disappears at Composio; blobs gone; auth user gone)
- [ ] Export tested on staging for a user with files, memory and ≥3 sessions
- [ ] Google OAuth consent screen shows privacy + terms links
- [ ] Contact inboxes live and monitored
- [ ] Operator entity exists and is named in Terms
- [ ] MFA on all operator accounts; production access list written (8.6)
- [ ] `auth.cookie_secure: true`, HTTPS only, HSTS on
- [ ] WISP and incident runbook written (PR 15)

### 9.2 Strongly recommended
- [ ] PR 12 help center, PR 4 annex pages, PR 14 vendored assets
- [ ] DMCA agent registered
- [ ] Insurance bound
- [ ] CSP enforced (not report-only)

### 9.3 Manual QA script (attach screenshots to PR 16)
1. Signed out: open `/`, `/terms`, `/privacy`, `/help`, `/.well-known/security.txt`.
2. Sign up with a non-MIT email → accepted after email confirmation (D4: open sign-up).
3. Confirmation email has branded sender and links.
4. First sign-in → consent gate; continue disabled until all boxes ticked.
5. Try `POST /sessions` via curl before consenting → 403 `consent_required`.
6. Accept → app loads; beta banner, composer disclaimer visible.
7. Connect Gmail → interstitial → Composio consent → connected.
8. Ask the agent to send an email → approval card shows "as you" copy and the
   irreversible warning.
9. Upload a file; delete it; confirm undo works; confirm the upload warning.
10. Bump `legal.terms_version` on staging → reload → consent gate reappears.
11. Export data → open the ZIP → every section present, no tokens.
12. Delete a session; delete the account (type email) → signed out with notice;
    sign in again with Google → fresh account, consent gate again.
13. Keyboard-only pass through sign-in, consent, settings, approval.
14. 375px-wide viewport pass on every legal and help page.

---

## 10. Appendix: verified product facts

Copywriters (human or agent) may state these without re-verifying. Anything not
listed must be verified in code and added here first. Source in brackets.

- Sign-in methods: email + password (minimum 8 characters), Google. Email
  confirmation is required. Resend/reset emails are limited to one per 60 seconds.
  [`frontend/views.jsx` `MIN_PASSWORD`, `RESEND_WAIT`]
- Session cookie `arkos_session`: HttpOnly, SameSite=Lax, Secure in production,
  7-day lifetime; sign-out revokes it server-side; a password change signs out
  everywhere. [`config.yaml` `auth`, `api.py` `/auth/session`, `/auth/sessions/revoke`]
- Every user gets a standing "Chat" home session. [`api.py` `_ensure_home_session`]
- Navigation: desk, approvals, files, projects; settings holds connections and
  account. [`frontend/app.jsx`]
- Connectors: Gmail, GitHub, Linear, Microsoft Outlook Mail, Notion, Google
  Calendar, Google Drive, all through Composio's managed OAuth apps; each is
  enabled per session; one sign-in can back several connectors.
  [`config.yaml` `mcp_servers`, `views.jsx` `SettingsModal`]
- Always-ask (destructive) actions include: sending, replying to and forwarding
  email (Gmail, Outlook); deleting/trashing mail, drafts, labels; deleting
  calendar events or removing attendees; editing, moving, trashing, or sharing
  Drive files; merging GitHub branches/PRs, updating refs, removing review
  requests; archiving Linear issues/projects; archiving/replacing/deleting Notion
  content. [`config.yaml` `tools.destructive`]
- All other connector tools require approval unless configured `auto_approve`
  (none are at present). [`config.yaml` `mcp_servers` comment]
- Session budgets: attended 6 steps / 5 minutes; autopilot (unattended) 15 steps
  / 30 minutes. [`config.yaml` `budgets`]
- Quotas: 5 autopilot sessions at once, 20 new sessions per hour, 25 MB per
  upload, 5 sandboxes at once. [`config.yaml` `quotas`, `sandbox`]
- Tools per turn capped at 128; extra connectors are benched with a notice.
  [`config.yaml` `llm.max_tools`, `CLAUDE.md`]
- Sandbox: a fresh cloud machine per session with a shell and your claimed
  folders at `~/store/<folder>/`; only that path persists; idle machines stop
  after 5 minutes. [`CLAUDE.md`, `config.yaml` `sandbox`]
- Browser: a headless browser the agent drives; you can watch it live in the
  session. [`config.yaml` `browser`, `lookingglass.jsx` `BrowserCanvas`]
- Web search uses SerpAPI. [`config.yaml` `tools.serpapi_key`]
- Memory: a curated core (up to 20,000 characters) plus notes (up to 2,000
  characters each), carried between sessions. [`config.yaml` `memory`]
- Deleted files can be restored with undo. [`api.py` `POST /files/undo`]
  (Retention window: set by PR 10.)
- Operational logs (`system_events`) older than 30 days are deleted each time the server starts, not on a timer. [`config.yaml` `system_log.retain_days`; `harness_module/system_log.py` `start()`]
- Model at default config: OpenAI `gpt-4.1-mini`. [`config.yaml` `llm`]
- License: AGPL-3.0. [`LICENSE`]
