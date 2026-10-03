# arkos launch — Decisions Queue

Every open choice that **blocks launch** and belongs to a person, not to an
implementing agent, is listed here and nowhere else. Specs never wait on these.
Each one ships with the **default** below (a config value or a placeholder), or
sits in `specs/drafting/` until it's answered. A choice that doesn't block launch
isn't queued: it ships with its default (see the last section).

**How to resolve one:** change `Status` to `resolved`, write the answer and the
date in `Resolution`, then do what `On resolve` says. `scripts/check_launch_ready.py`
(ready spec 10) fails while any entry here is still `open`.

**Legend.** *Who:* **owner** = you can answer it now; **counsel** = needs a lawyer;
**world** = needs a real-world action (register, sign, create).

**Open: 11** — owner: D3, D6, D7, D8, D9, D16 · counsel / world: D1, D2, D10, D17, D19

---

## Owner calls

### D6 — Pricing
- **Status:** open (owner, 2026-10-02: "not specified right now")
- **Who:** owner · **Blocks launch:** yes
- **Options:** free beta · paid from launch
- **Default / recommendation:** free beta; the Terms reserve the right to charge later with notice. The Terms fees section is drafted to this default but stays open until you confirm.
- **Where it lands:** Terms §10 (drafting 01)
- **On resolve:** write the Terms fees section.
- **Resolution:** —

### D7 — Retention windows
- **Status:** open (owner, 2026-10-02: "not yet specified")
- **Who:** owner · **Blocks launch:** yes
- **Options:** the values below, or others
- **Default / recommendation:** undo for deleted files 30 days · conversations, files and memory kept until the user deletes them · system log 30 days (already in code) · all data deleted within 90 days of shutting the service down
- **Where it lands:** `retention.deleted_files_days` (ready 01, 02); the retention table in the Privacy Policy (drafting 01); `RETENTION.md` (drafting 02)
- **On resolve:** set config; fill the Privacy Policy table. Until then the code ships with the 30-day default, and nothing here blocks building.
- **Resolution:** —

### D16 — Contact addresses and email sending
- **Status:** open
- **Who:** owner + world · **Blocks launch:** yes
- **Options:** which domain; which provider sends mail
- **Default / recommendation:** support@, privacy@, security@ and legal@ on a domain you control, with at least two people on each; Supabase Auth on custom SMTP for sign-up mail.
- **Where it lands:** `{{SUPPORT_EMAIL}}`, `{{PRIVACY_EMAIL}}`, `{{SECURITY_EMAIL}}`, `{{LEGAL_EMAIL}}` (drafting 01, ready 08); `security.contact` (ready 04)
- **On resolve:** create the inboxes; replace the placeholders; set `security.contact`.
- **Resolution:** —

### D3 — AI model provider at launch
- **Status:** open
- **Who:** owner · **Blocks launch:** yes
- **Options:** OpenAI API (current config) · self-hosted "MIT profile" on SGLang
- **Default / recommendation:** OpenAI, with zero data retention requested. It's the bigger privacy disclosure, but it works today.
- **Where it lands:** `{{LLM_PROVIDER}}`, the AI-processing section of the Privacy Policy, and the subprocessor list (drafting 01)
- **On resolve:** fill the placeholder; delete the unused branch comment in `privacy.html`.
- **Resolution:** —

### D8 — Research or training use of user data
- **Status:** open
- **Who:** owner (+ COUHES if yes) · **Blocks launch:** yes
- **Options:** never · opt-in later with a new policy version · yes at launch
- **Default / recommendation:** never at launch. Any later use is opt-in, with COUHES (MIT's review board) review first.
- **Where it lands:** Privacy Policy "how we use it"; Terms "your content" (drafting 01)
- **On resolve:** write the chosen sentence into both.
- **Resolution:** —

### D9 — Who on the team may read user data
- **Status:** open
- **Who:** owner · **Blocks launch:** yes
- **Options:** names, and when access is allowed
- **Default / recommendation:** named operators only, only for security, abuse, or a problem the user reported; every access logged.
- **Where it lands:** Privacy Policy "human access" (drafting 01); `DATA_ACCESS_POLICY.md` (drafting 02)
- **On resolve:** list the names; fill both.
- **Resolution:** —

---

## Counsel and real-world items

### D1 — Who operates arkos
- **Status:** open
- **Who:** owner + counsel + world · **Blocks launch:** yes
- **Options:** individual(s) · LLC · MIT unit (see D2)
- **Default / recommendation:** form an LLC before launch, so the Terms bind an entity and not a person.
- **Where it lands:** `{{OPERATOR_LEGAL_NAME}}`, `{{OPERATOR_POSTAL_ADDRESS}}` in every legal page (drafting 01)
- **On resolve:** fill both placeholders; sign vendor terms (OpenAI, e2b, Composio, SerpAPI, Supabase) as that entity.
- **Resolution:** —

### D2 — Relationship to MIT
- **Status:** open
- **Who:** owner + counsel · **Blocks launch:** yes
- **Options:** independent project · MIT-lab research project · official MIT service
- **Default / recommendation:** independent. Say plainly "not operated, sponsored or endorsed by MIT", and use no MIT marks.
- **Where it lands:** `{{MIT_RELATIONSHIP_STATEMENT}}` (ready 01 footer, drafting 01); whether the WISP applies (drafting 02)
- **On resolve:** fill the statement; if MIT-operated, hand the legal documents to MIT OGC.
- **Resolution:** —

### D10 — Disputes
- **Status:** open
- **Who:** counsel · **Blocks launch:** yes
- **Options:** courts in Massachusetts · arbitration with class-action waiver
- **Default / recommendation:** courts, with a 30-day informal-resolution step first.
- **Where it lands:** Terms §18 and `{{GOVERNING_LAW}}` (drafting 01)
- **On resolve:** delete the unused branch comment in `terms.html`.
- **Resolution:** —

### D17 — Legal review
- **Status:** open
- **Who:** counsel · **Blocks launch:** yes
- **Options:** —
- **Default / recommendation:** MIT OGC if D2 = MIT-operated, otherwise outside counsel.
- **Where it lands:** everything in drafting 01; the consent copy in ready 05
- **On resolve:** record the reviewer and date in `changelog.html`; remove the "Draft" banners.
- **Resolution:** —

### D19 — Spend caps
- **Status:** open
- **Who:** owner + world · **Blocks launch:** yes
- **Options:** cap amounts
- **Default / recommendation:** a hard monthly cap and a 50% alert on OpenAI, e2b and SerpAPI. With sign-up open (D4), per-user quotas don't cap the total.
- **Where it lands:** provider consoles; plan §8.9
- **On resolve:** record the caps in `LAUNCH_CHECKLIST.md`.
- **Resolution:** —

---

## Resolved

### D5 — Minimum age
- **Status:** resolved
- **Resolution:** 2026-10-02, owner: **18+.** `legal.min_age: 18` is already the shipped config value, so no spec changes. Terms and Privacy Policy fill `{{MIN_AGE}}` with 18.

### D4 — Restrict sign-up to `@mit.edu`
- **Status:** resolved
- **Resolution:** 2026-09-25, owner: **no.** Open to any confirmed email or Google account. The eligibility-gate spec was withdrawn.

---

## Not queued: shipping with defaults

These were in the queue until 2026-10-02. None blocks launch, so each now ships with
the recommendation. Change one by editing the named spec or config; nothing else moves.

| Was | Default now in effect | Where |
|---|---|---|
| D11 self-host fonts and scripts | Yes, vendor them | ready 07 |
| D12 inactive accounts | Never deleted at launch; the sweep is post-launch work | ready 02, Task 6 |
| D13 project delete | Sessions are detached, not deleted | ready 02 |
| D14 consent records on account deletion | Kept (no foreign key) | ready 05 |
| D15 export contents | Operational logs excluded; 2 GB ceiling | ready 03 |
| D18 DMCA agent | Registration is a launch-checklist task, not a decision; `copyright.html` names `{{LEGAL_EMAIL}}` as the contact | drafting 01, drafting 02 |
