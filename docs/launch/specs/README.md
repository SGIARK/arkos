# Launch specs — index and execution order

Feature specs for the MIT launch, written with `/nmorgan` against `main` @ `a522ff4`
and checked against the code in a verification pass on 2026-10-02. The parent plan
is `../LAUNCH_READINESS_PLAN.md`; every open choice that blocks launch lives in
`../DECISIONS.md` (11 open).

## The two folders

| Folder | Meaning | Who can start |
|---|---|---|
| `ready/` | No decision needed. Policy values ship as config defaults; unknowns are verification steps inside a task. | Any agent, now |
| `drafting/` | Can be written now with `{{placeholders}}` and `data-decision` markers; final only after the named decisions and counsel. | Any agent, now |

## Rules for an implementing agent

1. **Never make a decision.** If a spec seems to need one, it's a gap: add it to
   `DECISIONS.md` with a default, and ship the default through config or a placeholder.
2. **Verifications are yours.** An Open Question marked *Verify in Task N* is
   something you check (docs, a dev project, the code) and record in that spec's
   Implementation Notes before continuing.
3. **Migrations take the next free number at merge time.** Specs write `NNNN_…`.
4. **Frontend changes bump `?v=` on every tag in `frontend/index.html`.**
5. **Run all four CI commands** (`.github/CLAUDE.md`) before calling a PR green, and
   see UI changes in a browser.

## Ready — execution order

Specs in the same wave don't depend on each other and can run in parallel.

| Wave | Spec | Depends on |
|---|---|---|
| 1 | [01 Legal Surface & Public Config](ready/01-legal-surface-and-public-config.md) | — |
| 1 | [02 Deletion](ready/02-deletion.md), Tasks 1–2 and conversation delete in Task 3 | — |
| 1 | [03 Data Export](ready/03-data-export.md) | — |
| 1 | [04 Security Headers & security.txt](ready/04-security-headers.md) | — |
| 2 | [02 Deletion](ready/02-deletion.md), rest of Task 3, Tasks 4–5 | its Tasks 1–2; Task 5 also needs 01 |
| 2 | [05 Consent Capture](ready/05-consent-capture.md) | 01 |
| 2 | [06 In-Product Disclosures](ready/06-in-product-disclosures.md) | 01 |
| 2 | [07 Host Our Own Scripts and Fonts](ready/07-vendor-frontend-assets.md) | 01, 04 |
| 3 | [08 Help Center](ready/08-help-center.md) | 01; describes 02 and 03 |
| 3 | [09 Launch Gate Script](ready/09-launch-gate-script.md) | 01 |
| after launch | [02 Deletion](ready/02-deletion.md), Task 6 | its Task 4; D16 |

`harness_module/account.py` is shared by 02 (deletion) and 03 (export). Whichever
merges first creates it.

## Drafting

| Spec | Start after | Final after |
|---|---|---|
| [01 Legal Documents](drafting/01-legal-documents.md) | ready 01 Task 1 | D1, D2, D3, D6, D7, D8, D9, D10, D16, D17 (counsel) |
| [02 Operator Documents](drafting/02-operator-documents.md) | now | D2, D7, D9, D19 |

## Sharing the deletion design

[ready/02-deletion.md](ready/02-deletion.md) is self-contained and written for
non-engineers. It opens with a table of what can be deleted, what happens, and
whether there's undo, then five rules every delete follows, then the tasks. Share
that file on its own when someone needs to review how deletion works.

## History

- **2026-10-03, deletion condensed:** ready 02's 11 tasks merged into 6, with a task summary table; the Notion copy uses fold-out task cards.
- **2026-10-03, plain-language rewrite:** every spec rewritten to the nmorgan rules P-004 to P-006: plain words with a short glossary where needed, file lists instead of function-by-function touch points, no code stubs except short SQL or config, and tests as one table. Same tasks, tests and decisions.
- **2026-10-02, consolidation:** File Data Cleanup, Account Deletion, Session
  Delete & Retention, Project Delete and the Inactive Account Sweep merged into
  ready 02 Deletion. The other ready specs were renumbered: Legal Surface 02→01,
  Consent 07→05, Disclosures 08→06, Vendor Assets 12→07, Help Center 09→08, Launch
  Gate 10→09. `deferred/` went away; the sweep became a Deletion task.
- **2026-10-02, verification pass:** four claims fixed against the code. Memory is
  blob-backed; `Composio.disconnect` swallows a failed revoke; the CSP must allow
  inline script; the projects view has no empty state.
- **2026-10-02:** decisions that don't block launch left the queue and ship with
  defaults (end of `DECISIONS.md`); `parked/` went away.
- **2026-09-25:** Eligibility Gate withdrawn (D4: sign-up is open). The original flat
  `01`–`11` specs were split into folders.
