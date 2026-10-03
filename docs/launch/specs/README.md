# Launch specs — index and execution order

Feature specs for the MIT launch, written with `/nmorgan` against `main` @ `a522ff4`.
The parent plan is `../LAUNCH_READINESS_PLAN.md`; every open choice that blocks
launch lives in `../DECISIONS.md` (11 open).

## The three folders

| Folder | Meaning | Who can start |
|---|---|---|
| `ready/` | No decision needed. Policy values ship as config defaults; unknowns are verification steps inside a task. | Any agent, now |
| `drafting/` | Can be written now with `{{placeholders}}` and `data-decision` markers; final only after the named decisions and counsel. | Any agent, now |
| `deferred/` | Not needed to launch. Written so it can be picked up afterwards. | After launch |

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
| 1 | [01 File Data Cleanup](ready/01-file-data-cleanup.md) | — |
| 1 | [02 Legal Surface & Public Config](ready/02-legal-surface-and-public-config.md) | — |
| 1 | [03 Data Export](ready/03-data-export.md) | — |
| 1 | [04 Security Headers & security.txt](ready/04-security-headers.md) | — |
| 1 | [11 Project Delete](ready/11-project-delete.md) | — |
| 2 | [05 Account Deletion](ready/05-account-deletion.md) | 01 |
| 2 | [06 Session Delete & Retention](ready/06-session-delete-and-retention.md) | 01, 02 |
| 2 | [07 Consent Capture](ready/07-consent-capture.md) | 02 |
| 2 | [08 In-Product Disclosures](ready/08-in-product-disclosures.md) | 02 |
| 2 | [12 Vendor Frontend Assets](ready/12-vendor-frontend-assets.md) | 02, 04 |
| 3 | [09 Help Center](ready/09-help-center.md) | 02; describes 03, 05, 06, 11 |
| 3 | [10 Launch Gate Script](ready/10-launch-gate-script.md) | 02 |

`harness_module/account.py` is shared by 03 and 05. Whichever merges first creates it.

## Drafting

| Spec | Start after | Final after |
|---|---|---|
| [01 Legal Documents](drafting/01-legal-documents.md) | ready 02 Task 1 | D1, D2, D3, D6, D7, D8, D9, D10, D16, D17 (counsel) |
| [02 Operator Documents](drafting/02-operator-documents.md) | now | D2, D7, D9, D19 |

## Deferred

| Spec | Why it waits |
|---|---|
| [01 Inactive Account Sweep](deferred/01-inactive-account-sweep.md) | Not a launch requirement, and it needs an email sender (D16). Accounts are kept until their owners delete them. |

## What's no longer here

- **Eligibility Gate:** withdrawn. D4 resolved on 2026-09-25: sign-up is open.
- **`parked/`:** removed on 2026-10-02. The six decisions that kept specs parked
  or open (D11–D15, D18) don't block launch, so they ship with their defaults
  (see the end of `DECISIONS.md`): project delete and vendored assets moved to
  `ready/` as 11 and 12, and the inactive sweep moved to `deferred/`.
- **The old flat `01`–`11` specs:** replaced by the folders above. Blob deletion
  moved out of Account Deletion into ready 01; operator documents moved out of the
  Launch Gate into drafting 02; asset vendoring moved out of Security Headers into
  ready 12; project delete moved out of Session Delete into ready 11.
