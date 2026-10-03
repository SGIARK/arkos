# // Feature Spec: Legal Documents (NMORGAN)

*Drafting 01 · can start once ready 02's shell exists. Write every legal page now, with placeholders where a decision is open. Publishing waits on the decisions queue and counsel, and ready 10 enforces that.*

---

**Sources**
- `docs/launch/LAUNCH_READINESS_PLAN.md` §0.2 (placeholders), §4 (data inventory), §7.1–7.10 (outlines), §10 (verified facts)
- `docs/launch/DECISIONS.md` (D1, D2, D3, D6, D7, D8, D9, D10, D16, D17; D5 resolved: 18+)
- `config_module/config.yaml` (`llm`, `store`, `mcp`, `mcp_servers`, `tools.destructive`, `tools.serpapi_key`, `budgets`, `system_log`)
- `frontend/index.html` (third-party scripts and fonts), `LICENSE`, `NOTICE`
- Depends on: ready 02 (shell, `legal:` config)

**Status:** Drafting | **Author:** Vin Baker | **Last updated:** 2026-09-25

---

# Problem
arkos collects account data, conversations, files, agent memory and connected-account content, and sends them to six outside services. No document says so. No contract limits liability for an agent that can send email and merge PRs as the user. Without these pages, consent (ready 07) records agreement to nothing. Success: every page in plan §7.1–7.10 exists as a counsel-ready draft whose facts trace to plan §4/§10, and every undecided point is a placeholder or a comment tied to a `DECISIONS.md` ID.

---

# Technical Background
The facts these pages disclose live in config:
- `llm.base_url` defaults to `api.openai.com` with `gpt-4.1-mini`.
- `store.backend: supabase` puts bytes in the `arkos` bucket.
- `mcp.kind: composio` routes seven connectors through Composio's managed OAuth apps.
- `tools.serpapi_key` is an app-level key.
- `system_log.retain_days: 30`.
- `tools.destructive` is the always-ask list.

`frontend/index.html` loads React, ReactDOM, Babel and supabase-js from `unpkg.com`, and fonts from Google Fonts, so visitor IPs reach both until ready 12 lands. Once it does, the cookies and subprocessor pages drop the CDN rows; write them for the vendored state if ready 12 has merged first.

---

# Proposed Approach
Four content tasks, each a separate PR. Every page uses ready 02's shell, carries `arkos-doc-version`, and opens with a non-binding "short version" box.

**How decisions appear in the text:**
- A decision that fills a value → the §0.2 placeholder (`{{MIN_AGE}}`, `{{LLM_PROVIDER}}`, `{{OPERATOR_LEGAL_NAME}}`, …).
- A decision that picks between paragraphs → both paragraphs in the file, the default rendered, the other inside `<!-- D<n> alternative: … -->`.
- Every such spot carries `data-decision="D<n>"`, so `grep` finds every place a resolution has to touch.

A sentence of fact must trace to plan §4 or §10. If it doesn't, verify it in code, add it to §10 in the same PR, then write it.

**What "done" means here:** a complete draft. Final text happens after the decisions are resolved and counsel signs off (D17). Ready 10 refuses launch while a placeholder or a "Draft" banner remains.

**Not in scope:** help copy (ready 09), in-product strings (ready 08), the legal review itself.

---

# Implementation Plan

## Task 1: Privacy Policy and annexes
**Problem:** no disclosure of what's collected, who gets it, or for how long.
**Done when:** `privacy.html` has every §7.2 heading; `subprocessors.html` has one row per §4.2 provider; `cookies.html` lists `arkos_session`, Supabase's `localStorage` tokens, `arkos-theme`, `arkos-beta-ack`, and the CDN requests (unless ready 12 has merged); the Google Limited Use statement is present; the retention table says consent records are kept after account deletion, and says inactive accounts are kept until their owners delete them; D3, D7, D8 and D9 spots carry `data-decision`.
**Touch point:** `frontend/legal/privacy.html`, `frontend/legal/subprocessors.html`, `frontend/legal/cookies.html`.
**Priority:** High | **Effort:** 1 day | **Blockers:** ready 02 Task 1
**Out of scope:** resolving any decision.
**Acceptance test:** Tests 1 and 2; a reviewer can trace every §4.1 row to a sentence.

---

## Task 2: Terms of Service and Acceptable Use Policy
**Problem:** no agreement allocates responsibility for approved actions or disclaims AI output.
**Done when:** `terms.html` has §7.1 sections 1–22 in order, with warranty and liability sections in bold caps; eligibility says `{{MIN_AGE}}`+, a confirmed email, one account per person, open to anyone (D4 resolved); `acceptable-use.html` has every §7.3 prohibition, including the prohibited-data list and MIT academic integrity; D1, D2, D6, D8 and D10 spots carry `data-decision`.
**Touch point:** `frontend/legal/terms.html`, `frontend/legal/acceptable-use.html`.
**Priority:** High | **Effort:** 1 day | **Blockers:** ready 02 Task 1
**Out of scope:** choosing arbitration or courts (D10).
**Acceptance test:** Tests 1 and 2, plus an anchor for every §7.1 heading.

---

## Task 3: How arkos acts on your behalf
**Problem:** the approval model and its limits are explained nowhere, and they're the core of the liability allocation.
**Done when:** `ai-and-agents.html` covers §7.4; every `tools.destructive` entry appears as a readable name with `data-tool="<NAME>"`; budgets match `budgets.attended` (6 hops, 300 s) and `budgets.unattended` (15 hops, 1800 s).
**Touch point:** `frontend/legal/ai-and-agents.html`.
**Priority:** High | **Effort:** <4h | **Blockers:** ready 02 Task 1
**Out of scope:** generating the page from config at runtime. The test catches drift instead.
**Acceptance test:** Test 3.

---

## Task 4: Annex pages
**Problem:** no DMCA path, vulnerability policy, accessibility statement, or AGPL source offer.
**Done when:** `copyright.html` has the six §512(c)(3) elements, a repeat-infringer policy, and `{{LEGAL_EMAIL}}` and `{{OPERATOR_LEGAL_NAME}}` as the notice contact (registering a designated agent with the Copyright Office is a launch-checklist task, not a placeholder here); `security.html` and a new repo-root `SECURITY.md` share one reporting section with `{{SECURITY_EMAIL}}` (D16); `accessibility.html` lists known gaps; `licenses.html` has the AGPL notice, the `NOTICE` text, a row for each asset in `frontend/index.html`, and a `pip-licenses` table for `requirements.txt`; `changelog.html` has its first entry.
**Touch point:** `frontend/legal/{copyright,security,accessibility,licenses,changelog}.html`, new `SECURITY.md`.
**Priority:** Medium | **Effort:** 1 day | **Blockers:** ready 02 Task 1
**Out of scope:** registering the DMCA agent (drafting 02's checklist carries it).
**Acceptance test:** Test 4.

---

# Tests
New `tests/test_legal_pages.py`, using stdlib `html.parser`.

## Test 1: `test_legal_page_versions_match_config`
**What it verifies:** `arkos-doc-version` on `terms.html` and `privacy.html` equals `legal.terms_version` and `legal.privacy_version`.
**Why this matters:** the version a user accepts must be the text they read.

## Test 2: `test_placeholders_and_decisions_are_known`
**What it verifies:** every `{{…}}` token is in the §0.2 set, and every `data-decision` value is an ID heading in `DECISIONS.md`.
**Why this matters:** an invented token or an untracked decision is a question nobody will answer.

## Test 3: `test_destructive_list_matches_disclosure`
**What it verifies:** every name in `tools.destructive` appears as a `data-tool` on `ai-and-agents.html`.
**Why this matters:** adding an always-ask tool without disclosing it makes the page quietly wrong.

## Test 4: `test_licenses_page_names_every_frontend_asset`
**What it verifies:** each `<script src>` and font host in `frontend/index.html` has a row in `licenses.html`.
**Why this matters:** AGPL and OFL notices are obligations.

---

# Open Questions
1. *Verify in Task 1, not a decision:* the Supabase project region and backup window. Read them from the dashboard and fill §4.2/§4.3 before Task 1 merges.

---

# Implementation Notes
