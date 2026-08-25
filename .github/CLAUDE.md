# ARKOS — Coding Standards

Loaded automatically by AI coding tools. Follow these rules on every task.

---

## The one rule that covers most cases
 s
Write for the next person reading it, not the machine running it.
If a reviewer needs more than 10 seconds to understand a function, rewrite it.

---

## Code style

**Keep things small**
- Functions: under 40 lines. If longer, split into named helpers.
- PRs: under 400 lines changed. One concern per PR.
- Commits: one logical change. Imperative present tense, max 72 chars.
  `add StateOutput model` — not `Added StateOutput model`

**Naming over comments**
- Names should read like plain English. `completion_signal` not `sig`.
- Comments explain WHY, not WHAT. The code explains what.
- Only comment where the reasoning is genuinely non-obvious.

**Types**
- Every function signature has type annotations — parameters and return type.
- No bare `Any`. If you must use it, leave a comment explaining why.

**Do not**
- Catch bare `except:` — always catch a specific exception type.
- Use magic numbers — name your constants.
- Import `*` from any module.
- Leave `TODO` in a PR — fix it or open a separate issue.

---

## CI — must pass before merge

```bash
ruff check .                               # linting
ruff format --check .                      # formatting
mypy .                                     # type checking
pytest tests/ -v -m "not integration"      # unit tests
```

Run these locally before opening a PR. CI runs them in parallel and blocks the Docker build if any fail.

---

## Pull requests

PR should adhere to pull_request_template.md

**Branch naming:** `type/short-description`

| Type | When |
|------|------|
| `feat/` | new capability |
| `fix/` | bug fix |
| `refactor/` | restructure, no behavior change |
| `test/` | tests only |
| `chore/` | deps, config, tooling |
| `docs/` | documentation only |

One PR, one concern. If you touched something unrelated, revert it and open a separate PR.

If this PR changes a public API, config field, or environment variable, open a companion PR in the docs repo and link it in the description.

---

## Tests

Test behavior, not implementation. Write tests that would catch a real bug.

- Mirror the module: `agent_module/agent.py` → `tests/test_agent.py`
- Name tests: `test_[function]_[scenario]`
- Mark slow or external tests: `@pytest.mark.integration`
- Mock all external dependencies: LLM, database, APIs.
- Don't test that a mock was called — test what the function returns.

