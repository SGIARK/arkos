# Contributing to arkos-core

## Philosophy

Write code for the next person reading it, not for the machine running it.
arkos-core is built by a small team using AI-assisted development. That means readable,
focused, and well-named code matters more than clever or compact code.
The reviewer, human or AI, should understand a function in 10 seconds.

Small PRs merge faster, break less, and are easier to review.
One concern per PR. One job per function. One behavior per test.

## Quick reference

All coding rules, PR conventions, and test guidelines live in
[`CLAUDE.md`](CLAUDE.md), beside this file. That file is the source of truth for
both human contributors and AI coding tools. Read it before writing code.

## The CLA

Every pull request requires CLA agreement; the cla-assistant check enforces it.
The agreement is in [`CLA.md`](CLA.md), and the check comments on your first PR
with what to reply. A PR cannot merge until it passes.

## CI checks

Every PR must pass all four before merging:

```bash
ruff check .                     # linting
ruff format --check .            # formatting
mypy --follow-imports=silent --ignore-missing-imports --disable-error-code=arg-type \
    tool_module/tools/ tool_module/browser/tool.py
DB_URL=postgresql://test:test@localhost:5432/test pytest tests/ -q --timeout=120 -m "not integration"
```

Run these locally before opening a PR. The tests need a Postgres with
`python db/migrate.py` applied.

## Branch naming

`type/short-description`, for example `feat/gcal-per-user-auth` or `fix/manifest-cap-off-by-one`

Valid types: `feat`, `fix`, `refactor`, `test`, `chore`, `docs`

## Getting help

If you are unsure how to implement something in a way that fits the existing
pattern, whether in the store, the MCP integrations or the agent loop, read the
relevant module first, then ask before writing a large amount of code in the
wrong direction.

