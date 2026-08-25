# Contributing to ARKOS

## Philosophy

Write code for the next person reading it, not for the machine running it.
ARKOS is a student team using AI-assisted development. That means readable,
focused, and well-named code matters more than clever or compact code.
The reviewer — human or AI — should understand a function in 10 seconds.

Small PRs merge faster, break less, and are easier to review. 
One concern per PR. One job per function. One behavior per test.

## Quick reference

All coding rules, PR conventions, and test guidelines live in [`CLAUDE.md`](../CLAUDE.md)
at the repo root. That file is the source of truth for both human contributors
and AI coding tools. Read it before writing code.

## CI checks

Every PR must pass all four before merging:

```bash
ruff check .                                          # linting
ruff format --check .                                 # formatting
mypy .                                                # type checking
pytest tests/ -v -m "not integration"                 # unit tests
```

Run these locally before opening a PR. The CI pipeline runs them in parallel
and blocks the Docker build if any fail.

## Branch naming

`type/short-description` — e.g. `feat/gcal-per-user-auth`, `fix/state-output-ask-user`

Valid types: `feat`, `fix`, `refactor`, `test`, `chore`, `docs`

## Getting help

If you are unsure how to implement something in a way that fits the existing
pattern , MCP integrations, or the agent loop —
read the relevant module first, then ask before writing a large amount of code
in the wrong direction.

