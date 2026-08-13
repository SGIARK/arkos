# ARKOS

Mid-redesign. The old architecture guidelines are gone: they mandated state
graphs, `StateOutput`, routers, and mem0, all of which this redesign deletes.
They are kept at `docs/deprecated/CLAUDE_pre_redesign.md` for history only.

**Start at `docs/single_loop_redesign_spec.md`.** It routes to everything else.

`docs/contracts.md` is law. The spec says what to build; contracts says whether
it is correct. If they disagree, contracts wins and the spec is the bug.

**Never read `docs/deprecated/`.** It is the architecture being deleted. Do not
open it, cite it, or follow its task numbers. Stale pointers to it in docstrings
should be deleted, not followed.

**`state_module` is deprecated.** It still exists because Task 7 has not deleted
it yet. **Add no new states or routers**, and do not extend the ones there.

Coding standards will be rewritten in Task 7, once the new shape exists. Until
then: ruff, type hints on every signature, `async def` for anything that awaits,
no `print()` in production paths.
