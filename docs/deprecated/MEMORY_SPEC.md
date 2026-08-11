# Feature Spec: ARKOS Memory

**Sources**

- `arkos-inspo/hermes-agent` — `MemoryProvider` ABC + `MemoryManager`, frozen system-prompt snapshot, context compressor (protect head+tail, summarize middle), char-budgeted MEMORY.md/USER.md
- `arkos-inspo/claude-mem` — async non-blocking capture, content-hash dedup, session-end summaries, progressive-disclosure retrieval
- `arkos-inspo/claude-code` — memory-type taxonomy (user/feedback/project/reference), `/compact`, background "dream" consolidation as a forked subagent
- Current code: `memory_module/memory.py`, `agent_module/agent.py` (`add_context`/`get_context`/`step`), `config_module/config.yaml`, `base_module/task_runner.py`

**Status:** Not started | **Author:** | **Last updated:** 2026-05-31

---

# Problem

ARKOS has two of the four memory layers (working = `conversation_context` in Postgres; semantic = mem0), but the foundation is buggy and the conversation does not scale. Concretely:

- **Cross-session bleed (correctness bug).** `retrieve_short_memory` filters by `user_id` only, ignoring `session_id` (`memory.py:225`). Every session for a user is replayed into every other session. This silently defeats the whole subagent design — `task_runner.py` mints a fresh `session_id` per background task (`task_runner.py:125`) specifically to isolate its memory, and that isolation never takes effect on read.
- **Config is ignored.** `memory.short_term_turns: 50` is set in `config.yaml` but the loop hardcodes `turns=5` (`agent.py:198`, `agent.py:263`, `agent.py:345`). There is no token budget at all — context is a fixed message count regardless of size, so long turns silently blow the window and short ones waste it.
- **Failures vanish.** mem0 writes are fire-and-forget in a thread pool that `print()`s on error (`memory.py:136`), violating the structured-logging contract in CLAUDE.md. A dead mem0 backend looks identical to a healthy one.
- **No episodic layer.** Once a conversation exceeds the window, older turns are simply dropped. There is no summary, no compaction, no carry-over between sessions — the agent has amnesia at the window boundary.
- **No curated layer.** The agent cannot persist durable facts about the user or its own learned conventions across sessions. mem0 stores extracted snippets, but there is no human- or self-editable profile.
- **Hardcoded secret placeholder.** `os.environ["OPENAI_API_KEY"] = "sk"` (`memory.py:42`).

**Success looks like:** session-correct working memory under a real token budget; conversations that survive past the window via summaries; durable cross-session facts about the user that survive a restart; and every memory backend swappable behind one interface, with failures logged not swallowed.

---

# Technical Background

**The four-layer model** (common to all three sources):

1. **Working / short-term** — live conversation turns. ARKOS: `conversation_context` table.
2. **Episodic** — summaries of what happened in past turns/sessions. ARKOS: *missing*.
3. **Semantic** — durable facts about the user & world. ARKOS: mem0.
4. **Curated / procedural** — agent's own notes + user profile, hand- or self-edited. ARKOS: *missing*.

Compaction/consolidation is the cross-cutting process that moves information *down* the layers as context fills.

**Frozen-snapshot pattern (Hermes).** Curated memory is read once at session start and injected into the system prompt verbatim, then never mutated mid-session. Because the system-prompt prefix is identical on every turn, the LLM prefix cache stays warm for the whole session — a large latency/cost win, and it matters disproportionately for ARKOS's local vLLM backend.

**Head/tail-protected compaction (Hermes).** When context exceeds a threshold (~75%), keep the system prompt + first N turns (head) and the last M turns (tail) verbatim, and replace the middle with a single LLM-generated summary marked explicitly as reference-only. Head keeps task framing; tail keeps immediate coherence; the middle is where information is most compressible.

**ARKOS integration points already exist:**
- Write path: `agent.add_context()` (`agent.py:181`) is called on every inbound message and AI output.
- Read path: `agent.get_context()` (`agent.py:193`) assembles working + long-term before each state run.
- Background worker: `base_module/task_runner.py` + `agent_executor` already run detached subagents — the natural home for session-end summarization and consolidation, so no new infra is required.
- DB migrations: numbered, idempotent SQL in `db/migrations/` (see `0003_subagent_runtime.sql`). Note `conversation_context` is **not** in migrations — it is created by the mem0/supabase path; a new `session_summaries` table should be a proper migration.

---

# Proposed Approach

Refactor the existing `Memory` class behind a thin **provider interface** and grow it from two layers to four, in priority order of impact: **fix the foundation, add a token budget, add episodic compaction, add a curated layer.** mem0 (semantic) stays as-is functionally but gets moved behind the interface with proper async + logging.

What changes:

- Introduce `MemoryProvider` (ABC) and a `MemoryManager` facade. `Memory` becomes the default provider. Backends (Postgres working store, mem0 semantic store) sit behind the ABC so they are swappable and individually testable.
- `read_working` becomes session-scoped and token-budgeted (protect the most recent turns, drop oldest first), replacing the fixed `turns=N`.
- New `session_summaries` table + a compaction step in the agent loop: when the assembled context exceeds a configured token threshold, summarize the middle and persist the summary. At session start, inject the most recent prior-session summary.
- New curated layer: per-user profile + per-agent notes, frozen-snapshotted into the system prompt at session start, mutated only through an explicit `memory` tool.

What stays the same:

- `conversation_context` schema and the Pydantic (de)serialization in `memory.py` (`serialize`/`deserialize`) are kept.
- mem0 config and embedder/LLM endpoints are unchanged.
- The agent loop's `add_context`/`get_context` call sites stay; their *implementations* move behind `MemoryManager`.

Explicitly **not in scope** (deferred to keep this high-impact, not broad):

- Automatic background "dream" consolidation (Claude Code `autoDream`) — Open Question 1.
- Progressive-disclosure search tool (claude-mem 3-layer) and content-hash dedup for semantic memory — Open Question 2.
- Memory-poisoning / prompt-injection scanning of curated entries — Open Question 3.
- Markdown-file-on-disk storage for curated memory; this spec uses Postgres rows to stay inside the existing DB story.

---

# Implementation Plan

## Task 1: Harden the foundation + provider interface

**Problem:** Session-scoping bug corrupts the subagent feature; failures are swallowed; config is ignored; a secret placeholder is hardcoded. Backends are not swappable or testable in isolation.

**Done when:**
- `retrieve_short_memory` filters by both `user_id` **and** `session_id`.
- All `print()` in `memory.py` replaced with `emit_log()` from `logging_module`; mem0 write/read failures emit a structured error event.
- `os.environ["OPENAI_API_KEY"] = "sk"` removed; mem0 config values sourced from `config_module`.
- A `MemoryProvider` ABC exists (`memory_module/base.py`) with `write_turn`, `read_working`, `retrieve_semantic`; `Memory` implements it with no behavior change beyond the fixes above.
- Bare `except`/`traceback.print_exc()` replaced with specific exceptions per CLAUDE.md.

**Touch point:** `memory_module/memory.py`, new `memory_module/base.py`, `logging_module`, `config_module/config.yaml`.

**Priority:** P0 | **Effort:** ~1 day | **Blockers:** none

**Out of scope:** Token budgeting (Task 2); any new layer.

**Acceptance test:** `test_short_memory_is_session_scoped` (below).

---

## Task 2: Token-budgeted working memory + real semantic query

**Problem:** Context is a fixed message count (`turns=5`, hardcoded, config ignored), with no token awareness; the semantic query is just the last two messages concatenated.

**Done when:**
- `read_working` accepts a token budget (from `memory.short_term_*` config), returns the most recent turns that fit, oldest dropped first, never splitting a turn.
- The hardcoded `turns=5` call sites in `agent.py` (`:198`, `:263`, `:345`) read from config.
- The semantic-retrieval query is built from the latest user turn (the actual request), not `context[-2:]` concatenation.
- A token-count helper is used consistently (tiktoken or the model client's counter).

**Touch point:** `memory_module/working.py` (extracted from `memory.py`), `agent_module/agent.py`, `config_module/config.yaml`.

**Priority:** P0 | **Effort:** ~1–2 days | **Blockers:** Task 1

**Out of scope:** Summarizing dropped turns (that is Task 3 — until then, oldest turns are simply dropped, same as today but bounded by tokens).

**Acceptance test:** `test_working_memory_respects_token_budget` (below).

---

## Task 3: Episodic memory — session summaries + compaction

**Problem:** Past the window, conversation history is lost outright; nothing carries across sessions.

**Done when:**
- New migration `db/migrations/0005_session_summaries.sql` adds a `session_summaries` table keyed by `(user_id, session_id)` with `summary TEXT`, `turn_high_watermark`, `created_at` (idempotent, following the `0003` pattern).
- A compaction step runs in `agent.step()` before `get_context()`: when the token-counted working context exceeds `memory.compaction_threshold` (default 0.75 of budget), keep head (first N turns) + tail (last M turns) verbatim and summarize the middle via one LLM call. The summary row is upserted; compacted middle turns are excluded from future `read_working`.
- The summary is injected as a `SystemMessage` labeled reference-only (mirrors Hermes wording: "reference, not active instructions; discard if the latest user message contradicts").
- At session start, the most recent prior-session summary for the user is injected once.

**Touch point:** new `memory_module/episodic.py`, `agent_module/agent.py` (`step`/`get_context`), `db/migrations/0005_session_summaries.sql`, `model_module` (summarizer LLM call).

**Priority:** P1 | **Effort:** ~3–4 days | **Blockers:** Task 2

**Out of scope:** Background/async summarization — for v1 compaction runs inline in the loop. Moving it to `agent_executor` is Open Question 1.

**Acceptance test:** `test_compaction_protects_head_and_tail`, `test_prior_session_summary_injected_at_start` (below).

---

## Task 4: Curated memory — user profile + agent notes (frozen snapshot)

**Problem:** No durable, editable facts about the user or the agent's learned conventions survive across sessions; mem0 snippets are not a substitute for a coherent profile.

**Done when:**
- New migration adds a `curated_memory` table: `(scope, owner_id, kind, content, char_count, updated_at)` where `scope ∈ {user, agent}` and `kind` uses the Claude Code taxonomy (`user`/`feedback`/`project`/`reference`).
- At session start, curated entries for the user (and active agent) are read once and injected into the system prompt as a **frozen snapshot** with a usage header (`[chars/budget]`); the snapshot is not mutated mid-session, preserving the prefix cache.
- A `memory` tool (in `tool_module`) lets the agent `add`/`replace`/`remove` entries, char-budgeted (reject writes over the per-kind limit with a consolidation hint). Writes persist immediately and surface in the tool result, but do not alter the live system prompt until the next session.

**Touch point:** new `memory_module/curated.py`, `tool_module` (new `memory` tool), system-prompt assembly in `base_module/app.py` / `state_module/agent_buddy/state_ai.py`, new migration.

**Priority:** P1 | **Effort:** ~3–4 days | **Blockers:** Task 1

**Out of scope:** Injection scanning of entries (Open Question 3); on-disk markdown storage.

**Acceptance test:** `test_curated_snapshot_frozen_within_session`, `test_memory_tool_enforces_char_budget` (below).

---

# Tests

## Test 1: test_short_memory_is_session_scoped

**What it verifies:** Two sessions for the same `user_id` do not see each other's `conversation_context` rows; `read_working` returns only rows matching both `user_id` and `session_id`.

**Why this matters:** This is the correctness bug that defeats the entire subagent isolation design. Without this test the regression is invisible until a subagent leaks the chat agent's history into a background task.

---

## Test 2: test_working_memory_respects_token_budget

**What it verifies:** Given turns whose cumulative tokens exceed the budget, `read_working` returns the largest suffix of whole turns that fits, never splits a turn, and never returns more than the budget.

**Why this matters:** Fixed message counts silently overflow the context window on long turns and underuse it on short ones. The budget is the load-bearing guarantee that the agent loop never sends an over-window prompt.

---

## Test 3: test_compaction_protects_head_and_tail

**What it verifies:** When context exceeds the threshold, the first N and last M turns survive verbatim, the middle is replaced by exactly one summary message, and a `session_summaries` row is written at the correct watermark.

**Why this matters:** Head/tail protection is what keeps task framing and immediate coherence intact; a compaction that summarizes the tail breaks the current exchange. The watermark prevents re-summarizing the same turns twice.

---

## Test 4: test_prior_session_summary_injected_at_start

**What it verifies:** Starting a new session for a user with an existing summary injects exactly one reference-only summary message at position 0; a user with no prior summary gets none (no empty SystemMessage).

**Why this matters:** Cross-session carry-over is the user-visible payoff of episodic memory; the empty-message guard avoids polluting context (a bug class the current `retrieve_long_memory` already guards against).

---

## Test 5: test_curated_snapshot_frozen_within_session

**What it verifies:** After the session-start snapshot is built, a `memory(add)` tool call persists to the store but the system-prompt snapshot string is byte-identical to its pre-write value within the same session.

**Why this matters:** Mid-session mutation would invalidate the prefix cache on every write — the exact cost regression the frozen-snapshot pattern exists to prevent. The test pins the invariant.

---

## Test 6: test_memory_tool_enforces_char_budget

**What it verifies:** A `memory(add)` that would push a kind over its char limit is rejected with a consolidation hint and leaves the store unchanged; one within budget succeeds and updates `char_count`.

**Why this matters:** Unbounded curated memory grows the system prompt without limit, eroding the very context budget Tasks 2–3 protect.

---

# Open Questions

1. Should session-end summarization (Task 3) and consolidation run **inline** or move to a background `agent_executor` task via `task_runner.py`? Inline is simpler and ships first; background matches Claude Code's "dream" model and avoids latency on the user's turn. *Leaning: inline for v1, background as a fast-follow once the summary quality is validated.*
2. Does semantic memory (mem0) need **content-hash dedup** and **progressive-disclosure retrieval** (claude-mem), or is mem0's own dedup/ranking sufficient at current scale? Validate against real session volume before building.
3. Curated memory is agent-writable — do we need **prompt-injection scanning** of entries before they enter the system prompt (Hermes scans at snapshot-build time)? Risk rises if any external/tool content can reach `memory(add)`.
4. Where should the system prompt be assembled so the frozen snapshot lives in exactly one place — `base_module/app.py`, or inside `state_ai`? Current prompt text lives in `config.yaml:app.system_prompt`; the snapshot needs a single, cache-stable injection point.

---

# Implementation Notes

*Add entries here as work lands.*

- (pre-work) `conversation_context` is created by the mem0/supabase path, not by `db/migrations/`. New tables (`session_summaries`, `curated_memory`) should be proper numbered migrations so they exist independently of mem0 being enabled.
- (pre-work) `memory.short_term_turns: 50` in `config.yaml` is currently dead config — the loop hardcodes `5`. Task 2 makes the config live; double-check nothing else relied on the hardcoded value.
