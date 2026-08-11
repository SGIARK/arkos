# PROPOSED CHANGES — gap log

**RESTORED 2026-08-06.** This file was deleted prematurely; it is back.
Folding a gap into the law docs does NOT mean deleting its entry here — the
entry is the record of what was decided and why. Status is marked per gap.

- **G1–G11, G19–G21** — marked folded on 2026-08-05 by the prior session.
  **Their text is NOT in this file and I did not verify the fold.** See
  "MISSING ENTRIES" at the bottom.
- **G12, G13** — folded 2026-08-05 (canvas panel + quotas → D23, contracts).
- **G14, G16, G17, G18, E1, E2** — folded 2026-08-06, text preserved below.
- **G15** — struck.

---

## G14 — UNDERSPEC: `user_connections` table schema undefined (today: in-memory only)

**Status: FOLDED 2026-08-06, then SUPERSEDED by D24 the same day.**

**Finding.** `contracts.md` L61 lists `user_connections` as a new table; Task 3
requires it "persisted + rehydrated" with acceptance "restart = 1 DB read, 0
Smithery PUTs for connected servers" (`single_loop_redesign_spec.md` L179, L182).
But the schema is undefined — and the obvious thing to persist, the connection id,
is a **red herring**: `smithery.py`'s `_user_conn_id()` derives it deterministically
as `f"user-{safe}__{server_name}"`, a **pure function of `(user_id, server_name)`**,
not a Smithery-issued secret. Storing it buys nothing a `str.format` doesn't already
give you for free at rehydrate time.

**Proposed resolution.** Persist the **fact of connection + the tool cache**, not
the derivable id. What actually drives "restart = 0 PUTs" is knowing a server is
already connected (so the harness skips the PUT) and having its last tool list
(so the first turn needs no live introspection):

```sql
CREATE TABLE user_connections (
    user_id       UUID        NOT NULL,
    server_name   TEXT        NOT NULL,   -- 'linear', 'gmail', ...
    status        TEXT        NOT NULL DEFAULT 'connected',  -- the load-bearing fact
    tools_cache   JSONB,                  -- last known tool list (+ TTL) — skips introspection
    tools_cached_at TIMESTAMPTZ,          -- drives TTL revalidation (Task 3)
    connection_id TEXT,                   -- OPTIONAL/derived: = _user_conn_id(user_id, server_name); denormalized only for debugging
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, server_name)
);
```

`user_id UUID` matches the unified keyspace (G18). On startup the harness reads
this table once: for each `status='connected'` row it **rehydrates without a PUT**
(recomputing `connection_id` from `_user_conn_id` if needed) and serves tools from
`tools_cache` until TTL lapses, at which point Task 3 revalidates and updates
`tools_cache`/`tools_cached_at`. `connection_id` is nullable/derived, never
load-bearing. Credentials are **never** stored here — they stay behind the Smithery
vault+proxy (`contracts.md` L232, `auth.md` L73).

**Lands in.** `contracts.md` § db fate map (L61: expand the `user_connections`
entry with these columns); migration 0c (`single_loop_redesign_spec.md` L157).

**Confidence:** high.

> **OVERTURNED (John, 2026-08-06) → D24.** The "connection_id is derivable, don't
> store it" position is **wrong**, and the reason is the rest of this entry:
> `server_name` is OUR config-file key, not a Smithery identifier. Smithery knows
> the server by `mcp_url`. So the derivation `user-{user_id}__{server_name}` is a
> formula over a mutable input, which makes renaming a key under `mcp_servers:`
> silently disconnect every user and strand the old connection in Smithery's
> namespace. **Anything derived from mutable inputs is a drift surface.**
> Resolution: identity is `mcp_url`; the connection id is minted ONCE at connect
> and STORED; `_user_conn_id()`/`_shared_conn_id()` are deleted; no `server_name`
> column at all. Config keys survive as in-process labels referenced by nothing
> durable. See `decisions.md` D24, `schema.md`, Task 3.

---

# Tier 3 — Config / hygiene / keyspace

## G15 — STRUCK (superseded)

The two role vocabularies it flagged (`interactive/worker`,
`foreground/background`) no longer exist. A session is *attended* or
*unattended* — a phase, not a type — and that single word parameterizes the
manifest (there is only one), the retry policy, and termination. Nothing to map.

---

## G16 — CODE-MISMATCH: `config.yaml` says qwen3 / Qwen3-8B, spec says qwen25 / Qwen2.5-7B, `model_name` is "tgi"

**Status: FOLDED 2026-08-06 → D25 + Task 0a + contracts violations table.**

**Finding.** Three sources disagree:
- `run.sh` (what actually runs): `Qwen/Qwen2.5-7B-Instruct` on :30000, **no
  `--tool-call-parser`** flag.
- `config.yaml`: `llm.model_name: "tgi"` (a stale TGI-server leftover);
  `computer_agent.llm.model_name: "Qwen/Qwen3-8B"` with a comment
  "restart SGLang with `--tool-call-parser qwen3`."
- `single_loop_redesign_spec.md` L31–42: `Qwen/Qwen2.5-7B-Instruct` +
  `--tool-call-parser qwen25`.

Which is authoritative for Task 0a (the load-bearing tool-calling measurement)?

**Proposed resolution.** **`run.sh` — the actually-running server — is ground
truth**, and the spec's Task 0a explicitly commands "verify what the running
server was launched with … fix `run.sh` if it drifted" (L38–42, L136). Standardize
on **Qwen2.5-7B-Instruct + `--tool-call-parser qwen25`** (the parser must match
the model family; the 7B is the in-scope model — swap is "config later,"
`single_loop_redesign_spec.md` L109). Concretely:

1. Add `--tool-call-parser qwen25` to `run.sh` (the redesign has nothing to stand
   on without it — L38).
2. Set `llm.model_name` to the served model name SGLang expects (not `"tgi"`).
3. **Delete the entire `computer_agent:` config block** — `computer_module`
   dissolves and `ComputerAgent` is deleted (`contracts.md` L250, `single_loop`
   Task 8); its `Qwen3-8B`/`qwen3` values are dead config and the source of this
   contradiction.

(Whether workers later run a hosted frontier model is a separate cost call —
`single_loop_redesign_spec.md` Open Q4 — explicitly out of v1 scope.)

**Lands in.** `model_module/run.sh` (add flag), `config_module/config.yaml`
(fix `model_name`, delete `computer_agent`), and note in
`single_loop_redesign_spec.md` § Implementation Notes when Task 0a lands.

**Confidence:** high.

> **RESOLVED (John, 2026-08-06), with the recommendation reversed.** The proposal
> above picked the wrong winner: **Qwen3-8B is the model**, Qwen2.5-7B is wrong,
> `model_name` is wrong, SGLang is right. Two corrections found while folding:
> - `--tool-call-parser qwen3` **is not a real parser.** Qwen3 dense models use
>   the Hermes-style `<tool_call>` format read by **`qwen25`**; `qwen3_coder` is
>   for the Coder variants; `qwen3` is the name of the *reasoning* parser. The
>   config comment would have failed at launch.
> - **Qwen3-8B is a hybrid reasoning model.** It emits `<think>` by default.
>   Without `--reasoning-parser qwen3` that reasoning returns inside `content`,
>   streams into our `content` events, and renders as the model's reply — while
>   spending output tokens before the first useful one, which is the latency
>   complaint this redesign exists to fix.
>
> Authoritative: **Qwen3-8B + `--tool-call-parser qwen25` + `--reasoning-parser
> qwen3`**, declared once in `run.sh`. `computer_agent:` deleted. Task 0a now
> measures reasoning/tool-calling interaction (known Qwen3 failure: tool-call XML
> leaking into content when both are on) alongside malformed-call rate.
> See `decisions.md` D25.

---

## G17 — CONTRADICTION: `docs/specs/*` contradict the law docs; code docstrings point implementers there

**Status: FOLDED 2026-08-06 → files moved, CLAUDE.md guardrail, Task 0b.**

**Finding.** `docs/specs/` (**13** files, dated 2026-07-25) predates the redesign
law docs (`contracts.md`, dated 2026-08-04) and describes the architecture being
deleted (state graphs, mem0 memory, ComputerAgent). Several shipped docstrings
still route implementers there:
`agent_module/agent.py:305` ("see HARNESS_SPEC Task 4"),
`agent_module/agent.py:306` ("…and ENVIRONMENT_SPEC" — a spec that **does not
exist**; there are 13 spec files and none is named `ENVIRONMENT_SPEC`),
`agent_module/agent.py:328` ("MEMORY_SPEC Task 2"),
`computer_module/__init__.py:4` ("See … COMPUTER_SPEC.md and COMPUTER_AGENT_SPEC.md"),
`computer_module/agent.py:5` ("see COMPUTER_AGENT_SPEC.md"), and
`spike_sandbox.py:100` ("proceed to COMPUTER_SPEC Task 4"). An AI-assisted session
reading those follows dead guidance — and `ENVIRONMENT_SPEC` is a **dangling
reference to a doc that was never written.**

**Proposed resolution.** `contracts.md` is law; `docs/specs/*` is **archival,
superseded**. Two cheap actions:

1. Add a one-line banner to the top of each `docs/specs/*.md` (or move them under
   `docs/specs/archive/`): "SUPERSEDED by `docs/contracts.md` (2026-08-04). Kept
   for history; do not implement from this."
2. These offending docstrings all live in files that Task 7/8 deletes or rewrites
   (`agent_module/agent.py` → `loop.py`; `computer_module/*` dissolved,
   `contracts.md` L250; `spike_sandbox.py` is a throwaway proof "not imported by
   the app"). Retarget or drop them **during** that deletion — no separate work,
   and the `ENVIRONMENT_SPEC` dangling reference dies with the file.

This is the same hazard the spec's Task 0b flags for `CLAUDE.md`
(`single_loop_redesign_spec.md` L145–151); fold the `docs/specs/` banner into
Task 0b so the warning is consistent across all superseded guidance.

**Lands in.** Task 0b (`single_loop_redesign_spec.md` L145–151): extend it to
also banner `docs/specs/`. No law-doc change.

**Confidence:** high.

> **RESOLVED (John, 2026-08-06), stronger than proposed.** A banner is not enough
> — assistants read banners and keep going. **Moved, not bannered:**
> `docs/specs/*` (13 files) now live in `docs/deprecated/` with a README whose
> first instruction is "never open these files," and CLAUDE.md carries the same
> prohibition at the very top so it is read before anything else. Executed
> 2026-08-06; Task 0b updated to record it.

---

## G18 — CODE-MISMATCH: user keyspace split — `user_sandboxes` TEXT vs `tasks` UUID, bridged by a lossy hash

**Status: FOLDED 2026-08-06 → migrations deleted, schema.md, auth.md.**

**Finding.** `user_sandboxes.user_id` is `TEXT PRIMARY KEY` (`0005`), keyed by the
raw JWT `sub`; `tasks.user_id` is `UUID` (`0001`). They are bridged by
`_user_uuid()` (`task_store.py:23`), which `uuid.UUID(s)` when possible and
otherwise `uuid5(NAMESPACE_URL, "ark-legacy:"+s)` — a **lossy, one-way hash** for
non-UUID demo subs. `0007`'s comment calls out exactly this ("the `user_id` was
TEXT vs the tasks UUID"). After auth unification, which id keys a sandbox?

**Proposed resolution.** **The split dissolves itself under Supabase auth.**
Supabase `sub` claims are UUIDs, and `auth.md` deletes `demo-login`/`ARK_DEMO_MODE`
(L33) — so there are no more non-UUID subs, and the lossy hash has nothing to
bridge. Concretely, on the `0c` migration-0 reset:

1. Recreate `user_sandboxes.user_id` as **`UUID`** (referencing the unified
   `users.id`), not `TEXT`.
2. **Delete `_user_uuid()`** and its call sites — the raw `sub` *is* the UUID
   everywhere (tasks, sandboxes, connections, blobs).
3. The **unified user UUID keys the sandbox**, same as tasks — one keyspace.

Note (secondary, not the keying question): the fate map keeps `user_sandboxes` as
**one persistent sandbox per user** (`contracts.md` L59), lazy-provisioned on first
sandbox-tool call (Task 8). Because G13 allows up to 5 concurrent worker sessions,
that one sandbox can have multiple concurrent writers — a "single actor per
resource" (`looking_glass_spec.md` L63) violation the keying decision does **not**
resolve. **Promoted to an escalation (E2 below) — it is John's call, not a
footnote.**

**Lands in.** Migration 0c (`single_loop_redesign_spec.md` L157: `user_sandboxes`
with `user_id UUID`); `contracts.md` § db fate map (L59: note `user_id` is now
UUID, hash bridge deleted); `auth.md` L33 (add `_user_uuid` to the deletion list).

**Confidence:** high.

> **RESOLVED (John, 2026-08-06), stronger than proposed.** Not "recreate as UUID"
> — **the old migration chain is trashed.** `db/migrations/0001`–`0007` are
> DELETED (executed 2026-08-06); they built tables the redesign does not build and
> nothing migrates forward. Migration 0 in `schema.md` stands alone, every
> `user_id` is a Supabase UUID by construction, and `_user_uuid()` dies with them.

---

# ESCALATIONS FOR JOHN

Only genuine owner-level calls are here — everything else above is a decisive
recommendation.

### E1 (from G9) — Does buddy get an `ask` tool, and do ordinary questions glow ochre?

**Status: FOLDED 2026-08-06 → D26 + spec L195 corrected.**

> **RESOLVED (John, 2026-08-05):**
> **(a)** Buddy gets a real `ask` / `request_input` tool that parks (chosen over
> ask-as-text). This DOES reverse spec L72-73's "ask = plain text, not a tool";
> that line must change when folded. Asks become first-class in `/attention` for
> free, and the lifecycle path stays deterministic (no model-output heuristic).
> **(b)** Every open question glows ochre — an unanswered buddy question marks its
> project ochre. Grid becomes a full "waiting on me" inbox. This ratifies
> `looking_glass_spec.md` L37 ("approvals AND asks") rather than narrowing it.
> Consequence to design for: an active chat with an open question stays ochre
> until answered; that is intended.

Two coupled decisions, each reversing or narrowing stated design.

**(a) Ask-as-tool vs ask-as-text.** The spec (L72–73) says a buddy ask is *plain
text ending the turn* — deliberately **NOT a tool**. But `/attention` and ochre are
driven by `status=awaiting_approval`, which only a park event/tool can set. So a
text-only ask cannot raise grid attention without *some* new plumbing.

> **Decision:** Give buddy a real `ask`/`request_input` tool that parks (contradicts
> the spec's text-only ask, but makes asks first-class in `/attention` for free) —
> OR keep ask as plain text and have the runner *infer* "turn ended on an unanswered
> question" to synthesize the park (honors the spec's tool list, but reintroduces
> model-output interpretation into lifecycle, which the deterministic-gate stance
> resists).

**Trade-off:** the tool is explicit and deterministic but adds a manifest entry the
spec says shouldn't exist; the text-only path keeps the manifest pure but puts a
heuristic back on the lifecycle path.

**(b) Grid semantics for non-blocking questions.** Even once (a) is settled, decide
what an *ordinary, non-blocking* buddy question does to the grid:

> **Decision:** Should an unanswered ordinary buddy question mark its project ochre
> — or is ochre reserved for sessions genuinely parked (buddy only glows when it
> actually blocks; normal chat questions leave the bubble gray/idle)?

**Trade-off:** ochre-on-every-question makes the grid a complete "everything waiting
on me" inbox but risks every active chat perpetually glowing; reserving ochre for
real blocks keeps the grid honest as a background-work monitor but a question you
navigated away from won't surface there. **Recommended default:** reserve ochre for
real blocks — cleaner grid — but it reinterprets `looking_glass_spec.md` L37
("approvals AND asks"), so it needs ratification.

---

### E2 (from G18 + G13) — One sandbox per user vs. concurrent sandbox workers

**Status: FOLDED 2026-08-06 → D18 + contracts resource rule + auth.md browser profile.**

`contracts.md` L59 keeps **one persistent sandbox per user**; the worker manifest
grants sandbox tools; G13 sets `max_concurrent_sessions: 5`. So one user can run
several `source=background` workers at once, all writing the **same** sandbox —
violating `looking_glass_spec.md` L63 ("v1 has a single actor per resource by
construction"). A real either/or:

> **Decision:** (i) **sandbox-per-session** — each worker gets its own sandbox;
> (ii) **sandbox-per-user + serialization** — keep one sandbox, but serialize
> sandbox-tool calls so only one worker touches it at a time; or (iii) **cap
> concurrent sandbox-writing tasks to 1** per user (chat and non-sandbox workers
> still run free).

**Trade-off:** (i) restores true single-actor isolation but multiplies e2b
cost/provisioning latency and breaks the "one persistent sandbox" fate-map line;
(ii) keeps one cheap warm sandbox but adds a cross-session lock and can stall a
worker behind another's long tool call; (iii) is simplest and matches L63 with no
new locking, but throttles a power user's parallel sandbox work to serial.

**Coupled sub-question (G13).** The concurrency check "count `status='running'`" is
itself a **check-then-act race** (two `create_session` calls can both read 4 and
both proceed to 6) — it needs the same serialization as the G3 append. And it is
**unstated whether a running buddy chat counts against the 5** or only background
workers do. Both must be nailed down for the cap to be well-defined.

**Recommended default:** (iii) cap concurrent sandbox-writing tasks to 1 per user
for v1 — satisfies L63 by construction, keeps one warm sandbox, and defers
per-session sandbox cost; buddy chat does **not** count against the sandbox cap
(only against `max_concurrent_sessions`). Revisit if users need parallel sandbox
work.

---

## E2 — RESOLVED (John, 2026-08-05): the resource-lease model, applied to sandbox AND browser

John chose option (ii) generalized: stateful hands are **persistent per-user
resources held under a lease**, and this now covers the browser too (persistent
per-user browser profile), not just the sandbox.

**The general contract (promote into `contracts.md`, replacing the deferred-only
"single actor per resource" line).** Serialize a resource iff it is BOTH shared
across a user's concurrent tasks AND carries mutable state between calls. Under
that test:

| Resource | Model |
|---|---|
| sandbox (`sandbox:{user}`) | persistent per-user; **leased** |
| browser (`browser:{user}`) | persistent per-user profile (cookies/logins survive); **leased** |
| MCP (Smithery) | stateless per call → **no lease**, runs free |
| session log | serialized per session (G3 advisory lock) — a different writer-race, not this |

**One primitive — the resource lease** (same shape as the multi-worker task lease):

```
acquire(resource_key, task_id)   # before first use of that resource
# hold for the WHOLE task, not per tool call:
#   per-call leasing corrupts stateful state (task B interleaves task A's half-write)
release(resource_key)            # on terminal OR on park (awaiting_approval)
# contended -> the task PARKS as "waiting for {resource}" and wakes when free
# parked task releases (it isn't acting); re-acquires on resume — state persisted,
#   so files/cookies are still there, it just re-takes the actor token
```

**State persistent, runtime cattle.** Both hands follow the same shape as the
sandbox already did: the *state* persists (e2b filesystem / browser profile dir),
the *running instance* is cattle — lazily booted on first use, torn down when
idle, respawned against the persisted state on next lease. "Persistent" means the
profile survives, not that a browser stays warm.

**Concurrency, restated.** `max_concurrent_sessions: 5` bounds total tasks; each
resource lease independently bounds that resource to one holder. A user can run 5
tasks where at most one holds the sandbox and one holds the browser; the rest do
MCP work in parallel or park on a lease. Buddy chat holds no leases (interactive
manifest has neither hand).

**New implications to carry into the specs:**
1. **Security (auth.md):** a persistent browser profile now stores logged-in
   session cookies — credential-equivalent, per-user, must be protected like the
   sandbox and never exposed in events/logs. New sensitive at-rest state.
2. **Task 9 (browser):** add a per-user profile store (Browserless persistent
   context / `user-data-dir`) + the `browser:{user}` lease. Browser is no longer
   "fresh session per call."
3. **Config:** lease keys and a lease-wait timeout join the config block; a task
   that waits too long for a lease parks rather than fails.
4. **G13 count-then-act:** the concurrency-cap check and lease `acquire` share the
   G3 serialization mechanism (advisory lock), so this is one problem, not three.

**Coupled sub-question — RESOLVED (John, 2026-08-05):** a running buddy chat does
NOT count against `max_concurrent_sessions`; the cap governs background workers
only. E2 fully closed.

---

# MISSING ENTRIES — G1–G13, G19–G21

**These are not in this file and I cannot restore them from memory.** When this
session began, the file's header already claimed G1–G11 and G19–G21 were folded
into the law docs on 2026-08-05 and their text had been removed. G12 and G13 were
removed by me at the end of the prior session, after folding. I never read
G19–G21 at all.

The file was never committed to git, so `git show` cannot recover them either.

**Recoverable from:** the session transcript (both sessions), which still contains
the gap-finder subagent's original output. That is the only remaining source.

**Do not treat "folded" as verified for G1–G11 and G19–G21.** It is an inherited
claim, not something checked against the law docs.
