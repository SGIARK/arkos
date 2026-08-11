# Feature Spec: ARKOS Harness Architecture — Graph as Constitution

**Sources**

- Gat, *On Three-Layer Architectures* (1998) — organize by use of state/time: Controller / Sequencer / Deliberator
- Sutton, Precup & Singh, *Between MDPs and semi-MDPs* (1999) — the options framework: a behavior is ⟨initiation I, policy π, termination β⟩
- Sutton, *The Bitter Lesson* (2019) — general methods leveraging computation beat hand-encoded structure as capability scales
- Levin/Wulf et al., Hydra (1975) — separation of mechanism and policy
- Sumers, Yao, Narasimhan & Griffiths, *Cognitive Architectures for Language Agents (CoALA)* (2023) — memory as belief state; internal vs external actions; the decision cycle
- Yao et al. *ReAct* (2022); Shinn et al. *Reflexion* (2023) — reason+act and self-reflective inner loops
- Current code: `agent_module/agent.py` (`step`, `choose_transition`), `state_module/core/` (`State`, `StateOutput`, `StateHandler`), `state_module/agent_buddy/` (`reply/tool/ask` graph), `state_module/agent_executor/` (scaffolded `plan_steps`, `state_approval`)
- Companion specs: `HARNESS_SPEC.md` (resilience/mechanism), `MEMORY_SPEC.md` (belief state), `MULTIUSER_SPEC.md` (isolation)

**Status:** Not started — **scheduled last**, after the harness/multi-user/memory P0 work lands. This is a strategic refactor, not an MVP blocker; it *consumes* the mechanism that those specs build.

---

# Problem

The single design axis that governs both "is the harness good?" and "small model vs. frontier?" is: **how much of the agent's policy lives in code vs. in the weights?** Today ARKOS puts task policy in code (the YAML graph routes individual tool calls), which produces three coupled defects:

1. **Wrong layer (Gat).** Tool-use — a *Controller*-level activity (memoryless act→observe loop) — is encoded as *Sequencer* structure (`reply → tool → ask` graph nodes), with a *Deliberator* LLM call (`choose_transition`, `agent.py:149`) between every step. Three architectural layers that differ in their use of state and time are collapsed into one. This is the root of the rigidity *and* the per-tool LLM tax.
2. **Permanent scaffolding (bitter lesson).** Structure that exists only because a 7B model is weak — constrained tool choice, forced schemas, scaffolded plans — is baked in with no expiry. As the model improves, that structure stops being regularization and becomes pure bias: it caps a frontier model below its own policy. There is no mechanism to retire it, so improving the model requires a rewrite, not a config change.
3. **Mechanism and policy are fused (Hydra).** Invariants that must hold for *any* model (approval before destructive actions, auth handoffs) live in the same graph, in the same idiom, as task policy that should belong to the model. You cannot tell, from the graph, which edges are guarantees and which are guesses.

**What this costs:** every capability is graph surgery; a frontier model is throttled through enum choices and one-tool-at-a-time steps; and "make it work for a better model" is a project, not a setting.

**Success looks like:** the graph encodes only *invariants* (it becomes the agent's constitution, not its program); each node is a formal option with a checkable contract; the inner policy is model-driven and its scaffolding is *explicitly removable* with a named capability trigger; and the *same graph* runs a constrained 7B policy and a native frontier policy with no fork — policy slides from code to weights continuously as the model earns it.

---

# Technical Background

The design is derived from five load-bearing principles. Each maps to a concrete decision rule.

**1. Organize by use of state and time, not granularity (Gat 1998).** A three-layer architecture separates: **Controller** (memoryless, tight feedback — the in-state tool loop), **Sequencer** (stateful, selects the active behavior, no time-extensive compute — the graph), **Deliberator** (unbounded time, search/planning — explicit plan calls). *Rule:* a decision belongs in the graph only if it is a sequencing decision (rare, stateful, fast), never if it is work (continuous) or planning (time-extensive).

**2. A state is an option ⟨I, π, β⟩ (Sutton, Precup & Singh 1999).** Initiation set (when it may start), internal policy (what it does), termination condition (when it's done). *Rule:* `check_transition_ready` is β, a contract — not ad-hoc logic. The graph is a policy over options (a semi-MDP). π is swappable as long as I and β hold; the *contract* is invariant, the *policy* is not.

**3. Structure is a prior; its strength must anti-correlate with capability (bias–variance + bitter lesson).** Hand-authored structure regularizes a high-variance (small) model and *over-constrains* a low-variance (frontier) one. *Rule:* every structural element carries an explicit **removal trigger** — the capability threshold past which it becomes bias. Structure with no nameable removal trigger is permanent scaffolding and should not be added.

**4. Separate mechanism from policy (Hydra 1975).** *Mechanism* (tool execution, validation, retries, memory, invariant enforcement) is fixed and lives in the harness — the control plane, rare and correctness-critical. *Policy* (which tool, what plan, when done, what to say) is the data plane and migrates to the model. *Rule:* only mechanism/invariant decisions may be fixed in code; task policy in code must be deletable per Principle 3.

**5. It is one cognitive architecture (CoALA 2023).** A language agent = memory (working/episodic/semantic/procedural — the `MEMORY_SPEC.md` layers), an action space (internal: reason/retrieve/learn; external: tools), and a decision cycle. *Rule:* working memory is the belief state the option's π conditions on; the decision cycle *is* π; procedural memory is the channel through which code-structure becomes learned skill as it leaves code.

**Key constraint:** this refactor depends on the harness resilience layer (`HARNESS_SPEC.md`) existing first — validation, retries, named outcomes, and budgeting are the *mechanism* an option's π runs inside. Build mechanism, then move policy onto it.

---

# Proposed Approach

Reshape the harness into the three-layer architecture, derived from the principles above — not asserted:

- **The graph becomes the Sequencer over options** (P1, P2). Nodes are invariants only — e.g. `converse`, `plan`, `approve`, `execute`, `recover`. It does no work and makes no task-policy decisions. This is the "constitution," not the "program."
- **Each node is a formal option ⟨I, π, β⟩** (P2). I and β are harness-enforced contracts (this is where the existing deterministic-gate instinct was correct — it needed the options formalism). π is pluggable.
- **π is a ReAct/Reflexion inner loop** (P1, P5) running as Controller+Deliberator, whose *structure is a sized prior* (P3) with a named removal trigger. 7B: constrained, short-horizon, harness-validated. Frontier: native tool-calling, long-horizon, self-reflective. Same option; scaffolding is deleted as triggers fire.
- **Mechanism vs. policy is enforced** (P4). The harness owns tools/validation/retries/invariants permanently; the model owns task policy, with code-policy only where a removal trigger justifies it.
- **Memory is the belief state** (P5). The option conditions on working memory, writes episodic/semantic, and learns procedural — wiring this spec to `MEMORY_SPEC.md`.

The load-bearing artifact is the **removal-trigger table** (below): every current structural element, the capability that justifies it, and the signal that retires it. It is what makes "good for small, better for frontier" a *verifiable property* rather than a claim.

Explicitly **not in scope:** learned/RL-optimized routing (the sequencer stays hand-authored — it encodes invariants, which are exactly the thing you do *not* want learned); replacing the YAML graph format; multi-agent topology changes.

---

# The Removal-Trigger Table

The centerpiece. Two classes of structure: **priors** (scaffolding for a weak model — must be removable) and **invariants** (guarantees for any model — must be permanent). A piece of structure that can't be placed in one column is a design smell.

| Current structure | Class | Justifying capability gap | Removal trigger (retire when…) |
|---|---|---|---|
| `choose_transition` LLM call between states (`agent.py:149`) | Prior | 7B can't reliably self-route | Model emits next-action natively (tool-call or explicit done) with ≥X% valid-routing in eval |
| `state_tool` as a graph node + 2-call select→fill (`state_tool.py:48,66`) | Prior | 7B emits invalid tool calls | Model emits schema-valid native tool calls (single turn) above error budget |
| Forced JSON schema on every state output | Prior | 7B malformed structured output | Model's native structured-output validity clears the threshold; keep schema only at true I/O boundaries |
| Scaffolded `plan_steps` iteration (`agent_executor`) | Prior | 7B can't plan + track multi-step | Model plans internally and self-tracks across a long-horizon loop without drift |
| Short `max_iter` / step horizon (`agent.py:20`) | Prior | 7B wanders / fails to self-terminate | Model self-terminates reliably; horizon raised, not removed (still a safety bound) |
| One-tool-at-a-time execution | Prior | 7B can't manage parallel/branching tool use | Model issues correct parallel tool calls; enable concurrency |
| **Approval gate before destructive actions** (`state_approval.py`) | **Invariant** | — (not a capability gap) | **Never.** A frontier model doesn't earn the right to skip it. |
| **Auth handoff** (`AuthRequiredError`) | **Invariant** | — | **Never.** Credentials are mechanism, not policy. |
| **Mode boundary `converse↔execute`** (commit to long execution) | **Invariant** | — | **Never** as a boundary; *what triggers* the crossing may become model-led. |
| **`recover`/escalate on budget-exceeded or error** | **Invariant** | — | **Never.** The safety net for when in-state agency fails. |

The discipline: **priors shrink as the model grows; invariants are constant.** The graph in the limit (perfect model) collapses to *only the invariant column* — that is the architecture's asymptote, and it should be visible from day one.

---

# Implementation Plan

## Task 1: Formalize `State` as an option ⟨I, π, β⟩

**Problem:** States have implicit initiation/termination; transition logic is ad-hoc, which is why it's the brittle part.

**Done when:** `State` (or a new `Option` base) carries explicit `initiation(ctx) -> bool` (I), `run` (π), and `termination(output) -> bool` (β) as first-class, harness-checked contracts; `StateHandler` enforces I before entry and β before exit; existing `check_transition_ready` is re-expressed as β.

**Touch point:** `state_module/core/state.py`, `base_state.py`, `state_handler.py`.

**Priority:** P2 | **Effort:** ~2 days | **Blockers:** `HARNESS_SPEC.md` Task 3 (named outcomes feed β).

**Out of scope:** Changing any agent's actual graph yet (pure formalization pass).

**Acceptance test:** `test_option_contract_enforced` (below).

---

## Task 2: Demote tool-use from Sequencer to Controller — collapse `reply/tool/ask` into one `converse` option

**Problem:** The agentic loop is spread across graph nodes with a Deliberator call between steps (the wrong-layer defect).

**Done when:** `agent_buddy` is one `converse` option whose π is an in-state ReAct loop (reason → tool → observe → continue), bounded by β and the step horizon; `state_tool` and the inter-node `choose_transition` are gone from the buddy graph; the loop reuses `HARNESS_SPEC.md` mechanism (validate, retry, budget). The graph for buddy shrinks to `converse → respond` plus the invariant nodes.

**Touch point:** `state_module/agent_buddy/*`, `agent_module/agent.py` (loop runner).

**Priority:** P2 | **Effort:** ~4 days | **Blockers:** Task 1; `HARNESS_SPEC.md` Tasks 1–4.

**Out of scope:** Executor graph (Task 4); native vs constrained policy switch (Task 3).

**Acceptance test:** `test_converse_runs_tool_loop_in_state`, `test_buddy_graph_has_no_tool_node` (below).

---

## Task 3: Sized-prior policy with removal triggers (the capability layer, grounded)

**Problem:** There is no mechanism to make scaffolding model-relative or removable; improving the model means a rewrite.

**Done when:** An option's π is selected from a **policy prior** that is *derived from the removal-trigger table*, not free-form knobs — each prior names the capability gap it fills. Two concrete π implementations exist behind one interface: `ConstrainedPolicy` (enum choose + schema fill + short horizon) and `NativePolicy` (native tool-calling + long horizon + self-reflection). The harness binds the right one from the model tier; the *contract* (I, β) is identical across both.

**Touch point:** `agent_module/` (policy interface + two implementations), config (tier→prior binding).

**Priority:** P2 | **Effort:** ~4 days | **Blockers:** Task 2.

**Out of scope:** Auto-detecting model capability (bind explicitly from config for now); learned policies.

**Acceptance test:** `test_same_option_runs_constrained_and_native`, `test_invariant_holds_under_both_policies` (below).

---

## Task 4: Enforce mechanism/policy separation — invariants become harness-level, not graph edges

**Problem:** Invariants (approval, auth, recover) are indistinguishable from task policy in the graph.

**Done when:** The invariant nodes from the removal-trigger table are enforced by the harness as non-bypassable mechanism (a policy *cannot* route around approval/auth/recover regardless of what it emits); the executor graph is reshaped to `plan → approve → execute(option) → done` with `execute`'s π dialed per Task 3; planning becomes model-led where the trigger has fired, scaffolded otherwise.

**Touch point:** `agent_module/agent.py`, `state_module/agent_executor/*`, `state_handler.py`.

**Priority:** P2 | **Effort:** ~3 days | **Blockers:** Tasks 1–3.

**Out of scope:** Removing the approval gate (it's an invariant — never).

**Acceptance test:** `test_policy_cannot_bypass_invariant` (below).

---

## Task 5: Wire memory as belief state (CoALA), connecting to MEMORY_SPEC

**Problem:** Memory and harness are designed as separate systems; CoALA says they're one.

**Done when:** An option's π conditions on working memory and emits internal actions (retrieve/learn) alongside external (tools); procedural memory is the documented channel for retired scaffolding (a removed prior's behavior is captured as a learned/skill memory, not lost). This is a thin integration task once `MEMORY_SPEC.md` Stages 2–3 land.

**Touch point:** `memory_module/`, `agent_module/`, option π interface.

**Priority:** P2 | **Effort:** ~2 days | **Blockers:** Tasks 1–3; `MEMORY_SPEC.md` Stages 2–3.

**Out of scope:** Learning the priors automatically (manual capture for now).

**Acceptance test:** `test_option_conditions_on_working_memory` (below).

---

# Tests

## Test 1: test_option_contract_enforced
**What it verifies:** An option cannot be entered when I is false and cannot exit until β is true; β is derived from the typed outcome, not an LLM call.
**Why this matters:** Formalizing I/β is what turns brittle ad-hoc transitions into a checkable contract — the foundation everything else rests on.

## Test 2: test_converse_runs_tool_loop_in_state / test_buddy_graph_has_no_tool_node
**What it verifies:** `converse` performs a multi-step tool loop internally and returns one `StateOutput`; the buddy graph contains no `tool` node and no inter-node `choose_transition`.
**Why this matters:** Pins the wrong-layer fix — Controller work is in the Controller, not the Sequencer.

## Test 3: test_same_option_runs_constrained_and_native / test_invariant_holds_under_both_policies
**What it verifies:** The identical option, with policy bound to `Constrained` vs `Native`, completes the task under both; the option's contract (I, β) and any invariant hold regardless of which policy ran.
**Why this matters:** This is the "one harness, two model tiers, no fork" property — the core claim of the spec, made testable.

## Test 4: test_policy_cannot_bypass_invariant
**What it verifies:** No policy output — including a maliciously/erroneously crafted one — can route around the approval or auth invariant; the harness enforces it as mechanism.
**Why this matters:** Separates guarantee from guess; a frontier model's extra agency must not be able to skip a constitutional invariant.

## Test 5: test_option_conditions_on_working_memory
**What it verifies:** An option's policy reads working memory as belief state and its internal retrieve/learn actions are dispatched through the memory layer, not bespoke code.
**Why this matters:** Confirms harness and memory are one cognitive architecture (CoALA), not two bolted-together systems.

---

# Open Questions

1. Capability binding: bind model tier → policy prior from config (explicit, proposed) or detect from eval metrics? Explicit is safer and auditable; auto-detection risks silently promoting a model past its competence. *Leaning explicit; revisit when an eval harness exists.*
2. Where does a *retired* prior's behavior go — pure deletion, or captured as procedural/skill memory (Task 5) so it's recoverable if a model regresses? *Leaning: capture, don't delete, until the trigger is proven stable.*
3. The sequencer stays hand-authored (invariants shouldn't be learned). But *crossing* `converse→execute` could be model-led once the trigger fires — does that blur the invariant? Distinguish "the boundary exists" (invariant) from "what triggers crossing it" (policy). Validate the line holds.
4. Does collapsing fine states into coarse options hurt observability (fewer, larger nodes to trace)? May need richer in-option tracing (CoALA decision-cycle events) to keep the system debuggable — coordinate with the logging sweep in `HARNESS_SPEC.md` Task 6.

---

# Implementation Notes

*Add entries here as work lands.*

- (sequencing) This spec is scheduled **last** and *depends on* `HARNESS_SPEC.md` mechanism. Do not start Task 2 before Harness Tasks 1–4 land — the in-state loop must run on the validated/retrying/budgeted mechanism, or you rebuild brittleness inside the new structure.
- (principle) The asymptote test for any proposed structure: "name its column in the removal-trigger table." If it's neither a removable prior nor a permanent invariant, it doesn't belong.
- (cross-link) Task 5 is the seam where `ARCHITECTURE`, `MEMORY`, and the CoALA framing meet — keep the option's π interface and the memory belief-state interface co-designed.
