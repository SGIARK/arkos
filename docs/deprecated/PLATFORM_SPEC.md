# Platform Spec: ARKOS as a Persistent Per-User Computer

*The overarching architecture the feature specs sit underneath. Not to be confused with `ARCHITECTURE_SPEC.md`, which is about the agent-loop altitude (Gat/options/capability-dial). This doc is about the system: the per-user computer, the agents that operate it, and how the user reaches it.*

**Sources**

- This design conversation (2026-06-02) -- the convergence from "feature-first" to "infra-first"
- e2b (sandboxes-for-AI-agents: persistent filesystem, pause/resume, exec API) -- the sandbox infra candidate
- Smithery (MCP proxy: hosts MCP servers, holds OAuth tokens write-only, proxies calls)
- THREAD (recursive agent spawning) -- referenced for where recursion belongs
- Claude Agent SDK -- the lean agent loop over a sandbox
- Companion specs: `HARNESS_SPEC.md`, `MEMORY_SPEC.md`, `MULTIUSER_SPEC.md`, `LOGGING_SPEC.md`, `ARCHITECTURE_SPEC.md`, `OPENHANDS_SPEC.md` (now largely superseded -- see below)

**Status:** Vision settled, MVP slice defined | **Author:** | **Last updated:** 2026-06-02

---

# The Vision

The dev process is backwards: to workshop a spec, write a feature, or open a PR, you have to be at a keyboard. The goal is to talk to an agent from a phone and have it do real work -- read the codebase, draft a grounded spec, implement it, manage files.

The realization that reorganizes everything: **we were building outside-in (features first), but the real need is infra at the core.** Every feature discussed -- specs, PRs, file uploads, persistent workspace, computer-capable subagents -- bottoms out on the *same* primitive. Build it once and the features become thin layers; skip it and every feature reinvents a slice of it.

**The core primitive: each user has a persistent, isolated computer that agents operate on their behalf, with full visibility and control.**

This is the OpenHands / computer-use paradigm, but reframed: the *computer* is the core object, not the agent. The agent is a swappable guest in it.

---

# The Core Primitive

A **per-user, persistent, isolated sandbox with an exec API** -- a durable filesystem plus the ability to run commands, that both the agent and the user operate on.

- **Persistent filesystem** -- always exists, accumulates over time. The user's files and work live here.
- **Ephemeral compute** -- the sandbox wakes on use, sleeps when idle. State persists; you only pay compute while working. From the user's side the computer is "always there"; under the hood it is not running 24/7.
- **Isolated** -- per user; A can never read or touch B's computer. This is the security-critical part and the reason to buy rather than build the sandbox.
- **Headless, terminal-first** -- a shell + filesystem, *not* a GUI computer-use agent (no screenshots/clicking). Terminal + files covers ~90% of real work (run code, edit, git, install, test) and is faster, more reliable, and fully auditable. GUI computer-use is deferred indefinitely.

Persistence model: start with e2b **pause/resume** (sandbox hibernates with full state, wakes in seconds); move to **persistent-volume + fresh-sandbox** if cost/scale demands. Keep the sandbox warm during an active session; spin down on idle. Wake latency is a few seconds -- fine for a chat-driven agent.

---

# The Architecture

```
   User ──owns──►  PERSISTENT SANDBOX (their computer)        ← the infra core (e2b)
     │                   ▲            ▲   persistent FS, ephemeral compute, isolated
     │ chats             │ reads      │ reads + writes + runs
     ▼                   │ (eyes)     │
   BUDDY (Qwen) ─────────┘            │                        ← cheap front + router
     │  router + conversational home  │
     │                                │
     ├─ trivial lookup ──► buddy-direct MCP call (no computer)
     │
     └─ real work ──────► COMPUTER-AGENT (local model now → frontier later)  ← the one worker
                            one worker, holds BOTH natively:
                              - file/shell tools (the sandbox)
                              - MCP tools (via Smithery proxy)
                            plans + recurses internally (THREAD lives here)
     │
     └─ user also sees/uploads via the PROJECTS UI ─► the same sandbox   ← the window
```

Four layers, decoupled:

1. **Sandbox infra (the computer)** -- persistent, isolated, per-user. Buy it (e2b) -- isolating user+agent code execution is the hard, security-critical part you do not hand-roll.
2. **Computer-agent (the worker)** -- a lean agent loop with `run_command` / `read_file` / `write_file` over the sandbox exec API, *plus* MCP tools natively (like Claude Code). One worker, not two. **Runs on the local Qwen for the MVP** (proves the plumbing for free); swaps to a frontier model later via one config line -- which is exactly the point of owning the sandbox separately.
3. **Buddy (the front)** -- Qwen. Owns/manages the user's computer, routes requests, is the conversational home and the user's window. Cheap, always-on.
4. **Projects UI (the window)** -- browse/upload/download the sandbox + watch live agent activity.

---

# Key Decisions (and why)

| Decision | Rationale |
|---|---|
| **Decouple sandbox from agent** | The computer is the core object; the agent is a swappable guest. Keeps us un-locked-in down the road. |
| **Buy the sandbox (e2b), don't build** | Isolating arbitrary user+agent code is the non-trivial, security-critical 80%. e2b exists for exactly this. Self-host on Docker+volumes later only if cost demands. |
| **OpenHands demoted / likely dropped** | OpenHands' main gift was a bundled sandbox. Once we own the sandbox (e2b), the agent is just a lean Claude loop -- OpenHands' whole system is mostly redundant. `OPENHANDS_SPEC.md` is superseded except as a reference. |
| **Headless terminal, not GUI computer-use** | Terminal + files = ~90% of work, faster + more reliable + auditable. GUI CUA is a deferrable rabbit hole. |
| **Persistent FS + ephemeral compute** | "Always there" without 24/7 cost. Pause/resume now, volume-mount later. |
| **One worker, not two** | Don't maintain a separate MCP subagent *and* a compute subagent (duplicate contexts/state). Collapse to one computer-agent that holds file/shell *and* MCP tools. |
| **Buddy routes by task weight** | Trivial single lookup -> buddy-direct MCP (no computer, no Claude). Real/multi-step work -> the computer-agent. Don't wake the computer for "what's on my calendar." |
| **Computer-agent model: local Qwen now, frontier later** | Owning the sandbox separately makes the agent model a swappable knob. Run Qwen for the MVP to prove the plumbing for free; swap `computer_agent.llm` to a frontier model (Claude) when quality matters. Caveat: Qwen does real computer tasks poorly, so early demos prove "it works end to end," not "it codes well," and the step cap is load-bearing. |
| **Planning/recursion lives in the worker layer** | Decompose + integrate is hard reasoning. Buddy (Qwen) routes; it does not master-plan. THREAD-style recursion happens *inside* the computer-agent, not at buddy. (This pays off most once the worker is a frontier model.) |
| **Buddy reads, subagents write** | Buddy (weak model) gets read-only eyes on the computer for cheap lookups + routing. All writes/runs go through the Claude computer-agent. Never give the weak model hands. |
| **MCP via Smithery proxy, raw token only for git** | Smithery holds MCP tokens write-only and proxies calls -- the sandbox-agent makes proxied calls, no raw token needed. Only `git push` needs a real token (the GitHub PAT, collected in the connections panel). |

---

# Interaction Model -- how the user experiences it

**Buddy is the conversational home and the entry point.** The user talks to one assistant. Buddy routes, hosts, and relays final results -- it is *not* a lossy relay that re-narrates or re-interprets subagent output.

After buddy dispatches a worker, the subagent communicates with the user **directly, through one shared surface** (not as a separate voice/session):

| Subagent -> user | How it flows |
|---|---|
| **Work / progress** (file edits, command output) | streamed **directly** to the activity/projects view. Buddy does not re-narrate. |
| **Structured asks** (approve? which repo?) | the subagent's *own* question surfaces in the UI; the answer routes **straight back** to it. Buddy is a passthrough channel, never a re-interpreter (so Qwen never garbles Claude's question). This is the existing approval-tray pattern. |
| **Final result** (the doc, the PR link, the files) | appears as the subagent's output; buddy adds a light conversational frame at most. |

Hard rule: **never route a subagent's message through Qwen for re-interpretation.** Buddy routes and hosts; it does not paraphrase Claude.

The existing **"needs your input" tray** already implements the structured-ask surface. On mobile, decide inline-in-chat vs. tray by feel after the spike, not on paper.

Analogy: buddy is the project manager you talk to; the computer-agent is the worker. You don't chit-chat with the worker, but you can watch it work, and when it hits a decision that needs you, that specific question reaches you and your answer goes straight to it.

---

# Routing Logic

Buddy makes one constrained decision per request (the kind of routing Qwen can do):

| Request | Path | Cost |
|---|---|---|
| Conversational answer | buddy answers directly | cheap |
| Trivial single external lookup ("calendar today?") | buddy-direct MCP call, no computer | cheap |
| Anything real: needs files, runs code, chains tools, multi-step | wake the computer-agent (Claude, in the user's sandbox, with MCP tools) | pays compute + Claude, earned by the task |

The line is **task weight**, and it is tunable. The point: one worker surface for real work, a cheap front for cheap things, no duplicate workers.

---

# What Already Exists vs. What's New

Right-sizing the build -- it is smaller than the vision looks.

**Exists:**
- Buddy chat agent (Qwen front) + state graph
- MCP path via Smithery (calendar, etc.)
- The approval / "needs your input" tray
- Harness resilience (ModelError+retry, validation, named outcomes, budgeting -- shipped this session)
- Async background-task skeleton (`task_runner` + `task_events`) to reuse for worker dispatch

**New (the actual build):**
- The persistent per-user sandbox (e2b) -- the infra core
- The computer-agent (lean Claude loop + file/shell + MCP tools)
- Buddy's dispatch route to the computer-agent
- The projects UI (browse/upload + activity stream)
- Multiuser verified identity (gate before web exposure)

---

# The MVP Slice

Do **not** build the system. Build the spine, prove it, add one layer at a time.

**IN (the spine):**
- One e2b sandbox per user (persistent FS, pause/resume), single trusted user to start
- A lean agent loop **on the local Qwen** with `run_command`, `read_file`, `write_file` **plus MCP tools natively** (like Claude Code), over the sandbox
- Buddy can dispatch "this needs the computer" -> wake the sandbox -> run the computer-agent
- Results + structured asks land in the existing chat + tray
- A **Computer tab** to view the user's filesystem, and a **"using the computer" indicator** on subagents that stream their commands/thoughts

That is the entire "phone -> persistent computer -> real work" loop, visible to the user.

**OUT (deferred -- add when):**
- **Frontier computer-agent model** -- runs on local Qwen for now (proves plumbing); swap `computer_agent.llm` to Claude when task quality matters. One-line change.
- **OpenHands** -- dropped; e2b + lean local-model loop replaces it.
- **GUI computer-use (CUA)** -- add only if a GUI-only task ever appears.
- **Memory layers** (`MEMORY_SPEC`) -- buddy's existing memory is enough; add when sessions need real recall.
- **Audit table** (`LOGGING_SPEC` Task 4+) -- do `LOGGING_SPEC` Task 1 (structlog config, so you can *see*) now; defer the audit DB until multi-user/compliance needs it.
- **Projects UI polish** -- the Computer tab ships read-only (browse + view); upload/in-browser-edit/download come in v2.
- **PR / git flow** -- output is files in the sandbox first; add `git push` + PR (needs the GitHub PAT path) in v2.
- **THREAD recursion** -- start with a flat single worker; add recursive spawning inside the agent when a task needs decomposition.
- **Multiuser hardening** (`MULTIUSER_SPEC` Tasks 1-2) -- prove the loop solo first; this is the gate *before* opening it to the web.

**The first move is a one-day spike, not a build:** stand up a sandbox, write a file, tear down compute, bring it back, confirm the file persisted; then run one agent command in it. If that works, the spine is proven.

**Implementation breakdown:** `COMPUTER_SPEC.md` is the single self-contained **mega spec** for this spine, written to be implemented end-to-end in one pass. It folds in the identity/registry tasks from `MULTIUSER_SPEC` (as Phase 0 -- the gate), the per-user sandbox manager, the local-Qwen computer-agent with native MCP, buddy's dispatch, the **completion-notification** mechanism (chat injection + SSE), the progress/ask surface, the Computer tab + "using the computer" indicator, and the config rename. Read PLATFORM for the why; read COMPUTER for everything to build.

---

# How the Other Specs Fit

| Spec | Role under this platform |
|---|---|
| `HARNESS_SPEC` | The Qwen agent-loop resilience (buddy's loop). Done. The Claude computer-agent gets resilience from the Agent SDK. |
| `ARCHITECTURE_SPEC` | The agent-loop altitude / capability-dial. The "buddy routes, Claude plans" split is the same removable-prior principle applied at the platform level. |
| `MEMORY_SPEC` | Per-user memory. Layers onto buddy; the sandbox filesystem is also a form of durable memory (procedural/working). Deferred for MVP. |
| `MULTIUSER_SPEC` | Verified identity + isolation. Tasks 1-2 are the gate before web exposure. Sandbox isolation (e2b) covers the filesystem side. |
| `LOGGING_SPEC` | Visibility. Task 1 (structlog) now so the spike is debuggable; audit later. |
| `OPENHANDS_SPEC` | Superseded by the decouple decision. Keep as a reference for the agent-with-a-computer paradigm; not the build path. |

---

# Open Questions

1. **Sandbox: e2b vs self-host.** Buy (e2b) for MVP is the call. Re-evaluate self-hosted Docker+volumes once cost at real usage is known. *Measure before optimizing.*
2. **Persistence flavor.** Pause/resume (simple, holds the sandbox) vs volume-mount (stateless compute, scales). Start pause/resume; the agent code is identical either way (it talks to an exec API).
3. **Routing threshold.** Exactly where "buddy-direct" ends and "wake the computer" begins. Trivial single reads stay direct; everything multi-step goes to the computer. Tune from real use.
4. **Computer-agent: Agent SDK vs custom loop.** SDK gives the loop + tool handling for free (Claude-only, which is what we want here). Custom gives control we probably don't need. *Leaning SDK.*
5. **GitHub PAT UX.** Collected in the connections panel as a "bring your own token" entry alongside Smithery OAuth services. Only needed once the git/PR flow lands (v2).
6. **Idle/cost policy.** When does a warm sandbox spin down? Storage quota per user? Decide once real usage exists.

---

# Implementation Notes

*Add entries here as work lands.*

- (north star) The core object is the **per-user persistent computer**. Everything else -- buddy, agents, UI, specs, PRs -- is a layer on it. When a design question is unclear, ask "what does this mean for the user's computer?"
- (anti-scope-creep) The vision is the map; the MVP is one road. Build the spike first; add exactly one deferred layer at a time, driven by a real need, not by the map.
- (the decouple dividend) Owning the sandbox separately is what lets the agent stay lean and swappable and what dropped OpenHands from the critical path. Protect that separation -- keep the agent talking to a generic exec API, never coupled to a specific sandbox vendor's internals.
- (capability dial, again) Buddy=Qwen routes; the computer-agent=Claude plans and works. The boundary "router vs planner" is a removable prior: if buddy ever runs a frontier model, the line can move up. Build so moving it is a config change, not a rewrite.
