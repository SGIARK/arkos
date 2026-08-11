# Spec: The Computer-Agent (the worker)

*Detailed design of the persistent agent that operates the user's computer. Sits under `computer_module` as `ComputerAgent` and is the worker that `COMPUTER_SPEC.md` Task 5 points to. The scaffolding (system-prompt structure, the agent-computer tool interface, the read-before-edit / search-first discipline) is **borrowed as paradigms** from the Claude Code reference in `arkos-inspo/claude-code` -- studied for structure, then **re-authored in our own voice**. We do not copy its text or code (licensing); we copy the engineering ideas, which is what `arkos-inspo` is for.*

**Sources**

- `arkos-inspo/claude-code/src/utils/systemPrompt.ts` (layered prompt assembly), `src/tools/FileEditTool/prompt.ts` (read-before-edit + unique-string + prefer-edit discipline), `src/tools/{BashTool,FileReadTool,GlobTool,GrepTool,TodoWriteTool}` (the agent-computer interface)
- `computer_module/sandbox.py` (`SandboxManager` -- the tools execute against this; already built + verified)
- `COMPUTER_SPEC.md` (parent: dispatch, completion notification, surface, the gate), `PLATFORM_SPEC.md` (the persistent-computer vision)
- `model_module/ArkModelNew.py` (`ArkModelLink`, OpenAI-compatible client)

**Status:** Not started | **Author:** | **Last updated:** 2026-06-03

---

# What this is

A **persistent, capable agent that operates one user's computer** (their e2b sandbox) to accomplish a task end-to-end: read and search the filesystem, run commands, edit files, call MCP tools, verify its own work, and report a result. It is the "serious intelligence" worker -- buddy (Qwen) routes to it; it does the actual work.

Two commitments from the surrounding decisions:
1. **Persistent + separate.** It operates the user's durable sandbox (not a fresh per-task box), so work accumulates. It is a separate agent from buddy, dispatched async.
2. **Competitive scaffolding, swappable model.** Its prompts and tool interface are borrowed from Claude Code's proven design so it is competitive out of the gate. The *model* is a config knob (`computer_agent.llm`) -- point it at a frontier model for real capability, or at the local SGLang/Qwen to prove the loop. The scaffolding does not change with the model.

What it is **not**: it is not buddy (no routing/chat), not the sandbox (that's `SandboxManager`), and not a fresh-sandbox-per-task worker (that's OpenHands -- considered and set aside in favor of the persistent-computer vision).

---

# The scaffolding (the part that makes it competitive)

Three borrowed paradigms carry most of the quality. They are the spec's centerpiece.

## 1. A layered, structured system prompt

Claude Code's prompt is assembled in layers (default identity + environment + optional agent/append). We mirror the structure. The prompt is re-authored below in our voice; it is the agent's operating manual.

```
You are the ARKOS computer-agent. You operate a persistent Linux computer on
behalf of the user and complete their task by using tools -- running commands,
reading and editing files, searching, and calling connected services. You act;
you do not just describe.

# Environment
- You are working inside the user's persistent sandbox. Files you create persist
  across sessions. The working directory is {cwd}. This is the user's own computer.
- Today is {date}. The user is {username}.

# Tone
- Be concise and direct. Do not pad with preamble or restate the task back.
- Explain a non-obvious command or plan in one line before running it; otherwise
  just act. The user sees your tool calls, so don't narrate every step.
- Report results plainly: what you did, what you found, where the output is.

# How you work
1. UNDERSTAND first. Before changing anything, use search (grep/glob) and read the
   relevant files. Never edit a file you have not read.
2. PLAN multi-step work. For anything beyond a couple of steps, write a short plan
   with the todo tool and work through it. Keep it updated as you go.
3. IMPLEMENT in small, verifiable steps. Prefer editing existing files over creating
   new ones. Follow the conventions already present in the code/files you touch.
4. VERIFY your work. After changes, run the relevant check (tests, a lint, re-read
   the file, run the script) and confirm it actually does what was asked. Do not
   claim success you have not observed.

# Tool discipline
- Read a file before you edit it. Edits are exact string replacements and will fail
  if the target text is not unique -- include enough surrounding context.
- Use search (grep for content, glob for filenames) to navigate. Do not read whole
  large files when a search will find the part you need.
- Prefer one well-formed command over many. Quote paths with spaces. Avoid
  interactive commands (they will hang).

# Safety
- This is the user's computer; their files matter. Be careful with destructive
  commands (rm, overwrites, force-push). When an action is destructive and the
  intent is not explicit, ask first via the ask tool.
- Never print or exfiltrate secrets you encounter.

# Finishing
- When the task is done, stop and summarize: the outcome, key files/paths, and
  anything the user should know or decide next. If you could not finish, say what
  blocked you and what you tried.
```

This is built per-run by `compute_module`/`computer_module/prompt.py` with `{cwd}`, `{date}`, `{username}` filled in, plus a one-line inventory of the available tools (including the user's MCP tools).

## 2. The agent-computer interface (tools with discipline baked into descriptions)

The tool *descriptions* are scaffolding -- they encode the rules the model must follow. Borrowed from Claude Code's tool design, re-authored. Each maps to `SandboxManager` (or MCP).

| Tool | Maps to | Description discipline (the borrowed part) |
|---|---|---|
| `run_command(command)` | `SandboxManager.exec` | "Run a shell command in the user's computer. Returns stdout/stderr/exit_code. Quote paths with spaces. Do not run interactive or long-blocking commands. Prefer one well-formed command." |
| `read_file(path, offset?, limit?)` | `SandboxManager.read_file` | "Read a file. Returns content with line numbers. Read before you edit. For large files, use offset/limit or search instead of reading the whole thing." |
| `edit_file(path, old_string, new_string, replace_all?)` | read+write via sandbox | "Exact string replacement. You MUST read the file first (this errors otherwise). `old_string` must be UNIQUE in the file -- include surrounding context if needed, or use `replace_all`. Prefer editing existing files over creating new ones." |
| `write_file(path, content)` | `SandboxManager.write_file` | "Create or overwrite a file. Prefer `edit_file` for changes to existing files; only create new files when the task requires it." |
| `list_dir(path?)` | `SandboxManager.list_dir` | "List a directory's entries (name, is_dir, size)." |
| `grep(pattern, path?, glob?)` | `run_command` -> ripgrep | "Search file *contents* by regex. Use this to find where something is defined/used instead of reading files blindly. Returns matching files and lines." |
| `glob(pattern)` | `run_command` -> find/ls | "Find *files* by name pattern (e.g. `**/*.py`). Use to locate files before reading." |
| `todo_write(items)` | in-memory + workspace file | "Maintain a short plan for multi-step work: a list of `{step, status}`. Update statuses as you go. Use proactively for anything non-trivial; it keeps you on track and shows the user the plan." |
| `ask_user(prompt)` | approval tray | "Ask the human a question and wait for their answer. Use for genuine decisions or before a destructive action whose intent is unclear -- not for routine progress." |
| *(MCP tools)* | `tool_manager.call_tool(..., user_id)` | the user's connected services (calendar, linear, etc.), user-scoped, native alongside the above. |

The two that punch above their weight (and are the most "Claude Code"): **`grep`/`glob`** (navigate by searching, not by reading everything -- this is what makes the agent effective in a large workspace without blowing context) and **`edit_file`** (read-before-edit + unique-string discipline -- what makes edits reliable instead of clobbering).

## 3. Plan-and-verify discipline

The `todo_write` tool + the "PLAN / VERIFY" steps in the prompt are the loop's backbone for non-trivial tasks: the agent decomposes, tracks progress, and -- critically -- verifies (runs the test, re-reads the file, runs the script) before declaring done. This is the single biggest quality lever and is cheap to add.

---

# The loop

`ComputerAgent.run(prompt)` (`computer_module/agent.py`):

```
build system prompt (prompt.py) + tool schemas (tools.py, incl. user's MCP tools)
messages = [system, user(prompt)]
for step in range(step_cap):
    resp = model.call(messages, tools=tool_schemas)          # native tool-calling
    if resp has tool calls:
        for call in resp.tool_calls:
            emit(kind-tagged event: shell|file|search|mcp|plan|ask)   # -> SSE/UI
            result = execute(call)                            # -> SandboxManager / MCP / ask
            append tool result to messages
        continue
    else:                                                     # model produced a final answer
        return {status: completed, summary: resp.text, outputs: [...]}
return {status: failed, reason: "step cap reached"}           # bounded; matters most on weak models
```

- **Native tool-calling** via the model's tool API (OpenAI-style `tools=`/`tool_calls`). Works against SGLang for tool-capable models (Qwen2.5 supports function-calling -- verify on our deployment, Open Question 1) and against a frontier model via an OpenAI-compatible endpoint. The model is the knob; the loop is constant.
- **Every tool call emits a `kind`-tagged event** (`shell`/`file`/`search`/`mcp`/`plan`/`ask`) -> the SSE stream + activity view (COMPUTER_SPEC Task 8/9), so the user watches it work with the "using the computer" indicator.
- **Errors never escape:** a failed tool returns its error as a tool result (the model sees it and adapts -- this is the "feed the error to the model" pattern); a model/transport error is retried by the client then surfaced as a failed run.
- **Step cap** bounds runaway loops -- load-bearing on a weak model, a safety net on a strong one.
- **Persistence:** because the tools operate the user's durable sandbox, files (and a `PLAN.md` the todo tool can mirror) persist; a later run on the same computer sees prior work.

---

# Implementation Plan

### Task A: `prompt.py` -- the system prompt builder
**Done when:** `build_system_prompt(cwd, username, date, tool_inventory) -> str` produces the layered prompt above, with the tool inventory (including the user's MCP tool names) injected. Re-authored text, our voice.
**Touch point:** `computer_module/prompt.py`.
**Acceptance test:** `test_system_prompt_includes_env_and_tools`.

### Task B: `tools.py` -- tool schemas + dispatch
**Done when:** each tool above is defined as a JSON-schema tool (name, description with the borrowed discipline, params) and a dispatcher executes a tool call against `SandboxManager` / `tool_manager.call_tool(user_id)` / the approval tray, returning a structured result. `grep`/`glob` shell out to ripgrep/find inside the sandbox. `edit_file` enforces read-before-edit and unique `old_string` (error if violated).
**Touch point:** `computer_module/tools.py`.
**Acceptance test:** `test_edit_requires_prior_read`, `test_edit_rejects_nonunique_old_string`, `test_grep_finds_content`, `test_tool_dispatch_is_user_scoped`.

### Task C: `agent.py` -- `ComputerAgent` loop
**Done when:** the loop above runs against `computer_agent.llm` with native tool-calling, a step cap, kind-tagged `emit` per tool call, error-as-tool-result, and returns `{status, summary, outputs}`. Operates exactly one user's sandbox.
**Touch point:** `computer_module/agent.py`, `model_module` (tool-calling support -- see Task E).
**Acceptance test:** `test_agent_writes_and_runs_file`, `test_step_cap_bounds_loop`, `test_tool_error_is_fed_back_not_raised`, `test_agent_uses_correct_users_sandbox`.

### Task D: `todo_write` + verify discipline
**Done when:** the todo tool maintains an in-run plan (and optionally mirrors to `PLAN.md` in the workspace for persistence); the prompt's PLAN/VERIFY steps are exercised -- the agent is observed to plan before multi-step work and to run a verification step before finishing.
**Touch point:** `computer_module/tools.py`, `prompt.py`.
**Acceptance test:** `test_todo_round_trips`; manual: a multi-step task shows a plan and a verify step.

### Task E: model client tool-calling
**Done when:** the model client used by the agent supports OpenAI-style `tools=`/`tool_calls` (extend `ArkModelLink` or a thin `computer_module` client). Verify SGLang exposes tool-calling for the served model; if not, fall back to a constrained-JSON action format behind the same interface (the loop doesn't change).
**Touch point:** `model_module/ArkModelNew.py` or `computer_module/model.py`.
**Acceptance test:** `test_client_emits_and_parses_tool_calls` (mocked).

*(Dispatch, the `computer_tasks` table, completion-to-chat, SSE, and the UI are in `COMPUTER_SPEC.md` Tasks 6-10 and are unchanged -- this spec is the worker they invoke.)*

---

# Open Questions

1. **SGLang tool-calling support (Task E).** Does our SGLang deployment expose OpenAI function-calling for the served model? If yes, native tool-calling is the path. If no, use constrained-JSON actions behind the same loop interface. *Verify on the running endpoint before Task C.*
2. **Which model.** Frontier (real capability, costs tokens, needs a key) vs local Qwen (free, weaker, may need constrained-JSON). The scaffolding is the same; this is the `computer_agent.llm` knob. *The "serious intelligence" goal points at frontier; decide per cost.*
3. **Scaffolding vs the bitter lesson.** How much structure is worth it? The borrowed scaffolding (read-before-edit, search-first, plan/verify) makes *any* model more reliable, but heavy bespoke structure can fight a strong model. Keep the scaffolding to the proven, general disciplines above; resist over-fitting. Revisit once we see a frontier model run it.
4. **Plan persistence.** Mirror the todo list to `PLAN.md` in the workspace (survives across runs, visible in the Computer tab) or keep it in-run only? *Leaning: mirror it -- cheap, and it makes the persistent computer feel coherent across sessions.*
5. **Verification depth.** How hard do we push "verify before done"? A prompt instruction is the floor; a forced verification step (the loop won't accept `completed` without at least one post-change check) is stronger but more rigid. *Start with the prompt instruction; consider enforcing later.*

---

# Implementation Notes

- **Licensing discipline.** Borrow paradigms and prompt *structure* from `arkos-inspo/claude-code`; author all text ourselves. Do not paste its prompt strings or code. The disciplines (read-before-edit, unique old_string, search-first, plan/verify) are engineering ideas, not protected expression -- express them in our words.
- **Scaffolding > model, but model is the ceiling.** The borrowed scaffolding is what makes the worker competitive at a given model tier; the model knob sets the ceiling. Get the scaffolding right once; raise the ceiling by turning the knob.
- **The two highest-leverage tools are `grep`/`glob` and `edit_file`.** If time is short, nail those four first -- they are what separate a competent computer-agent from a flailing one.
- **It operates the persistent sandbox we already built and verified.** `SandboxManager` is done; this spec is the brain that drives it.
