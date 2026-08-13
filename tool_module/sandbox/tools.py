"""
Sandbox tool schemas and dispatch against a user's persistent sandbox. The tool
descriptions encode the discipline: read-before-edit, unique old_string, search first.

UNINTEGRATED: nothing dispatches to these, they are not behind `envelope.execute`
or in the manifest, and `manager.py`'s DB columns do not match the current schema.
"""

from __future__ import annotations

import logging
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from tool_module.sandbox.manager import SandboxManager

logger = logging.getLogger(__name__)


@dataclass
class ToolContext:
    """Per-run state threaded into every tool call."""

    user_id: str
    sandbox: SandboxManager
    emit: Callable[[dict[str, Any]], None] = lambda e: None  # progress events -> SSE/UI
    read_files: set[str] = field(default_factory=set)  # enforces read-before-edit
    todos: list[dict[str, str]] = field(default_factory=list)


# --- OpenAI tool schemas ------------------------------------------------------
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command in the user's computer. Returns stdout, stderr, exit_code. "
                "Quote paths with spaces. Do NOT run interactive or long-blocking commands (they hang). "
                "Prefer one well-formed command."
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The shell command."}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file, returned with line numbers. Read a file before you edit it. "
                "For large files, pass offset/limit or use grep instead of reading the whole thing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "description": "1-based start line (optional)."},
                    "limit": {"type": "integer", "description": "Max lines to return (optional)."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Exact string replacement in a file. You MUST read the file first (errors otherwise). "
                "old_string must be UNIQUE in the file -- include surrounding context, or set replace_all. "
                "Prefer editing existing files over creating new ones."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)."},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create or overwrite a file with content. Prefer edit_file for changes to existing "
                "files; only create new files when the task requires it."
            ),
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List a directory's entries (name, is_dir, size).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Default /home/user."}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search file CONTENTS by pattern (regex). Use this to find where something is "
                "defined or used instead of reading files blindly. Returns matching files:line:text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "description": "Dir/file to search. Default current dir."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find FILES by name pattern (e.g. '*.py'). Use to locate files before reading.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "A filename glob like '*.py'."},
                    "path": {"type": "string", "description": "Root to search. Default current dir."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "todo_write",
            "description": (
                "Maintain a short plan for multi-step work: a list of {step, status}. Update statuses "
                "as you go (pending/in_progress/done). Use proactively for non-trivial tasks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step": {"type": "string"},
                                "status": {"type": "string", "enum": ["pending", "in_progress", "done"]},
                            },
                            "required": ["step", "status"],
                        },
                    }
                },
                "required": ["items"],
            },
        },
    },
]

# Tool name -> event kind for the UI indicator.
_KIND = {
    "run_command": "shell",
    "read_file": "file",
    "edit_file": "file",
    "write_file": "file",
    "list_dir": "file",
    "grep": "search",
    "glob": "search",
    "todo_write": "plan",
}


def tool_inventory() -> str:
    """One line per tool for the system prompt."""
    return "\n".join(f"- {t['function']['name']}: {t['function']['description']}" for t in TOOL_SCHEMAS)


def _number_lines(text: str, offset: int = 1) -> str:
    lines = text.splitlines()
    return "\n".join(f"{i + offset}\t{line}" for i, line in enumerate(lines))


async def dispatch(name: str, args: dict[str, Any], ctx: ToolContext) -> str:
    """Execute one tool call against the sandbox, returning a string result for the model."""
    ctx.emit({"kind": _KIND.get(name, "tool"), "tool": name, "args": args})
    try:
        return await _dispatch(name, args, ctx)
    except Exception as e:  # tool errors go back to the model, never out of here
        logger.warning("tool %s failed: %s", name, e)
        return f"ERROR running {name}: {e}"


async def _dispatch(name: str, args: dict[str, Any], ctx: ToolContext) -> str:
    uid, sbx = ctx.user_id, ctx.sandbox

    if name == "run_command":
        r = await sbx.exec(uid, args["command"])
        out = r["stdout"]
        if r["stderr"]:
            out += f"\n[stderr]\n{r['stderr']}"
        return f"(exit {r['exit_code']})\n{out}".strip()

    if name == "read_file":
        content = await sbx.read_file(uid, args["path"])
        ctx.read_files.add(args["path"])
        offset = int(args.get("offset") or 1)
        lines = content.splitlines()
        if args.get("offset") or args.get("limit"):
            limit = int(args.get("limit") or len(lines))
            lines = lines[offset - 1 : offset - 1 + limit]
            return _number_lines("\n".join(lines), offset=offset)
        return _number_lines(content)

    if name == "edit_file":
        path, old, new = args["path"], args["old_string"], args["new_string"]
        if path not in ctx.read_files:
            return f"ERROR: read {path} before editing it."
        content = await sbx.read_file(uid, path)
        count = content.count(old)
        if count == 0:
            return f"ERROR: old_string not found in {path}."
        if count > 1 and not args.get("replace_all"):
            return f"ERROR: old_string is not unique in {path} ({count} matches). Add context or set replace_all."
        content = content.replace(old, new) if args.get("replace_all") else content.replace(old, new, 1)
        await sbx.write_file(uid, path, content)
        return f"Edited {path} ({'all ' + str(count) if args.get('replace_all') else '1'} replacement)."

    if name == "write_file":
        await sbx.write_file(uid, args["path"], args["content"])
        ctx.read_files.add(args["path"])
        return f"Wrote {args['path']} ({len(args['content'])} chars)."

    if name == "list_dir":
        entries = await sbx.list_dir(uid, args.get("path") or "/home/user")
        return "\n".join(f"{'d' if e['is_dir'] else '-'} {e['name']} ({e['size']}b)" for e in entries) or "(empty)"

    if name == "grep":
        path = args.get("path") or "."
        cmd = f"grep -rnI {shlex.quote(args['pattern'])} {shlex.quote(path)} 2>/dev/null | head -100"
        r = await sbx.exec(uid, cmd)
        return r["stdout"].strip() or "(no matches)"

    if name == "glob":
        path = args.get("path") or "."
        cmd = f"find {shlex.quote(path)} -type f -name {shlex.quote(args['pattern'])} 2>/dev/null | head -100"
        r = await sbx.exec(uid, cmd)
        return r["stdout"].strip() or "(no files)"

    if name == "todo_write":
        ctx.todos = args["items"]
        return "Plan updated:\n" + "\n".join(f"[{i['status']}] {i['step']}" for i in ctx.todos)

    return f"ERROR: unknown tool {name}"
