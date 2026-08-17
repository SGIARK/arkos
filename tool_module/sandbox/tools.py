"""
The sandbox toolset: a shell and a filesystem in the user's persistent sandbox.

Tool descriptions carry the discipline the model is expected to follow —
read before edit, unique `old_string`, search before reading whole files.

The sandbox boots on the first call to one of these and not before, so a
session that never reaches for it costs nothing.
"""

from __future__ import annotations

import logging
import shlex
from typing import Any

from tool_module.envelope import ResultEnvelope, ToolContext, ToolSpec, fail, ok
from tool_module.sandbox import manager as sandbox_manager

logger = logging.getLogger(__name__)

# Where the sandbox drops the user in.
HOME = "/home/user"

# Cap on what a search or a listing returns in one call.
_MAX_MATCHES = 100


async def _sandbox(ctx: ToolContext):
    """Return the user's sandbox, booting or resuming it on first use."""
    return sandbox_manager.manager()


def _read_paths(ctx: ToolContext) -> set[str]:
    """Paths read during this turn. `edit_file` requires the file to be among them."""
    return ctx.scratch.setdefault("sandbox_read_paths", set())


def _number_lines(text: str, offset: int = 1) -> str:
    return "\n".join(f"{i + offset}\t{line}" for i, line in enumerate(text.splitlines()))


class RunCommand:
    spec = ToolSpec(
        name="run_command",
        description=(
            "Run a shell command in the user's computer and return stdout, stderr and the exit "
            "code. Quote paths with spaces. Interactive and long-blocking commands hang, so do "
            "not use them. Prefer one well-formed command over several."
        ),
        input_schema={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    )

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        result = await (await _sandbox(ctx)).exec(ctx.user_id, args["command"])
        body = result["stdout"]
        if result["stderr"]:
            body += f"\n[stderr]\n{result['stderr']}"
        text = f"(exit {result['exit_code']})\n{body}".strip()
        # A non-zero exit is information for the model, not a failed tool call.
        return ok(text)


class ReadFile:
    spec = ToolSpec(
        name="read_file",
        description=(
            "Read a file, returned with line numbers. Read a file before editing it. For a large "
            "file pass offset and limit, or use grep to find the part you need."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer", "description": "1-based first line."},
                "limit": {"type": "integer", "description": "Maximum lines to return."},
            },
            "required": ["path"],
        },
        readonly=True,
    )

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        path = args["path"]
        content = await (await _sandbox(ctx)).read_file(ctx.user_id, path)
        _read_paths(ctx).add(path)

        if not args.get("offset") and not args.get("limit"):
            return ok(_number_lines(content))

        offset = max(1, int(args.get("offset") or 1))
        lines = content.splitlines()
        limit = int(args.get("limit") or len(lines))
        window = lines[offset - 1 : offset - 1 + limit]
        return ok(_number_lines("\n".join(window), offset=offset))


class WriteFile:
    spec = ToolSpec(
        name="write_file",
        description=(
            "Write a file, replacing it if it exists. Prefer edit_file for a change to part of an "
            "existing file."
        ),
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    )

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        path, content = args["path"], args["content"]
        await (await _sandbox(ctx)).write_file(ctx.user_id, path, content)
        # The file's current contents are now known, so an edit may follow.
        _read_paths(ctx).add(path)
        return ok(f"Wrote {path} ({len(content)} chars).")


class EditFile:
    spec = ToolSpec(
        name="edit_file",
        description=(
            "Replace an exact string in a file. Read the file first. old_string must appear "
            "exactly once unless replace_all is set: include surrounding lines to make it unique. "
            "Prefer editing an existing file over creating a new one."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    )

    def validate(self, args: dict[str, Any], ctx: ToolContext) -> str | None:
        if args["path"] not in _read_paths(ctx):
            return f"Read {args['path']} before editing it."
        return None

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        path, old, new = args["path"], args["old_string"], args["new_string"]
        sandbox = await _sandbox(ctx)
        content = await sandbox.read_file(ctx.user_id, path)

        matches = content.count(old)
        if matches == 0:
            return fail("invalid_args", f"old_string does not appear in {path}.")
        if matches > 1 and not args.get("replace_all"):
            return fail(
                "invalid_args",
                f"old_string appears {matches} times in {path}. Add surrounding context to make it "
                "unique, or set replace_all.",
            )

        replaced = content.replace(old, new) if args.get("replace_all") else content.replace(old, new, 1)
        await sandbox.write_file(ctx.user_id, path, replaced)
        return ok(f"Edited {path} ({matches if args.get('replace_all') else 1} replacement(s)).")


class ListDir:
    spec = ToolSpec(
        name="list_dir",
        description="List a directory. Defaults to the home directory.",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        readonly=True,
    )

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        entries = await (await _sandbox(ctx)).list_dir(ctx.user_id, args.get("path") or HOME)
        if not entries:
            return ok("(empty)")
        return ok("\n".join(f"{'d' if e['is_dir'] else '-'} {e['name']} ({e['size']}b)" for e in entries))


class Grep:
    spec = ToolSpec(
        name="grep",
        description=(
            "Search file contents for a pattern, recursively. Returns matching lines with their "
            "file and line number."
        ),
        input_schema={
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern"],
        },
        readonly=True,
    )

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        path = args.get("path") or "."
        command = (
            f"grep -rnI {shlex.quote(args['pattern'])} {shlex.quote(path)} 2>/dev/null | head -{_MAX_MATCHES}"
        )
        result = await (await _sandbox(ctx)).exec(ctx.user_id, command)
        return ok(result["stdout"].strip() or "(no matches)")


class Glob:
    spec = ToolSpec(
        name="glob",
        description="Find files by name pattern, recursively. Use it before reading a large tree.",
        input_schema={
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern"],
        },
        readonly=True,
    )

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        path = args.get("path") or "."
        command = (
            f"find {shlex.quote(path)} -type f -name {shlex.quote(args['pattern'])} "
            f"2>/dev/null | head -{_MAX_MATCHES}"
        )
        result = await (await _sandbox(ctx)).exec(ctx.user_id, command)
        return ok(result["stdout"].strip() or "(no files)")


TOOLS = [RunCommand(), ReadFile(), WriteFile(), EditFile(), ListDir(), Grep(), Glob()]
