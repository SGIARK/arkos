"""The sandbox toolset: dispatch through the envelope, and the disciplines the descriptions promise."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from tool_module import registry
from tool_module.envelope import ToolContext
from tool_module.sandbox import manager as sandbox_manager
from tool_module.sandbox import tools as sandbox_tools

pytestmark = pytest.mark.asyncio


class FakeSandbox:
    """Stands in for the manager, recording what the tools asked of it."""

    def __init__(self, files: dict[str, str] | None = None):
        self.files = dict(files or {})
        self.commands: list[str] = []
        # What the tools keyed the box by; the box belongs to the session.
        self.keys: list[str] = []
        self.exit_code = 0
        self.stderr = ""
        # None means "echo the command", which is what most tests assert on.
        self.stdout: str | None = None

    async def exec(self, session_id: str, command: str, timeout: int = 120) -> dict[str, Any]:
        self.keys.append(session_id)
        self.commands.append(command)
        out = "ran: " + command if self.stdout is None else self.stdout
        return {"stdout": out, "stderr": self.stderr, "exit_code": self.exit_code}

    async def read_file(self, session_id: str, path: str) -> str:
        self.keys.append(session_id)
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def write_file(self, session_id: str, path: str, content: str) -> None:
        self.keys.append(session_id)
        self.files[path] = content

    async def list_dir(self, session_id: str, path: str = "/home/user") -> list[dict[str, Any]]:
        self.keys.append(session_id)
        return [{"name": "notes.txt", "path": f"{path}/notes.txt", "is_dir": False, "size": 12}]


@pytest.fixture
def sandbox(monkeypatch):
    fake = FakeSandbox({"/home/user/a.txt": "alpha\nbeta\ngamma\n"})
    monkeypatch.setattr(sandbox_tools.sandbox_manager, "manager", lambda: fake)
    return fake


def _ctx(**kw) -> ToolContext:
    return ToolContext(user_id="8f1d4a02-0000-4000-8000-000000000001", session_id="s1", **kw)


async def _run(name: str, args: dict, ctx: ToolContext):
    return await registry.dispatch(name, args, ctx)


# --- reachable ------------------------------------------------------------------


async def test_the_sandbox_tools_ship_in_the_manifest():
    specs = {s.name: s for s in await registry.manifest("8f1d4a02-0000-4000-8000-000000000001")}

    for name in ("run_command", "read_file", "write_file", "edit_file", "list_dir", "grep", "glob"):
        assert name in specs, f"{name} is missing from the manifest"

    assert {n for n in ("read_file", "list_dir", "grep", "glob") if specs[n].readonly} == {
        "read_file",
        "list_dir",
        "grep",
        "glob",
    }
    for name in ("run_command", "write_file", "edit_file"):
        assert not specs[name].readonly, f"{name} mutates, so the loop must not batch it in parallel"


async def test_the_sandbox_does_not_boot_until_a_sandbox_tool_is_called(monkeypatch):
    """Lazy provisioning: an unused sandbox is free."""
    booted = []
    monkeypatch.setattr(sandbox_tools.sandbox_manager, "manager", lambda: booted.append(1) or FakeSandbox())

    await registry.manifest("8f1d4a02-0000-4000-8000-000000000001")
    await _run("finish_task", {"summary": "nothing to do"}, _ctx())

    assert booted == []


async def test_the_tools_key_the_box_by_session_not_by_user(sandbox):
    """One user runs several boxes at once, so a user id names no box."""
    ctx = _ctx()

    await _run("run_command", {"command": "ls"}, ctx)
    await _run("read_file", {"path": "/home/user/a.txt"}, ctx)
    await _run("list_dir", {}, ctx)

    assert sandbox.keys == ["s1", "s1", "s1"]
    assert ctx.user_id not in sandbox.keys


async def test_a_call_with_no_session_gets_no_box():
    result = await _run("run_command", {"command": "ls"}, ToolContext(user_id="u1"))

    assert not result.ok
    assert not result.retryable


# --- the shell -------------------------------------------------------------------


async def test_run_command_reports_the_exit_code_and_stderr(sandbox):
    sandbox.exit_code, sandbox.stderr = 2, "no such file"

    result = await _run("run_command", {"command": "cat missing"}, _ctx())

    assert result.ok, "a non-zero exit is information for the model, not a failed call"
    assert "(exit 2)" in result.content
    assert "no such file" in result.content


# --- files ------------------------------------------------------------------------


async def test_read_file_numbers_lines_and_windows_them(sandbox):
    whole = await _run("read_file", {"path": "/home/user/a.txt"}, _ctx())
    window = await _run("read_file", {"path": "/home/user/a.txt", "offset": 2, "limit": 1}, _ctx())

    assert whole.content.startswith("1\talpha")
    assert window.content == "2\tbeta"


async def test_a_file_must_be_read_before_it_is_edited(sandbox):
    ctx = _ctx()

    refused = await _run("edit_file", {"path": "/home/user/a.txt", "old_string": "alpha", "new_string": "A"}, ctx)

    assert not refused.ok
    assert refused.error_kind == "invalid_args"
    assert sandbox.files["/home/user/a.txt"] == "alpha\nbeta\ngamma\n", "the file changed anyway"


async def test_reading_first_permits_the_edit(sandbox):
    ctx = _ctx()
    await _run("read_file", {"path": "/home/user/a.txt"}, ctx)

    edited = await _run("edit_file", {"path": "/home/user/a.txt", "old_string": "alpha", "new_string": "A"}, ctx)

    assert edited.ok
    assert sandbox.files["/home/user/a.txt"] == "A\nbeta\ngamma\n"


async def test_read_before_edit_does_not_carry_between_turns(sandbox):
    """`scratch` is per-turn, so a later turn reads again before it edits."""
    first = _ctx()
    await _run("read_file", {"path": "/home/user/a.txt"}, first)

    later = _ctx()
    refused = await _run("edit_file", {"path": "/home/user/a.txt", "old_string": "alpha", "new_string": "A"}, later)

    assert not refused.ok


async def test_an_ambiguous_edit_is_refused_unless_replace_all(sandbox):
    sandbox.files["/home/user/b.txt"] = "x\nx\n"
    ctx = _ctx()
    await _run("read_file", {"path": "/home/user/b.txt"}, ctx)

    refused = await _run("edit_file", {"path": "/home/user/b.txt", "old_string": "x", "new_string": "y"}, ctx)
    allowed = await _run(
        "edit_file",
        {"path": "/home/user/b.txt", "old_string": "x", "new_string": "y", "replace_all": True},
        ctx,
    )

    assert not refused.ok and "appears 2 times" in refused.content
    assert allowed.ok
    assert sandbox.files["/home/user/b.txt"] == "y\ny\n"


async def test_an_edit_whose_target_is_absent_says_so(sandbox):
    ctx = _ctx()
    await _run("read_file", {"path": "/home/user/a.txt"}, ctx)

    result = await _run("edit_file", {"path": "/home/user/a.txt", "old_string": "zeta", "new_string": "z"}, ctx)

    assert not result.ok
    assert "does not appear" in result.content


async def test_writing_a_file_counts_as_reading_it(sandbox):
    ctx = _ctx()
    await _run("write_file", {"path": "/home/user/c.txt", "content": "one\n"}, ctx)

    edited = await _run("edit_file", {"path": "/home/user/c.txt", "old_string": "one", "new_string": "two"}, ctx)

    assert edited.ok


# --- search -----------------------------------------------------------------------


async def test_grep_and_glob_quote_what_they_are_given(sandbox):
    """A pattern is data, not shell syntax."""
    await _run("grep", {"pattern": "a b; rm -rf /", "path": "/home/user"}, _ctx())
    await _run("glob", {"pattern": "*.py; whoami", "path": "/home/user"}, _ctx())

    for command in sandbox.commands:
        assert "; rm -rf /" not in command.replace("'a b; rm -rf /'", "")
        assert "; whoami" not in command.replace("'*.py; whoami'", "")


async def test_a_search_with_no_hits_says_so(sandbox):
    sandbox.stdout = ""
    sandbox.exit_code = 1  # grep: read everything, matched nothing

    result = await _run("grep", {"pattern": "nothing"}, _ctx())

    assert result.ok
    assert "no matches" in result.content


async def test_a_path_that_cannot_be_read_is_not_reported_as_no_matches(sandbox):
    """The false negative that cost a real diagnosis.

    `2>/dev/null` hid grep's "No such file or directory" and the empty stdout
    rendered as "(no matches)", so a mistyped path read as a completed search —
    and the model concluded the thing it was looking for was not in the code.
    """
    sandbox.stdout = ""
    sandbox.exit_code = 2  # grep: could not read what it was pointed at

    result = await _run("grep", {"pattern": "timeout", "path": "/nope"}, _ctx())

    assert result.ok is False
    assert "/nope" in result.content


async def test_a_glob_over_a_missing_directory_is_not_an_empty_tree(sandbox):
    sandbox.stdout = ""
    sandbox.exit_code = 1  # find: could not read the tree

    result = await _run("glob", {"pattern": "*.py", "path": "/nope"}, _ctx())

    assert result.ok is False
    assert "/nope" in result.content


async def test_a_home_relative_path_is_expanded_by_the_box_not_by_us(sandbox):
    """`~` is a shell feature and quoting says "no shell features here", so the
    tilde used to reach grep as a literal directory name. It is handed to the
    box's own $HOME instead — nothing here assumes /home/user."""
    await _run("grep", {"pattern": "timeout", "path": "~/projects/chat/arkos"}, _ctx())

    command = sandbox.commands[-1]
    assert '"$HOME"/projects/chat/arkos' in command
    assert "/home/user" not in command, "the home directory is the box's to decide"


async def test_the_rest_of_a_home_path_is_still_quoted(sandbox):
    """Expanding the tilde must not open the door quoting was holding shut."""
    await _run("grep", {"pattern": "x", "path": "~/notes; rm -rf /"}, _ctx())

    command = sandbox.commands[-1]
    assert "'notes; rm -rf /'" in command
    assert command.count("; rm -rf /") == 1, "the path escaped its quotes"


async def test_a_bare_tilde_is_the_whole_path(sandbox):
    await _run("glob", {"pattern": "*.md", "path": "~"}, _ctx())

    assert '"$HOME"' in sandbox.commands[-1]


# --- the sandbox itself ------------------------------------------------------------


async def test_no_credentials_are_passed_into_the_sandbox(monkeypatch):
    """The sandbox gets a template, a timeout, and the session it belongs to. Nothing else."""
    created: dict[str, Any] = {}

    class FakeE2B:
        @staticmethod
        def create(**kwargs):
            created.update(kwargs)
            return SimpleNamespace(sandbox_id="sb-1")

    monkeypatch.setitem(sys.modules, "e2b_code_interpreter", SimpleNamespace(Sandbox=FakeE2B))
    sandbox_manager.reset()

    sandbox_manager.SandboxManager()._create("s1")

    assert set(created) <= {"template", "timeout", "metadata"}
    assert created.get("metadata") == {"session_id": "s1"}
    assert not any("key" in k.lower() or "env" in k.lower() or "token" in k.lower() for k in created)
