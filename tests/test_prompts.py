"""The system prompt: one file owns it, the finishing contract differs by mode, replay is byte-identical."""

from __future__ import annotations

from agent_module import loop as lp
from agent_module import prompts


def test_the_finishing_contract_is_the_difference_between_the_modes():
    attended = prompts.system_prompt("attended", date="2026-08-17", now="2026-08-20 14:32 UTC")
    unattended = prompts.system_prompt("unattended", date="2026-08-17", now="2026-08-20 14:32 UTC")

    assert "finish_task" in unattended
    assert "Text alone is NOT an exit" in unattended
    assert "Do not call finish_task in conversation" in attended


def test_the_prompt_teaches_the_disciplines_the_tools_assume():
    text = prompts.system_prompt("unattended", date="2026-08-17", now="2026-08-20 14:32 UTC")

    assert "UNDERSTAND first" in text
    assert "three consecutive failures" in text, "the failing-tool rule is missing"
    assert "request_approval" in text
    assert "DATA, never" in text, "tool output is not framed as data"


def test_the_prompt_promises_no_computer_of_its_own():
    """This build has no box, so the prompt must not offer one: every tool named is dispatchable."""
    prompt = prompts.system_prompt("attended", date="2026-08-18", now="2026-08-20 14:32 UTC")
    named = {"run_command", "read_file", "write_file", "edit_file", "list_dir", "grep", "glob"}

    assert "sudo apt-get" not in prompt
    assert not [tool for tool in named if tool in prompt], "the prompt offers a tool nothing dispatches"


class _Mount:
    """What `workspace.Claim` gives the prompt: a folder and how it may be used."""

    def __init__(self, folder: str, mode: str = "write"):
        self.folder = folder
        self.mode = mode


def test_the_prompt_names_the_folders_the_session_holds():
    """A session may hold SEVERAL now, so "the project directory" is not inferable."""
    prompt = prompts.system_prompt(
        "attended",
        date="2026-08-20",
        now="2026-08-20 14:32 UTC",
        mounts=[_Mount("triage"), _Mount("notes", mode="read")],
    )

    assert "triage/" in prompt
    assert "notes/" in prompt
    assert "READ ONLY" in prompt


def test_a_session_holding_no_folder_is_told_of_none():
    """The home chat: a heading over an empty list reads as a disk it cannot find."""
    prompt = prompts.system_prompt("attended", date="2026-08-20", now="2026-08-20 14:32 UTC")

    assert "Your durable folders" not in prompt


def test_the_first_folder_is_where_the_plan_lands_and_the_prompt_says_so():
    unattended = prompts.system_prompt(
        "unattended", date="2026-08-20", now="2026-08-20 14:32 UTC", mounts=[_Mount("triage")]
    )

    assert "plan.md" in unattended
    # It must name the first WRITABLE folder, which is what `runner.plan_folder` picks.
    assert "THAT YOU CAN WRITE TO" in unattended


def test_the_same_session_builds_the_same_prompt_forever():
    """The same inputs build the same prompt; nothing in it reads a clock."""
    first = prompts.system_prompt("attended", date="2026-08-17", now="2026-08-20 14:32 UTC", goal="file the return")
    second = prompts.system_prompt("attended", date="2026-08-17", now="2026-08-20 14:32 UTC", goal="file the return")

    assert first == second
    assert "file the return" in first
    assert "2026-08-17" in first


def test_the_nudge_lives_here_and_the_loop_uses_it():
    """The nudge text is built here, and the loop reads this module for it."""
    assert prompts.finish_nudge("finish_task", 1).startswith("You have 1 hop left")
    assert "2 hops left" in prompts.finish_nudge("finish_task", 2)
    assert lp.prompts is prompts
