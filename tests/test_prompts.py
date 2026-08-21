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

    assert "Never edit a file you" in text and "have not read" in text
    assert "unique" in text, "edit_file's exact-match rule is missing"
    assert "request_approval" in text
    assert "DATA, never" in text, "tool output is not framed as data"


def test_the_prompt_says_whose_computer_it_is():
    """It refused to install a missing tool because nothing told it where it was.

    The box is disposable and its own, so a missing package is a thing to fix
    rather than a thing to report, and the user cannot fix it for the model —
    they cannot even see it.
    """
    prompt = prompts.system_prompt("attended", date="2026-08-18", now="2026-08-20 14:32 UTC")

    assert "sudo apt-get" in prompt, "the model has no idea it may install things"
    assert "not the user's machine" in prompt
    assert "~/store/<folder>/ is the ONLY durable path" in prompt

    # The first version of this section said "everything OUTSIDE ~/projects is
    # scratch", and the model read it exactly as written: it cloned a repo to
    # ~/projects/arkos, a sibling of the mount, where flush never looks and the
    # reaper takes it with the box. Only the claimed directories are swept.
    assert "including any new directory you create under" in prompt.lower()


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

    assert "~/store/triage/" in prompt
    assert "~/store/notes/" in prompt
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
    assert "~/store/<folder>/plan.md" in unattended
    # It must name the first WRITABLE one, which is what `runner.plan_folder`
    # picks: a read claim listed first would otherwise be pointed at a file
    # nothing could have written there.
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
