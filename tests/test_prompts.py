"""The system prompt: one file owns it, the finishing contract differs by mode, replay is byte-identical."""

from __future__ import annotations

from agent_module import loop as lp
from agent_module import prompts


def test_the_finishing_contract_is_the_difference_between_the_modes():
    attended = prompts.system_prompt("attended", date="2026-08-17")
    unattended = prompts.system_prompt("unattended", date="2026-08-17")

    assert "finish_task" in unattended
    assert "Text alone is NOT an exit" in unattended
    assert "Do not call finish_task in conversation" in attended


def test_the_prompt_teaches_the_disciplines_the_tools_assume():
    text = prompts.system_prompt("unattended", date="2026-08-17")

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
    prompt = prompts.system_prompt("attended", date="2026-08-18")

    assert "sudo apt-get" in prompt, "the model has no idea it may install things"
    assert "not the user's machine" in prompt
    assert "~/projects/<project>/ is the ONLY durable path" in prompt

    # The first version of this section said "everything OUTSIDE ~/projects is
    # scratch", and the model read it exactly as written: it cloned a repo to
    # ~/projects/arkos, a sibling of the mount, where flush never looks and the
    # reaper takes it with the box. Only the claimed directories are swept.
    assert "including any new directory you create under" in prompt.lower()


def test_the_same_session_builds_the_same_prompt_forever():
    """The same inputs build the same prompt; nothing in it reads a clock."""
    first = prompts.system_prompt("attended", date="2026-08-17", goal="file the return")
    second = prompts.system_prompt("attended", date="2026-08-17", goal="file the return")

    assert first == second
    assert "file the return" in first
    assert "2026-08-17" in first


def test_the_nudge_lives_here_and_the_loop_uses_it():
    """The nudge text is built here, and the loop reads this module for it."""
    assert prompts.finish_nudge("finish_task", 1).startswith("You have 1 hop left")
    assert "2 hops left" in prompts.finish_nudge("finish_task", 2)
    assert lp.prompts is prompts
