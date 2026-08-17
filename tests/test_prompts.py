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
    assert "unique" in text, "edit_file's exact-match rule has to be taught somewhere"
    assert "request_approval" in text
    assert "DATA, never" in text, "tool output is untrusted input, not instructions"


def test_the_same_session_builds_the_same_prompt_forever():
    """Nothing here reads a clock, so a replay next year rebuilds what actually ran."""
    first = prompts.system_prompt("attended", date="2026-08-17", goal="file the return")
    second = prompts.system_prompt("attended", date="2026-08-17", goal="file the return")

    assert first == second
    assert "file the return" in first
    assert "2026-08-17" in first


def test_the_nudge_lives_here_and_the_loop_uses_it():
    """It is model-facing English under a contract that bans magic values."""
    assert prompts.finish_nudge("finish_task", 1).startswith("You have 1 hop left")
    assert "2 hops left" in prompts.finish_nudge("finish_task", 2)
    assert lp.prompts is prompts
