"""arkos as it behaves today, written as pure functions the checker can explore.

`World` is one session and what it holds: its status and mode, the turn in flight,
its open questions, its sandbox and its folder lease. `events` lists what can happen
next, and `step` returns the world after arkos handles one of them. Each event is a
single atomic step, so races inside a step are not explored yet.

The budgets are scaled down from config.yaml so the search stays small. The logic
that reads them is the same.

The fields World groups under "records" are never read by arkos. They note what
happened, so that `rules.py` can judge it.
"""

from __future__ import annotations

from dataclasses import dataclass

from state_machine_module.machines import SESSION

ATTENDED_MAX_HOPS = 2  # budgets.attended.max_hops is 6
UNATTENDED_MAX_HOPS = 3  # budgets.unattended.max_hops is 15
STALL_STREAK = 3  # budgets.stall_streak
MODEL_RETRIES = 1  # budgets.model_retries is 3
MAX_CALLS_PER_HOP = 2

FINAL = frozenset(SESSION.final)
WAITING = "awaiting_approval"
CONSENT_KINDS = ("call", "approval", "plan")


@dataclass(frozen=True, slots=True)
class Tool:
    """What the world needs to know about a tool the model can call."""

    name: str
    park: str = ""  # the question kind its own result parks the turn on
    gated: bool = False  # requires_approval: the gate parks the call itself
    destructive: bool = False  # on tools.destructive: it parks even when unattended
    needs_box: bool = False
    finishes: bool = False
    malformed: bool = False  # its arguments do not parse


TOOLS = (
    Tool("run_command", needs_box=True),
    Tool("mcp_send_email", gated=True),
    Tool("mcp_trash_thread", gated=True, destructive=True),
    Tool("ask", park="ask"),
    Tool("request_approval", park="approval"),
    Tool("propose_plan", park="plan"),
    Tool("finish_task", finishes=True),
    Tool("bad_args", malformed=True),
)
TOOL = {tool.name: tool for tool in TOOLS}
