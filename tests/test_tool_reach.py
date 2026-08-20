"""What a session reaches, and the prompt that tells the model about it (11.5).

Two rules meet here and they are not the same rule. The TOGGLES say what a human
asked for; the MANIFEST says what the request actually carries. The provider
refuses a request over `llm.max_tools` outright, so the manifest gets the last
word — and the prompt is generated from the manifest, never from the toggles,
or the emergency exit reintroduces the bug it exists to prevent.
"""

from __future__ import annotations

import pytest

from agent_module import prompts
from tool_module import registry as reg
from tool_module.envelope import ToolSpec

# Applied per-test rather than module-wide: the prompt half is plain synchronous
# text and an asyncio mark on it is a warning, not a fact.
asyncio_test = pytest.mark.asyncio

SESSION = "session-1"
USER = "u1"


class _Server:
    def __init__(self, label: str, tools: int):
        self.label = label
        self.name = label.title()
        self.mcp_url = f"https://{label}.example"
        self.specs = [ToolSpec(name=f"{label}_{i}", description="d") for i in range(tools)]


class _Mcp:
    def __init__(self, *servers: _Server):
        self.servers = list(servers)

    async def reach(self, user_id: str) -> list[_Server]:
        return list(self.servers)


@pytest.fixture
def toggles(monkeypatch):
    """Set which servers the session was given, longest-enabled first."""

    def use(*urls: str):
        async def enabled_urls(session_id):
            assert session_id == SESSION
            return list(urls)

        monkeypatch.setattr(reg.session_tools, "enabled_urls", enabled_urls)

    use()
    return use


@pytest.fixture
def cap(monkeypatch):
    """Set `llm.max_tools` to ours plus a stated allowance for the human."""

    def use(allowance: int):
        real = reg._cfg
        ours = len(reg.local_tools())
        monkeypatch.setattr(
            reg,
            "_cfg",
            lambda key, default: ours + allowance if key == "llm.max_tools" else real(key, default),
        )

    return use


def _mcp_names(manifest: reg.Manifest) -> list[str]:
    return [s.name for s in manifest.specs if s.name.startswith(reg.MCP_PREFIX)]


# --- what a session reaches --------------------------------------------------------


@asyncio_test
async def test_a_session_with_nothing_enabled_gets_exactly_our_tools(toggles):
    """The default is ours alone. A connected server is not a reachable one."""
    connected = _Mcp(_Server("gmail", 12), _Server("slack", 38))

    shipped = await reg.manifest(USER, mcp=connected, session_id=SESSION)

    assert _mcp_names(shipped) == []
    assert {s.name for s in shipped.specs} == set(reg.local_tools())
    assert shipped.used == 0
    assert [s.enabled for s in shipped.servers] == [False, False]


@asyncio_test
async def test_enabling_a_server_adds_only_its_tools(toggles):
    connected = _Mcp(_Server("gmail", 3), _Server("slack", 4))
    toggles("https://gmail.example")

    shipped = await reg.manifest(USER, mcp=connected, session_id=SESSION)

    assert _mcp_names(shipped) == ["mcp_gmail_0", "mcp_gmail_1", "mcp_gmail_2"]
    assert shipped.used == 3
    assert [(s.label, s.shipped) for s in shipped.servers] == [("gmail", True), ("slack", False)]


@asyncio_test
async def test_a_turn_with_no_session_reaches_nothing_remote(toggles):
    """No session means no toggles, and no toggles means ours."""
    shipped = await reg.manifest(USER, mcp=_Mcp(_Server("gmail", 3)))

    assert _mcp_names(shipped) == []


# --- the cap, regardless of what the toggles say -----------------------------------


@asyncio_test
async def test_the_manifest_never_exceeds_the_budget_the_toggles_ask_for(toggles, cap):
    """The 164-schema request, refused where it is built rather than by the provider."""
    cap(20)
    connected = _Mcp(_Server("gmail", 12), _Server("slack", 38))
    toggles("https://gmail.example", "https://slack.example")

    shipped = await reg.manifest(USER, mcp=connected, session_id=SESSION)

    assert shipped.used == 12
    assert shipped.used <= shipped.budget
    assert len(shipped.specs) <= shipped.ours + shipped.budget
    assert [s.label for s in shipped.benched] == ["slack"]


@asyncio_test
async def test_a_server_that_grew_overnight_is_benched_whole(toggles, cap):
    """Never a subset: half a server is a model that thinks it can post and cannot."""
    cap(20)
    toggles("https://gmail.example")

    small = await reg.manifest(USER, mcp=_Mcp(_Server("gmail", 12)), session_id=SESSION)
    grown = await reg.manifest(USER, mcp=_Mcp(_Server("gmail", 40)), session_id=SESSION)

    assert small.used == 12
    assert _mcp_names(grown) == [], "no partial server"
    assert [s.label for s in grown.benched] == ["gmail"]


@asyncio_test
async def test_the_most_recently_enabled_server_is_the_one_dropped(toggles, cap):
    """The session keeps the reach it has been working with."""
    cap(20)
    connected = _Mcp(_Server("gmail", 12), _Server("linear", 11))
    toggles("https://gmail.example", "https://linear.example")

    shipped = await reg.manifest(USER, mcp=connected, session_id=SESSION)

    assert [s.label for s in shipped.servers if s.shipped] == ["gmail"]
    assert [s.label for s in shipped.benched] == ["linear"]


@asyncio_test
async def test_a_later_smaller_server_does_not_jump_the_queue(toggles, cap):
    """Stopping, not skipping: keeping a newer server while an older one is cut
    would make the drop rule a lie the next time somebody read it."""
    cap(20)
    connected = _Mcp(_Server("gmail", 5), _Server("slack", 30), _Server("tiny", 1))
    toggles("https://gmail.example", "https://slack.example", "https://tiny.example")

    shipped = await reg.manifest(USER, mcp=connected, session_id=SESSION)

    assert [s.label for s in shipped.servers if s.shipped] == ["gmail"]
    assert sorted(s.label for s in shipped.benched) == ["slack", "tiny"]


@asyncio_test
async def test_a_budget_spent_to_the_last_slot_is_legal(toggles, cap):
    cap(12)
    toggles("https://gmail.example")

    shipped = await reg.manifest(USER, mcp=_Mcp(_Server("gmail", 12)), session_id=SESSION)

    assert shipped.used == 12 == shipped.budget


@asyncio_test
async def test_ours_are_never_counted_against_the_humans_allowance(toggles, cap):
    cap(10)
    shipped = await reg.manifest(USER, mcp=_Mcp(), session_id=SESSION)

    assert shipped.budget == 10
    assert shipped.ours == len(reg.local_tools())
    assert len(shipped.specs) == shipped.ours


# --- the prompt, generated from the manifest that shipped --------------------------


def _reach(**kw) -> reg.ServerReach:
    base = {"label": "x", "name": "X", "mcp_url": "https://x.example", "tools": 1, "enabled": True, "shipped": True}
    return reg.ServerReach(**{**base, **kw})


def test_the_prompt_names_what_is_enabled_and_what_is_connected_but_off():
    text = prompts.connected_services(
        [
            _reach(label="gmail", name="Gmail", tools=12),
            _reach(label="slack", name="Slack", tools=38, enabled=False, shipped=False),
        ]
    )

    assert "Gmail (12 tools)" in text
    assert "Connected to this user's account but NOT enabled here: Slack." in text
    assert "not enabled in this session" in text


def test_the_prompt_says_so_when_nothing_is_enabled():
    text = prompts.connected_services([_reach(name="Slack", enabled=False, shipped=False)])

    assert "No service is enabled in this session" in text
    assert "Slack" in text


def test_nothing_connected_buys_no_section_at_all():
    assert prompts.connected_services([]) == ""


def test_a_benched_server_is_named_unavailable_not_available():
    """The backstop's own failure mode: a prompt built from toggles would promise
    a server the cap quietly dropped."""
    text = prompts.connected_services(
        [
            _reach(label="gmail", name="Gmail", tools=12),
            _reach(label="slack", name="Slack", tools=38, enabled=True, shipped=False),
        ]
    )

    assert "Gmail (12 tools)" in text
    assert "NOT loaded this turn" in text and "Slack" in text
    assert "Slack (38 tools)" not in text, "never listed as available"


@asyncio_test
async def test_the_prompt_is_generated_from_the_manifest_not_the_toggles(toggles, cap):
    """The card's pinned scenario, end to end.

    A server is enabled and fits. Overnight it doubles its tool list. The toggles
    have not changed and still say "enabled" — but the manifest drops it, and the
    prompt for THAT turn must say it is unavailable.
    """
    cap(20)
    toggles("https://gmail.example", "https://slack.example")

    monday = await reg.manifest(USER, mcp=_Mcp(_Server("gmail", 12), _Server("slack", 5)), session_id=SESSION)
    tuesday = await reg.manifest(USER, mcp=_Mcp(_Server("gmail", 12), _Server("slack", 40)), session_id=SESSION)

    before = prompts.system_prompt("attended", date="2026-08-19", now="2026-08-20 14:32 UTC", reach=monday.servers)
    after = prompts.system_prompt("attended", date="2026-08-19", now="2026-08-20 14:32 UTC", reach=tuesday.servers)

    assert "Slack (5 tools)" in before
    assert "NOT loaded this turn" not in before

    assert "NOT loaded this turn" in after
    assert "Slack (40 tools)" not in after
    assert "Gmail (12 tools)" in after, "the one that still fits is untouched"


@asyncio_test
async def test_the_prompt_changes_between_turns_when_a_toggle_does(toggles):
    connected = _Mcp(_Server("gmail", 3), _Server("slack", 4))

    toggles()
    first = prompts.system_prompt(
        "attended",
        date="2026-08-19",
        now="2026-08-20 14:32 UTC",
        reach=(await reg.manifest(USER, mcp=connected, session_id=SESSION)).servers,
    )
    toggles("https://slack.example")
    second = prompts.system_prompt(
        "attended",
        date="2026-08-19",
        now="2026-08-20 14:32 UTC",
        reach=(await reg.manifest(USER, mcp=connected, session_id=SESSION)).servers,
    )

    assert "No service is enabled in this session" in first
    assert "Slack (4 tools)" in second
    assert "NOT enabled here: Gmail." in second
