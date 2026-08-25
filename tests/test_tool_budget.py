"""The tool budget's arithmetic and its refusals, without a database.

`tests/test_api.py` pins the same rules over HTTP with real rows, and skips
wherever Postgres is absent. This file exists because the one thing that must
never be wrong here — the sum that decides whether a request will be refused by
the provider — should not be verifiable only on a machine with a database.
"""

from __future__ import annotations

import pytest

from harness_module import api

pytestmark = pytest.mark.asyncio

SESSION = "s-1"
USER = "u-1"


def _server(label, *, tools, connected=True):
    return {
        "server": label.title(),
        "label": label,
        "name": label.title(),
        "status": "connected" if connected else "disconnected",
        "tool_count": tools,
        "refreshed_at": None,
        "setup_url": None,
        "scopes": [],
        "shares_with": [],
    }


class _FakeArcade:
    def __init__(self, rows):
        self._rows = rows

    async def connections(self, user_id):
        return [dict(r) for r in self._rows]

    async def always(self, user_id):
        """Google Search rides the gateway and is counted in `ours`, not in a row."""
        return []


@pytest.fixture
def wired(monkeypatch):
    """Stand the endpoints up over an in-memory store and a fixed manifest size."""
    state: dict[str, object] = {"on": set(), "rows": [], "ours": 20, "max_tools": 64}

    async def enabled_servers(session_id):
        assert session_id == SESSION
        return set(state["on"])

    async def set_enabled(session_id, server, enabled):
        assert session_id == SESSION
        state["on"].add(server) if enabled else state["on"].discard(server)

    async def owned(session_id, user_id):
        return {"id": session_id}

    real_cfg = api._cfg
    monkeypatch.setattr(api.session_tools, "enabled_servers", enabled_servers)
    monkeypatch.setattr(api.session_tools, "set_enabled", set_enabled)
    monkeypatch.setattr(api, "_owned_session", owned)
    monkeypatch.setattr(api.hands, "arcade", lambda: _FakeArcade(state["rows"]))
    monkeypatch.setattr(api.registry, "local_tools", lambda: dict.fromkeys(range(state["ours"])))
    monkeypatch.setattr(
        api, "_cfg", lambda key, default: state["max_tools"] if key == "llm.max_tools" else real_cfg(key, default)
    )
    return state


async def test_the_meter_is_what_is_left_after_ours(wired):
    """Ours are always loaded and never spend the human's allowance."""
    wired["rows"] = [_server("gmail", tools=12)]

    document = await api.session_tools_state(SESSION, USER)

    assert document["ours"] == 20
    assert document["max_tools"] == 64
    assert document["budget"] == 44
    assert document["used"] == 0


async def test_the_denominator_moves_when_we_add_a_local_tool(wired):
    """A local tool we add is spent from the same 128, so the meter must say so."""
    before = (await api.session_tools_state(SESSION, USER))["budget"]
    wired["ours"] = 21

    assert (await api.session_tools_state(SESSION, USER))["budget"] == before - 1


async def test_used_counts_only_the_servers_this_session_was_given(wired):
    wired["rows"] = [_server("gmail", tools=12), _server("slack", tools=38)]

    await api.set_session_tool(SESSION, "Gmail", {"enabled": True}, USER)
    document = await api.session_tools_state(SESSION, USER)

    assert document["used"] == 12
    assert [s["enabled"] for s in document["servers"]] == [True, False]


async def test_a_toggle_that_would_overflow_the_cap_is_refused_with_both_numbers(wired):
    """The 164-schema request that started this card, refused where it is caused."""
    wired["rows"] = [_server("gmail", tools=12), _server("slack", tools=38)]
    wired["max_tools"] = 44  # 24 for the human, once ours are taken out

    await api.set_session_tool(SESSION, "Gmail", {"enabled": True}, USER)

    with pytest.raises(api.ApiError) as raised:
        await api.set_session_tool(SESSION, "Slack", {"enabled": True}, USER)

    assert raised.value.status == 409
    assert raised.value.code == "tool_budget"
    assert "38" in raised.value.message and "12" in raised.value.message
    assert (await api.session_tools_state(SESSION, USER))["used"] == 12, "and nothing was recorded"


async def test_a_toggle_that_exactly_fills_the_cap_is_allowed(wired):
    """The boundary is `>`, not `>=`: a budget spent to the last slot is legal."""
    wired["rows"] = [_server("gmail", tools=24)]
    wired["max_tools"] = 44

    document = await api.set_session_tool(SESSION, "Gmail", {"enabled": True}, USER)

    assert document["used"] == 24 == document["budget"]


async def test_turning_something_off_is_never_over_budget(wired):
    """Even when the server grew past the cap while it was enabled."""
    wired["rows"] = [_server("gmail", tools=12)]
    await api.set_session_tool(SESSION, "Gmail", {"enabled": True}, USER)

    wired["rows"] = [_server("gmail", tools=9000)]
    document = await api.set_session_tool(SESSION, "Gmail", {"enabled": False}, USER)

    assert document["used"] == 0


async def test_an_already_enabled_server_is_not_re_checked(wired):
    """Re-asserting a toggle must not fail because the server outgrew the budget."""
    wired["rows"] = [_server("gmail", tools=12)]
    await api.set_session_tool(SESSION, "Gmail", {"enabled": True}, USER)

    wired["rows"] = [_server("gmail", tools=9000)]
    document = await api.set_session_tool(SESSION, "Gmail", {"enabled": True}, USER)

    assert document["used"] == 9000, "honest about the state it is in, over budget or not"


async def test_a_server_that_is_not_connected_cannot_be_given_to_a_session(wired):
    wired["rows"] = [_server("linear", tools=3, connected=False)]

    with pytest.raises(api.ApiError) as raised:
        await api.set_session_tool(SESSION, "Linear", {"enabled": True}, USER)

    assert raised.value.status == 409
    assert raised.value.code == "not_connected"


async def test_an_unknown_server_is_not_found(wired):
    with pytest.raises(api.ApiError) as raised:
        await api.set_session_tool(SESSION, "Nope", {"enabled": True}, USER)

    assert raised.value.status == 404


async def test_a_body_without_enabled_is_a_bad_request(wired):
    wired["rows"] = [_server("gmail", tools=12)]

    with pytest.raises(api.ApiError) as raised:
        await api.set_session_tool(SESSION, "Gmail", {}, USER)

    assert raised.value.status == 400


async def test_without_mcp_configured_the_meter_still_reads(wired, monkeypatch):
    """No Arcade client is a server list of none, not a failure."""
    monkeypatch.setattr(api.hands, "arcade", lambda: None)

    document = await api.session_tools_state(SESSION, USER)

    assert document["servers"] == []
    assert document["used"] == 0
    assert document["budget"] == 44
