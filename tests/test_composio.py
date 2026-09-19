"""The Composio connector backend: dialect, dispatch, consent, isolation."""

from __future__ import annotations

import json
import uuid

import pytest
import pytest_asyncio

from db import pool
from tests.dbgate import require_db
from tool_module import connections as conns
from tool_module.composio_mcp import (
    Composio,
    ComposioClient,
    ComposioError,
    _parse_rpc,
    _render,
    _status_of,
    _to_spec,
    prefix_of,
)
from tool_module.envelope import ToolContext

pytestmark = pytest.mark.asyncio

SERVERS = {
    "gmail": {"server": "GMAIL", "name": "Gmail", "auth_config_id": "ac_gmail"},
    "linear": {
        "server": "LINEAR",
        "name": "Linear",
        "auth_config_id": "ac_linear",
        "auto_approve": ["LINEAR_GET_LINEAR_ISSUE"],
    },
}
MCP_CFG = {"server_id": "srv-1", "api_key": "k", "callback_url": "https://arkos.test/connections/done"}


def _tool(name: str) -> dict:
    return {"name": name, "description": f"{name} does a thing", "inputSchema": {"type": "object"}}


ROSTER = [
    *[_tool(f"GMAIL_{i}") for i in range(3)],
    *[_tool(f"LINEAR_{i}") for i in range(2)],
    _tool("COMPOSIO_SEARCH_TOOLS"),
]


class FakeClient:
    """Stands in for the live server and counts what went over the wire."""

    def __init__(self, *, tools=None, result=None, accounts=None, link=None):
        self.tools = ROSTER if tools is None else tools
        self.result = result or {"content": [{"type": "text", "text": "ok"}]}
        self.accounts = accounts or []
        self.link_response = link or {"redirect_url": "https://connect.composio.dev/link/lk_1", "id": "ca_1"}
        self.calls: list[tuple[str, str]] = []
        self.account_reads = 0
        self.linked: list[tuple[str, str, str]] = []
        self.deleted: list[str] = []

    async def list_tools(self, user_id):
        return self.tools

    async def call_tool(self, user_id, name, args):
        self.calls.append((user_id, name))
        return self.result

    async def link(self, user_id, auth_config_id, callback_url):
        self.linked.append((user_id, auth_config_id, callback_url))
        return self.link_response

    async def connected_accounts(self, user_id):
        self.account_reads += 1
        return self.accounts

    async def delete_connected_account(self, account_id):
        self.deleted.append(account_id)

    async def close(self):
        pass


def _hands(client=None, servers=None) -> Composio:
    c = Composio(servers or SERVERS, MCP_CFG)
    c.client = client or FakeClient()
    return c


def _ctx(user_id: str) -> ToolContext:
    return ToolContext(user_id=user_id)


def _user() -> str:
    return str(uuid.uuid4())


async def _seed_user(user_id: str) -> None:
    """user_connections has an FK to users."""
    await pool.execute("INSERT INTO users (id) VALUES ($1) ON CONFLICT DO NOTHING", uuid.UUID(user_id))


@pytest_asyncio.fixture
async def db():
    """Skip a case unless the real schema is reachable."""
    await require_db()
    yield
    # Each test gets its own event loop, and a pool outliving its loop is dead.
    await pool.close()


# --- the dialect ---------------------------------------------------------------


async def test_the_per_user_url_is_derived_not_minted():
    """Identity is a query param, and the resolved /v3/.../mcp form is what we speak.

    The `/v3.1/mcp/{id}` form the dashboard and SDK hand back answers 307 with a
    JSON body rather than a Location header, so the client never speaks it.
    """
    client = ComposioClient(server_id="srv-1", api_key="k")

    url = client.url_for("alice")

    assert url == "https://backend.composio.dev/v3/mcp/srv-1/mcp?user_id=alice"
    assert "/v3.1/" not in url
    assert client.url_for("bob") != url


async def test_auth_is_the_api_key_header_and_never_a_bearer():
    """Bearer is a 401 (code 906) and sending BOTH is a 401 as well."""
    client = ComposioClient(server_id="srv-1", api_key="secret")

    headers = client._headers()

    assert headers["x-api-key"] == "secret"
    assert "Authorization" not in headers


async def test_an_sse_framed_reply_is_read():
    """Replies arrive `event: message` / `data: {...}` even on a plain POST."""
    framed = 'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"tools":[]}}\n\n'

    assert _parse_rpc(framed, "tools/list") == {"tools": []}


async def test_a_plain_json_reply_is_also_read():
    assert _parse_rpc('{"jsonrpc":"2.0","id":1,"result":{"ok":true}}', "x") == {"ok": True}


async def test_an_unreadable_reply_is_an_error_not_an_empty_result():
    with pytest.raises(ComposioError):
        _parse_rpc("<html>gateway down</html>", "tools/list")


async def test_the_content_block_is_double_parsed_and_successfull_is_read():
    """The block's text is a JSON string, and success is spelled with three l's."""
    payload = {"content": [{"type": "text", "text": json.dumps({"successfull": False, "error": "nope"})}]}

    text, is_error = _render(payload)

    assert is_error, "a payload that says it failed is an error even without isError"
    assert "nope" in text


async def test_the_rest_spelling_of_successful_is_read_too():
    payload = {"content": [{"type": "text", "text": json.dumps({"successful": False})}]}

    _, is_error = _render(payload)

    assert is_error


async def test_a_successful_payload_is_not_an_error():
    payload = {"content": [{"type": "text", "text": json.dumps({"successfull": True, "data": {"x": 1}})}]}

    _, is_error = _render(payload)

    assert not is_error


async def test_status_mapping_keeps_errored_distinct_from_never_connected():
    """ERRORED is `reconnect`: pending would read as never-connected, connected would loop."""
    assert _status_of("ACTIVE") == conns.CONNECTED
    assert _status_of("INITIATED") == conns.PENDING
    assert _status_of("ERRORED") == conns.RECONNECT
    assert _status_of("EXPIRED") == conns.RECONNECT


async def test_the_prefix_is_the_durable_key():
    assert prefix_of("GMAIL_FETCH_EMAILS") == "GMAIL"
    assert prefix_of("GOOGLECALENDAR_EVENTS_LIST") == "GOOGLECALENDAR"


# --- dispatch ------------------------------------------------------------------


async def test_an_unconnected_toolkit_is_refused_before_the_wire(db):
    user_id = _user()
    await _seed_user(user_id)
    client = FakeClient()
    hands = _hands(client)

    result = await hands.call("GMAIL_FETCH_EMAILS", {}, _ctx(user_id))

    assert result.error_kind == "auth_required"
    assert not client.calls, "an unconnected toolkit must not reach the wire at all"


async def test_a_connected_toolkit_dispatches(db):
    user_id = _user()
    await _seed_user(user_id)
    await conns.mark(user_id, "GMAIL", conns.CONNECTED, "ca_1")
    client = FakeClient()
    hands = _hands(client)

    result = await hands.call("GMAIL_FETCH_EMAILS", {}, _ctx(user_id))

    assert result.ok
    assert client.calls == [(user_id, "GMAIL_FETCH_EMAILS")]


async def test_a_revoked_grant_mid_session_flips_the_row_to_reconnect(db):
    """Composio reports a dead grant as a SUCCESSFUL call carrying isError."""
    user_id = _user()
    await _seed_user(user_id)
    await conns.mark(user_id, "GMAIL", conns.CONNECTED, "ca_1")
    refusal = {
        "content": [{"type": "text", "text": f"No connected account found for user ID {user_id} for toolkit gmail"}],
        "isError": True,
    }
    hands = _hands(FakeClient(result=refusal))

    result = await hands.call("GMAIL_FETCH_EMAILS", {}, _ctx(user_id))

    assert result.error_kind == "auth_required"
    assert not result.retryable, "retrying into a dead grant is the loop this exists to stop"
    rows = await conns.load(user_id)
    assert rows["GMAIL"].status == conns.RECONNECT


async def test_composios_own_helper_tools_are_not_reachable(db):
    hands = _hands()

    result = await hands.call("COMPOSIO_SEARCH_TOOLS", {}, _ctx(_user()))

    assert result.error_kind == "not_found"


async def test_an_unknown_prefix_is_not_found(db):
    hands = _hands()

    result = await hands.call("SLACK_SEND_MESSAGE", {}, _ctx(_user()))

    assert result.error_kind == "not_found"


# --- consent -------------------------------------------------------------------


async def test_connect_mints_a_link_against_the_configured_auth_config(db):
    user_id = _user()
    await _seed_user(user_id)
    client = FakeClient()
    hands = _hands(client)

    row = await hands.connect(user_id, "GMAIL")

    assert client.linked == [(user_id, "ac_gmail", "https://arkos.test/connections/done")]
    assert row["setup_url"].startswith("https://connect.composio.dev/")
    assert row["status"] == conns.PENDING


async def test_a_connector_without_an_auth_config_is_refused(db):
    hands = _hands(FakeClient(), servers={"gmail": {"server": "GMAIL", "name": "Gmail"}})

    with pytest.raises(ComposioError, match="auth_config_id"):
        await hands.consent(_user(), "GMAIL")


async def test_an_already_connected_toolkit_does_not_mint_a_second_link(db):
    user_id = _user()
    client = FakeClient(accounts=[{"id": "ca_1", "status": "ACTIVE", "toolkit": {"slug": "gmail"}}])
    hands = _hands(client)

    consent = await hands.consent(user_id, "GMAIL")

    assert consent.connected
    assert not client.linked


async def test_disconnect_revokes_at_composio_and_takes_only_that_toolkit(db):
    """A connected account is per TOOLKIT: Gmail going does not take Calendar."""
    user_id = _user()
    await _seed_user(user_id)
    await conns.mark(user_id, "GMAIL", conns.CONNECTED, "ca_1")
    await conns.mark(user_id, "LINEAR", conns.CONNECTED, "ca_2")
    client = FakeClient(accounts=[{"id": "ca_1", "status": "ACTIVE", "toolkit": {"slug": "gmail"}}])
    hands = _hands(client)

    taken = await hands.disconnect(user_id, "GMAIL")

    assert taken == ["GMAIL"]
    assert client.deleted == ["ca_1"]
    rows = await conns.load(user_id)
    assert "GMAIL" not in rows
    assert rows["LINEAR"].connected, "a sibling toolkit must survive"


async def test_the_callback_settles_the_row_from_the_account_id(db):
    """`/connections/done` knows an id and a status, never which toolkit."""
    user_id = _user()
    await _seed_user(user_id)
    client = FakeClient(accounts=[{"id": "ca_9", "status": "ACTIVE", "toolkit": {"slug": "gmail"}}])
    hands = _hands(client)

    settled = await hands.reconcile(user_id, "ca_9")

    assert settled == "GMAIL"
    rows = await conns.load(user_id)
    assert rows["GMAIL"].connected
    assert rows["GMAIL"].connected_account_id == "ca_9"


async def test_an_account_id_that_is_not_this_users_settles_nothing(db):
    """A url is only a claim; reconciling asks Composio whose account that id is."""
    user_id = _user()
    hands = _hands(FakeClient(accounts=[]))

    assert await hands.reconcile(user_id, "ca_someone_else") is None
    assert not await conns.load(user_id)


# --- the manifest half ----------------------------------------------------------


async def test_reach_returns_only_connected_toolkits(db):
    user_id = _user()
    await _seed_user(user_id)
    await conns.mark(user_id, "GMAIL", conns.CONNECTED, "ca_1")
    hands = _hands()

    reached = await hands.reach(user_id)

    assert [s.server for s in reached] == ["GMAIL"]
    assert len(reached[0].specs) == 3


async def test_a_reconnect_row_is_not_reachable(db):
    user_id = _user()
    await _seed_user(user_id)
    await conns.mark(user_id, "GMAIL", conns.RECONNECT, "ca_1")
    hands = _hands()

    assert await hands.reach(user_id) == []


async def test_nothing_is_always_on_this_wire(db):
    """Web search is ours and native; no toolkit is unconditional on this wire."""
    assert await _hands().always(_user()) == []


async def test_auto_approve_waives_the_gate_for_the_named_tool():
    spec = _to_spec(_tool("LINEAR_GET_LINEAR_ISSUE"), ["LINEAR_GET_LINEAR_ISSUE"])
    other = _to_spec(_tool("LINEAR_CREATE_LINEAR_ISSUE"), ["LINEAR_GET_LINEAR_ISSUE"])

    assert not spec.requires_approval
    assert other.requires_approval, "a remote tool is gated unless config names it"


async def test_the_toggle_path_asks_the_vendor_nothing(db):
    """`refresh=False` answers from stored rows: the toggle needs no round trip."""
    user_id = _user()
    await _seed_user(user_id)
    await conns.mark(user_id, "GMAIL", conns.CONNECTED, "ca_1")
    client = FakeClient()
    hands = _hands(client)

    rows = await hands.connections(user_id, refresh=False)

    assert client.account_reads == 0, "the toggle path must not reach the vendor"
    gmail = next(r for r in rows if r["server"] == "GMAIL")
    assert gmail["status"] == conns.CONNECTED
    assert gmail["tool_count"] == 3
    assert gmail["account_id"] == "ca_1"


async def test_the_panel_path_does_refresh(db):
    """The settings panel is the moment the human is looking; it pays the trip."""
    user_id = _user()
    await _seed_user(user_id)
    client = FakeClient(accounts=[{"id": "ca_1", "status": "ACTIVE", "toolkit": {"slug": "gmail"}}])
    hands = _hands(client)

    rows = await hands.connections(user_id)

    assert client.account_reads == 1
    assert next(r for r in rows if r["server"] == "GMAIL")["status"] == conns.CONNECTED
    assert (await conns.load(user_id))["GMAIL"].connected
