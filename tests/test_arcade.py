"""The Arcade gateway client: the handshake, the paging, consent, and dispatch.

The wire facts pinned here were measured against the live gateway before any of
this was written — see `docs/implementation_notes.md` § "Arcade gateway" and
`scripts/probe_arcade.py`. The tests that touch connection rows run against a
real Postgres with the migrations applied; each uses its own user id.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from db import pool
from tests.dbgate import require_db
from tool_module import connections as conns
from tool_module.arcade import Arcade, ArcadeClient, ArcadeError, Consent, _render, _to_spec, prefix_of
from tool_module.envelope import ToolContext

pytestmark = pytest.mark.asyncio

GATEWAY = "https://api.arcade.dev/mcp/gw_test"

SERVERS = {
    "gmail": {"server": "Gmail", "name": "Gmail", "consent_tool": "Gmail_ListEmails"},
    "google-calendar": {"server": "GoogleCalendar", "name": "Google Calendar"},
    "linear": {"server": "Linear", "name": "Linear", "auto_approve": ["Linear_GetIssue"]},
}
MCP_CFG = {"api_key": "k", "gateway_url": GATEWAY, "search": {"server": "GoogleSearch"}}


def _tool(name: str) -> dict:
    return {"name": name, "description": f"{name} does a thing", "inputSchema": {"type": "object"}}


ROSTER = [
    *[_tool(f"Gmail_{i}") for i in range(3)],
    *[_tool(f"GoogleCalendar_{i}") for i in range(2)],
    *[_tool(f"Linear_{i}") for i in range(2)],
    _tool("Linear_GetIssue"),
    _tool("GoogleSearch_Search"),
    _tool("Arcade_ListApps"),
]


class FakeClient:
    """Counts what goes over the wire, and answers the way the gateway did."""

    def __init__(self, *, tools=None, consent=None):
        self.tools = ROSTER if tools is None else tools
        # server -> the `/v1/tools/authorize` payload for it
        self.consent = consent or {}
        self.listed: list[str] = []
        self.calls: list[tuple[str, str]] = []
        self.authorized: list[tuple[str, str]] = []
        self.deleted: list[str] = []
        self.held: list[dict] = []
        self.result = {"content": [{"type": "text", "text": "ok"}]}

    async def list_tools(self, user_id):
        self.listed.append(user_id)
        return list(self.tools)

    async def call_tool(self, user_id, name, args):
        self.calls.append((user_id, name))
        return self.result

    async def authorize(self, user_id, tool_name):
        self.authorized.append((user_id, tool_name))
        return self.consent.get(prefix_of(tool_name), {"status": "pending", "url": "https://x"})

    async def user_connections(self, user_id):
        return list(self.held)

    async def delete_connection(self, user_id, connection_id):
        self.deleted.append(connection_id)

    async def close(self):
        pass


def _hands(client=None, servers=None) -> Arcade:
    a = Arcade(servers or SERVERS, MCP_CFG)
    a.client = client or FakeClient()
    return a


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


# --- the transport ------------------------------------------------------------


async def test_the_tool_list_is_paged_until_no_cursor_comes_back():
    """The bug this exists for: one page is 100 of 169, and the 69 look absent.

    Reading a single page did not fail — it returned a plausible roster with two
    whole apps missing, and every symptom pointed at the gateway's app selection
    rather than at the reader.
    """
    pages = [
        {"tools": [_tool(f"Gmail_{i}") for i in range(100)], "nextCursor": "p2"},
        {"tools": [_tool(f"Notion_{i}") for i in range(10)], "nextCursor": "p3"},
        {"tools": [_tool(f"MicrosoftOutlookMail_{i}") for i in range(30)]},
    ]
    seen: list[dict] = []

    client = ArcadeClient("k", GATEWAY)

    async def rpc(user_id, method, params=None):
        seen.append(params or {})
        return pages[len(seen) - 1]

    client.rpc = rpc

    tools = await client.list_tools("u1")

    assert len(tools) == 140
    assert {prefix_of(t["name"]) for t in tools} == {"Gmail", "Notion", "MicrosoftOutlookMail"}
    assert seen == [{}, {"cursor": "p2"}, {"cursor": "p3"}], "each page asks for the next by cursor"


async def test_paging_stops_and_says_so_rather_than_looping_forever():
    client = ArcadeClient("k", GATEWAY)

    async def rpc(user_id, method, params=None):
        return {"tools": [_tool("Gmail_A")], "nextCursor": "always"}

    client.rpc = rpc

    tools = await client.list_tools("u1")

    assert len(tools) == 20, "capped at _MAX_PAGES rather than spinning"


async def test_the_handshake_reads_the_session_id_off_the_header():
    """The one detail that decides whether a from-scratch client works at all."""
    posts: list[dict] = []

    class Response:
        def __init__(self, headers):
            self.status = 200
            self.headers = headers

        async def text(self):
            return '{"jsonrpc":"2.0","id":"1","result":{"protocolVersion":"2025-11-25"}}'

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class Http:
        def post(self, url, json=None, headers=None):
            posts.append({"body": json, "headers": headers})
            return Response({"Mcp-Session-Id": "sid-1"})

    client = ArcadeClient("k", GATEWAY)
    client.http = lambda: _done(Http())

    session_id = await client._session("alice")

    assert session_id == "sid-1"
    assert posts[0]["body"]["method"] == "initialize"
    assert posts[0]["headers"]["Arcade-User-Id"] == "alice"
    assert posts[0]["headers"]["Authorization"] == "Bearer k"
    assert "text/event-stream" in posts[0]["headers"]["Accept"]
    assert "Mcp-Session-Id" not in posts[0]["headers"], "the first call cannot carry one"
    assert posts[1]["body"]["method"] == "notifications/initialized"
    assert "id" not in posts[1]["body"], "a notification carries no id"
    assert posts[1]["headers"]["Mcp-Session-Id"] == "sid-1"


async def test_each_user_gets_their_own_gateway_session():
    """The session is minted under `Arcade-User-Id`, so it cannot be shared."""
    minted: list[str] = []

    class Response:
        status = 200

        def __init__(self, headers):
            self.headers = headers

        async def text(self):
            return "{}"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class Http:
        def post(self, url, json=None, headers=None):
            user = headers["Arcade-User-Id"]
            if json.get("method") == "initialize":
                minted.append(user)
            return Response({"Mcp-Session-Id": f"sid-{user}"})

    client = ArcadeClient("k", GATEWAY)
    client.http = lambda: _done(Http())

    assert await client._session("alice") == "sid-alice"
    assert await client._session("bob") == "sid-bob"
    assert await client._session("alice") == "sid-alice", "and it is reused"
    assert minted == ["alice", "bob"], "one handshake each, not one per call"


async def test_gateway_sessions_evict_the_least_recently_used_user():
    client = ArcadeClient("k", GATEWAY)
    client._max_cached_users = 2
    client._handshake = lambda user_id: _done(f"sid-{user_id}")

    await client._session("alice")
    await client._session("bob")
    await client._session("alice")
    await client._session("carol")

    assert list(client._sessions) == ["alice", "carol"]
    assert len(client._session_locks) == 64


async def test_tool_and_setup_caches_are_bounded():
    hands = Arcade(SERVERS, MCP_CFG)
    hands._max_cached_users = 2
    hands.client = FakeClient()

    await hands.tools_by_server("alice")
    await hands.tools_by_server("bob")
    await hands.tools_by_server("alice")
    await hands.tools_by_server("carol")
    for user_id in ("alice", "bob", "carol"):
        for server in hands.prefixes:
            hands._remember_setup_url(user_id, server, f"https://setup/{user_id}/{server}")

    assert list(hands._tools) == ["alice", "carol"]
    assert len(hands._setup_urls) == 2 * len(hands.prefixes)
    assert len(hands._tool_locks) == 64


def _done(value):
    """A coroutine that is already finished, for stubbing an awaited accessor."""

    async def run():
        return value

    return run()


# --- consent is the status read -----------------------------------------------


async def test_a_granted_service_reads_connected_and_offers_no_link():
    hands = _hands(FakeClient(consent={"Gmail": {"status": "completed", "provider_id": "arcade-google"}}))

    consent = await hands.consent("alice", "Gmail")

    assert consent.connected
    assert consent.setup_url is None


async def test_an_ungranted_service_carries_the_link_and_the_scopes():
    hands = _hands(
        FakeClient(
            consent={
                "Gmail": {
                    "status": "pending",
                    "url": "https://accounts.google.com/o/oauth2/v2/auth?x=1",
                    "provider_id": "arcade-google",
                    "scopes": ["https://www.googleapis.com/auth/gmail.readonly"],
                }
            }
        )
    )

    consent = await hands.consent("alice", "Gmail")

    assert not consent.connected
    assert consent.setup_url.startswith("https://accounts.google.com/")
    assert consent.scopes == ("https://www.googleapis.com/auth/gmail.readonly",)


async def test_consent_uses_the_tool_named_in_config():
    """The consent tool is a measured choice: its scopes must cover the whole app."""
    client = FakeClient()
    hands = _hands(client)

    await hands.consent("alice", "Gmail")

    assert client.authorized == [("alice", "Gmail_ListEmails")]


async def test_consent_falls_back_to_a_listed_tool_when_config_names_none():
    client = FakeClient()
    hands = _hands(client)

    await hands.consent("alice", "GoogleCalendar")

    assert client.authorized == [("alice", "GoogleCalendar_0")]


async def test_authorize_answering_with_neither_a_url_nor_completion_is_an_error():
    hands = _hands(FakeClient(consent={"Gmail": {"status": "pending"}}))

    with pytest.raises(ArcadeError):
        await hands.consent("alice", "Gmail")


async def test_refresh_status_records_every_connector(db):
    user_id = _user()
    await _seed_user(user_id)
    hands = _hands(
        FakeClient(
            consent={
                "Gmail": {"status": "completed", "provider_id": "arcade-google"},
                "GoogleCalendar": {"status": "pending", "url": "https://c", "provider_id": "arcade-google"},
                "Linear": {"status": "pending", "url": "https://l", "provider_id": "arcade-linear"},
            }
        )
    )

    await hands.refresh_status(user_id)
    stored = await conns.load(user_id)

    assert stored["Gmail"].connected
    assert not stored["GoogleCalendar"].connected
    assert set(stored) == {"Gmail", "GoogleCalendar", "Linear"}


async def test_services_sharing_a_sign_in_are_named_on_each_other(db):
    """Arcade's connection is per provider account, so a revoke is shared."""
    user_id = _user()
    await _seed_user(user_id)
    hands = _hands(
        FakeClient(
            consent={
                "Gmail": {"status": "completed", "provider_id": "arcade-google"},
                "GoogleCalendar": {"status": "completed", "provider_id": "arcade-google"},
                "Linear": {"status": "completed", "provider_id": "arcade-linear"},
            }
        )
    )

    rows = {row["server"]: row for row in await hands.connections(user_id)}

    assert rows["Gmail"]["shares_with"] == ["Google Calendar"]
    assert rows["GoogleCalendar"]["shares_with"] == ["Gmail"]
    assert rows["Linear"]["shares_with"] == []


async def test_google_search_is_in_no_connection_row(db):
    """It is ours: no grant to make, so nothing to offer a human."""
    user_id = _user()
    await _seed_user(user_id)
    hands = _hands()

    rows = await hands.connections(user_id)

    assert "GoogleSearch" not in {row["server"] for row in rows}
    assert "Arcade" not in {row["server"] for row in rows}


# --- the manifest half ---------------------------------------------------------


async def test_reach_offers_only_the_connectors_this_user_has_connected(db):
    user_id = _user()
    await _seed_user(user_id)
    await conns.mark(user_id, "Gmail", conns.CONNECTED)
    await conns.mark(user_id, "Linear", conns.PENDING)
    hands = _hands()

    reached = {server.server: server for server in await hands.reach(user_id)}

    assert set(reached) == {"Gmail"}
    assert len(reached["Gmail"].specs) == 3
    assert reached["Gmail"].name == "Gmail"


async def test_google_search_is_always_reachable_and_never_a_server(db):
    user_id = _user()
    await _seed_user(user_id)
    hands = _hands()

    always = await hands.always(user_id)

    assert [spec.name for spec in always] == ["GoogleSearch_Search"]
    assert always[0].readonly, "a search changes nothing"
    assert not always[0].requires_approval, "and gating it would card the human on every lookup"
    assert not await hands.reach(user_id), "with nothing connected, no server is reached"


async def test_the_meta_tool_is_never_offered(db):
    user_id = _user()
    await _seed_user(user_id)
    await conns.mark(user_id, "Gmail", conns.CONNECTED)
    hands = _hands()

    offered = {spec.name for spec in await hands.specs(user_id)}

    assert "Arcade_ListApps" not in offered


async def test_connector_tools_require_approval_unless_config_waives_it(db):
    user_id = _user()
    await _seed_user(user_id)
    await conns.mark(user_id, "Linear", conns.CONNECTED)
    hands = _hands()

    specs = {spec.name: spec for spec in await hands.specs(user_id)}

    assert specs["Linear_GetIssue"].requires_approval is False
    assert specs["Linear_0"].requires_approval is True
    assert specs["Linear_0"].readonly is False, "a remote server never says whether a tool mutates"


async def test_the_tool_list_is_read_once_per_user_per_ttl(db):
    user_id = _user()
    await _seed_user(user_id)
    await conns.mark(user_id, "Gmail", conns.CONNECTED)
    client = FakeClient()
    hands = _hands(client)

    await hands.reach(user_id)
    await hands.reach(user_id)

    assert client.listed == [user_id]


# --- dispatch ------------------------------------------------------------------


async def test_a_never_connected_service_is_refused_without_reaching_the_gateway(db):
    user_id = _user()
    await _seed_user(user_id)
    client = FakeClient()
    hands = _hands(client)

    result = await hands.call("Gmail_0", {}, _ctx(user_id))

    assert not result.ok
    assert result.error_kind == "auth_required"
    assert not result.retryable
    assert "Settings" in result.content
    assert client.calls == [], "and no call was made against a grant that does not exist"


async def test_a_connected_service_dispatches(db):
    user_id = _user()
    await _seed_user(user_id)
    await conns.mark(user_id, "Gmail", conns.CONNECTED)
    client = FakeClient()
    hands = _hands(client)

    result = await hands.call("Gmail_0", {}, _ctx(user_id))

    assert result.ok
    assert result.content == "ok"
    assert client.calls == [(user_id, "Gmail_0")]


async def test_google_search_needs_no_connection(db):
    """Its key is ours and app-level, so there is no per-user grant to check."""
    user_id = _user()
    await _seed_user(user_id)
    client = FakeClient()
    hands = _hands(client)

    result = await hands.call("GoogleSearch_Search", {"q": "x"}, _ctx(user_id))

    assert result.ok
    assert client.calls == [(user_id, "GoogleSearch_Search")]


async def test_the_meta_tool_is_not_callable(db):
    user_id = _user()
    await _seed_user(user_id)
    client = FakeClient()
    hands = _hands(client)

    result = await hands.call("Arcade_ListApps", {}, _ctx(user_id))

    assert not result.ok
    assert result.error_kind == "not_found"
    assert client.calls == []


async def test_an_unknown_prefix_is_not_found(db):
    user_id = _user()
    await _seed_user(user_id)
    hands = _hands()

    result = await hands.call("Dropbox_Upload", {}, _ctx(user_id))

    assert not result.ok
    assert result.error_kind == "not_found"


async def test_a_gateway_failure_is_an_upstream_error_not_a_dead_run(db):
    user_id = _user()
    await _seed_user(user_id)
    await conns.mark(user_id, "Gmail", conns.CONNECTED)
    client = FakeClient()

    async def boom(user, name, args):
        raise ArcadeError("gateway said no")

    client.call_tool = boom
    hands = _hands(client)

    result = await hands.call("Gmail_0", {}, _ctx(user_id))

    assert not result.ok
    assert result.error_kind == "upstream_error"
    assert "gateway said no" in result.content


# --- disconnect ----------------------------------------------------------------


async def test_disconnect_revokes_at_arcade_and_takes_the_siblings_with_it(db):
    user_id = _user()
    await _seed_user(user_id)
    client = FakeClient(
        consent={
            "Gmail": {"status": "completed", "provider_id": "arcade-google"},
            "GoogleCalendar": {"status": "completed", "provider_id": "arcade-google"},
            "Linear": {"status": "completed", "provider_id": "arcade-linear"},
        }
    )
    client.held = [
        {"id": "conn-google", "provider_id": "arcade-google"},
        {"id": "conn-linear", "provider_id": "arcade-linear"},
    ]
    hands = _hands(client)
    await hands.refresh_status(user_id)

    taken = await hands.disconnect(user_id, "Gmail")
    stored = await conns.load(user_id)

    assert set(taken) == {"Gmail", "GoogleCalendar"}
    assert client.deleted == ["conn-google"], "the Linear connection is untouched"
    assert "Gmail" not in stored and "GoogleCalendar" not in stored
    assert stored["Linear"].connected, "which is not this button's to take"


# --- envelope shape ------------------------------------------------------------


async def test_an_mcp_error_result_becomes_a_failed_envelope(db):
    user_id = _user()
    await _seed_user(user_id)
    await conns.mark(user_id, "Gmail", conns.CONNECTED)
    client = FakeClient()
    client.result = {"isError": True, "content": [{"type": "text", "text": "nope"}]}
    hands = _hands(client)

    result = await hands.call("Gmail_0", {}, _ctx(user_id))

    assert not result.ok
    assert result.error_kind == "upstream_error"
    assert "nope" in result.content


async def test_content_blocks_are_flattened_not_dropped():
    text, is_error = _render({"content": [{"type": "text", "text": "a"}, {"type": "image", "data": "b"}]})

    assert "a" in text and "b" in text
    assert not is_error


async def test_remote_tools_are_never_marked_readonly():
    assert _to_spec({"name": "Gmail_Send"}).readonly is False


async def test_a_prefix_is_everything_before_the_first_underscore():
    assert prefix_of("MicrosoftOutlookMail_ListEmails") == "MicrosoftOutlookMail"
    assert prefix_of("GoogleSearch_Search") == "GoogleSearch"


async def test_an_oversized_result_is_stored_and_referenced(db):
    user_id = _user()
    await _seed_user(user_id)
    await conns.mark(user_id, "Gmail", conns.CONNECTED)
    client = FakeClient()
    client.result = {"content": [{"type": "text", "text": "x" * 20000}]}
    hands = _hands(client)

    stored: list[str] = []

    async def store_blob(text):
        stored.append(text)
        return "ref-1"

    result = await hands.call("Gmail_0", {}, ToolContext(user_id=user_id, store_blob=store_blob))

    assert result.ok
    assert result.ref == "ref-1"
    assert len(stored[0]) == 20000, "the blob holds the whole thing"
    assert "read_result" in result.content


# --- isolation -----------------------------------------------------------------


async def test_one_users_connection_is_never_visible_to_another(db):
    alice, bob = _user(), _user()
    await _seed_user(alice)
    await _seed_user(bob)
    await conns.mark(alice, "Gmail", conns.CONNECTED)
    hands = _hands()

    assert [s.server for s in await hands.reach(alice)] == ["Gmail"]
    assert await hands.reach(bob) == []


async def test_a_tool_another_user_connected_is_not_callable(db):
    alice, bob = _user(), _user()
    await _seed_user(alice)
    await _seed_user(bob)
    await conns.mark(alice, "Gmail", conns.CONNECTED)
    client = FakeClient()
    hands = _hands(client)

    result = await hands.call("Gmail_0", {}, _ctx(bob))

    assert not result.ok
    assert result.error_kind == "auth_required"
    assert client.calls == []


async def test_consent_carries_the_provider_so_siblings_can_be_grouped():
    consent = Consent("Gmail", conns.CONNECTED, None, "arcade-google", ())

    assert consent.connected
    assert consent.provider_id == "arcade-google"
