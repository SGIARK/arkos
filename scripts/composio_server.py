#!/usr/bin/env python3
"""Mint, verify or diff the Composio MCP server against the tracked roster.

    python3 scripts/composio_server.py verify   # does the live server match config_module/composio_tools.json?
    python3 scripts/composio_server.py mint     # create a NEW server from the file, print its id
    python3 scripts/composio_server.py list     # every server on the account, and the ~16 cap

WHY THIS EXISTS. `allowed_tools` cannot be updated in place — the SDK's
`mcp.update` answers ValidationError and PATCH/PUT on /api/v3 and /api/v3.1 are
both 404 (measured 2026-08-26). So a roster change is: edit the JSON, `mint`,
point `mcp.server_id` at the new id, delete the old server. An account caps at
about 16, and past the cap every create fails with a message that reads exactly
like a rejected toolkit list, so `list` is worth running first.

Needs COMPOSIO_API_KEY in .env. Talks to nothing else and writes no files.
"""

from __future__ import annotations

import json
import pathlib
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).parent.parent
ROSTER = ROOT / "config_module" / "composio_tools.json"
BASE = "https://backend.composio.dev"
UA = "buddy-ops/1"


def api_key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if line.startswith("COMPOSIO_API_KEY="):
            key = line.split("=", 1)[1].strip()
            if key:
                return key
    sys.exit("COMPOSIO_API_KEY is not in .env")


KEY = api_key()


def rest(method: str, path: str, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"x-api-key": KEY, "Content-Type": "application/json", "User-Agent": UA},
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:400]}


def roster() -> tuple[dict, list[str]]:
    doc = json.loads(ROSTER.read_text())
    return doc, sorted({t for v in doc["toolkits"].values() for t in v})


def live_tools(server_id: str) -> list[str]:
    """What the server actually serves, over MCP, as a client sees it."""
    url = f"{BASE}/v3/mcp/{server_id}/mcp?user_id=roster-check"
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"x-api-key": KEY, "Content-Type": "application/json",
                 "Accept": "application/json, text/event-stream", "User-Agent": UA},
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        text = r.read().decode()
    payload = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        for line in text.splitlines():          # replies are SSE-framed
            if line.startswith("data: "):
                payload = json.loads(line[6:])
                break
    return sorted(t["name"] for t in ((payload or {}).get("result") or {}).get("tools", []))


def verify() -> int:
    doc, want = roster()
    server_id = doc["server_id"]
    print(f"server {server_id}\nroster file: {len(want)} tools")
    have = live_tools(server_id)
    print(f"live server: {len(have)} tools")
    missing = [t for t in want if t not in have]
    extra = [t for t in have if t not in want]
    for label, rows in (("in the file but NOT served", missing), ("served but NOT in the file", extra)):
        print(f"  {label}: {len(rows)}")
        for t in rows[:20]:
            print(f"    {t}")
    return 1 if (missing or extra) else 0


def mint() -> int:
    doc, want = roster()
    name = f"buddy-prod-{len(want)}"
    status, body = rest("POST", "/api/v3.1/mcp/servers/custom", {
        "name": name, "toolkits": sorted(doc["toolkits"]), "allowed_tools": want,
    })
    if status >= 300:
        print(f"mint failed ({status}): {body}")
        print("If this reads like a rejected toolkit list, check the ~16 server cap with `list`.")
        return 1
    print(f"minted {body.get('id')}  ({len(body.get('allowed_tools') or [])} tools)")
    print("Now: set mcp.server_id in config.yaml, verify, then delete the old server.")
    return 0


def listing() -> int:
    status, body = rest("GET", "/api/v3/mcp/servers?limit=50")
    items = body.get("items") or []
    print(f"{len(items)} servers on the account (cap is about 16)")
    for it in items:
        print(f"  {it.get('id')}  {it.get('name')}")
    return 0


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "verify"
    sys.exit({"verify": verify, "mint": mint, "list": listing}.get(action, verify)())
