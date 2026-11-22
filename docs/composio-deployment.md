# Enable the existing Composio connectors

Deployment checklist for MIT-270 / #266. The integration already exists; enabling
it requires private configuration and a deployment, not a key committed to Git.
Use the normal PR review and merge deployment process. Do not restart production
outside that process.

## Project configuration

The Composio project must own the MCP server referenced by `mcp.server_id` and
the five managed auth configurations in `config_module/config.yaml`:

| ArkOS connector | Composio toolkit |
| --- | --- |
| GitHub | `github` |
| Linear | `linear` |
| Notion | `notion` |
| Google Calendar | `googlecalendar` |
| Google Drive | `googledrive` |

The toolkit catalog is not the project's enabled-app list. Check the project's
auth configurations and MCP server roster, not just catalog visibility.
Other catalog apps are not exposed by this configuration.

## Private deployment configuration

1. Set `COMPOSIO_API_KEY` in the deployment's private `.env` or environment to a
   key from the same project. Never put its value in YAML, Git, PR text, logs or
   screenshots. An empty key makes `harness_module/hands.py` skip the client.
2. Verify `mcp.server_id` and all five `auth_config_id` values belong to that
   project before replacing them. Do not mint a new server merely to enable an
   existing one.
3. Check `mcp.callback_url` against the URL users actually open. The current
   `http://localhost:1121/connections/done` requires the user's matching local
   port forward and authenticated browser session. A remote origin needs an
   appropriate callback configured explicitly; it is not derived automatically
   from `app.public_url`.
4. Apply the configuration through the team's PR deployment process. A change
   to `.env` alone does not reload the running application's cached config.

## Verification after deployment

- Confirm the configured MCP server is readable with the project key and exposes
  the expected tools. `python scripts/composio_server.py verify` reads the key
  from the repository-root `.env` and the server ID from
  `config_module/composio_tools.json`, not runtime YAML. Check that both server
  IDs agree before using it. It lists tools, not user account contents.
- In an authenticated ArkOS session, connect one account and confirm the OAuth
  popup returns successfully and its connector becomes connected.
- Enable that connector for the session, then request one read-only tool call.
  Account connection alone does not enable session tools.
- Confirm unrelated connectors remain disabled unless explicitly enabled.

A readable tool roster does not prove user OAuth works. Do not mark the live
integration verified until the account connection and read-only call succeed.
If configuration must be rolled back, restore the previous private setting and
use the same deployment process. Never include credential values in the rollback
record.

## Provisioning verification (2026-11-21)

Created `arkos-prod-five-apps` through the Composio API using the five existing
managed auth configurations. No duplicate auth configurations were created.
The server exposes exactly the tracked 155 tools: live MCP `tools/list` returned
zero missing and zero extra tools. Gmail and Outlook are excluded.

The vendor-generated per-user URL returned HTTP 307. The resolved `/mcp` form
already used by ArkOS successfully listed the roster with the project key.
No runtime URL behavior change is needed for this endpoint.

The private deployment key was staged without restarting ArkOS. OAuth popup,
session enablement and a real read-only account call remain post-deploy checks.
