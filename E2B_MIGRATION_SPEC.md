# Hosted E2B to local E2B on ark.mit.edu

Implementation specification and handoff for an AI coding agent.

Prepared: **2026-10-02**. Target confirmed by the user: **hosted E2B → local E2B on `ark.mit.edu`**.

## 1. Assignment and authority

Make Arkos use E2B running locally on `service@ark.mit.edu` for its session sandboxes. Preserve the existing shell/file tools, session lifecycle, Supabase-backed durable files, and user isolation. Finish a reproducible deployment with small, understandable changes and explicit verification.

**This document is a specification, not a deployment action.** The user previously requested that E2B be stopped. Its VM was gracefully stopped on September 26 and its container's automatic restart was disabled. Writing or reviewing this document does not resume it. When the user subsequently instructs an agent to implement and run this migration, that instruction authorizes the implementation phases below; otherwise keep runtime work stopped. Do not restart E2B merely because you discover its stopped container.

The current task is not a migration away from the E2B API, a replacement with plain Docker execution, or a move back to E2B Cloud. The existing E2B integration should remain the application boundary.

### Constraints carried forward from the user

- Install, run, and test Arkos and its providers on `ark.mit.edu`, not on the user's workstation. This workstation holds the specification, command record, and short maintenance guide.
- Treat `ark.mit.edu` as a shared server. Preserve unrelated processes, the existing inference service, user data, and the user's edits.
- Keep E2B and Supabase in separate attachable tmux sessions. Preserve a separate Arkos API session as well.
- Use minimal disk, memory, dependencies, and custom code. Reuse the existing stopped VM and downloaded artifacts when sound.
- After **each remote command**, update the operator's local `ARK_COMMANDS.md` with context, exact command, sanitized output, exit status, changes, and whether the intended result occurred. Record failures and later completion of commands that initially return a running handle.
- Keep the operator's local `ARK_SETUP.md` short and accurate. Put detailed implementation evidence in the command record and this specification's eventual verification report.
- Keep secrets on the server. Never print or paste `.env`, private SSH keys, cookies, recovery links, or E2B credential-bearing output into the record or chat.
- Do not reboot the shared host, restart its Docker daemon or SSH service, disable its firewall, flush firewall rules, change host kernel settings, or delete unrelated resources as part of this migration.

No parallel deployment agent is required. The work should be understandable as one ordered implementation.

## 2. Evidence and starting point

### 2.1 What was inspected for this specification

The local source checkout is commit `a522ff4a95b323d158da003fd1cac448f7bf6548`. The application source was read locally on October 2. The only pre-existing untracked files were `ARK_COMMANDS.md` and `ARK_SETUP.md`. No server command was run and no service was started while preparing this specification.

**Public handoff:** this branch publishes the specification only. `ARK_COMMANDS.md` and `ARK_SETUP.md` are operational records kept on the operator's workstation, not files included in the public branch. The implementation requirements and relevant historical findings are reproduced here. Numbered command references below are optional audit cross-references. If those records are unavailable, establish fresh server evidence and create new local records; do not assume that missing documentation means the installation is absent or disposable.

Server details below are **historical observations from September 22–26**, recorded in those files. They are a starting inventory, not proof of the server's current state. Reinspect before implementing. Read command entries 25, 28, 41, 49, 51, 59, 61–64, and 73–76 in particular. The command record is an audit trail; do not execute it wholesale as a bootstrap script.

| Item | Last recorded state | Implementation consequence |
| --- | --- | --- |
| Login | `service@ark.mit.edu`; user belongs to `docker` | Reuse this account and authorized SSH identity. |
| Application | `/home/service/dev/arkos`, same base commit as above | Inspect the remote diff before applying changes. |
| Python | Existing `.venv`, Python 3.14 on server; CI targets Python 3.11 | Reuse the venv; retain Python 3.11 source compatibility. |
| Source edits on server | `config_module/config.yaml`, `harness_module/workspace.py`, `tests/test_workspace.py`, `tool_module/sandbox/manager.py`, and an older user edit to `model_module/run.sh` | The workstation and server are not identical. Preserve and reconcile these changes. |
| SDK versions | `e2b-code-interpreter==2.10.0`, transitive `e2b==2.51.0` | Revalidate installed versions; these are a previously exercised baseline, not an instruction to upgrade. |
| Local Supabase | `/home/service/dev/arkos-local/supabase`; six healthy containers on September 26 | Preserve its database, Auth accounts, private bucket, credentials, and volumes. |
| E2B outer VM | `/home/service/dev/arkos-local/e2b-vm`; Compose project `arkos-e2b-vm`, service `vm` | Reuse this installation after inspecting its files and state. |
| E2B guest | Ubuntu 24.04; guest user `arkos`; `/opt/e2b` | Run E2B's privileged setup only inside this guest. |
| E2B source pin | `9dd5b727318831ebbdd84cc9f51b25c5f3af96c8` | Verify `SOURCE_COMMIT`, cached files, and image digests before reuse. |
| E2B stopped state | Container `arkos-e2b-vm-vm-1` exited; live container restart policy `no` | Its original Compose file said `unless-stopped`; resolve this difference deliberately. |
| Existing model | Qwen3-8B at `127.0.0.1:30000/v1`; container was `cool_mahavira` | Do not redeploy, restart, rename, or tune it. Names may have changed; verify ownership. |
| tmux | Custom socket `arkos`: `arkos-supabase`, `arkos-e2b`, `arkos-api` | Use `tmux -L arkos`; do not repurpose the default tmux server. |
| Existing default tmux | `inference_engine`; also an obsolete E2B setup shell | Preserve inference. Remove an obsolete shell only after proving it is ours and idle. |
| Previous tests | 862 non-integration tests and 12 live provider tests passed on September 22 | Useful history, but not full migration acceptance. |
| Previous live application checks | Signup, local email confirmation, login, upload, and readback passed | The final model-driven agent workflow never started. |
| Private verification state | `.venv/setup-verification.json`, mode `0600` | Contains a temporary account and test object IDs. Inspect privately, refresh expired auth, and do not assume previous requests ran. |

The old integration attempt is incomplete. In particular, it did not prove the full agent-to-storage workflow, reopening in a fresh sandbox, controlled restart recovery, or tested backup/restore.

### 2.2 SSH incident and operational lesson

During the earlier setup there was a burst of six SSH attempts within 27 seconds, followed by a brief refusal. A later sustained timeout occurred at a different time. The user reported that internal and mobile networks could connect and that a sysadmin mentioned a UFW threshold of six attempts. Attribution of the sustained outage was never established.

Use one multiplexed connection or an interactive SSH session for the implementation. Do not repeatedly launch new SSH connections when an operation can use an existing connection. A timeout is not proof that the server or a process has stopped. Preserve running handles and inspect them before restarting work. Do not disable firewall protections to make deployment easier.

### 2.3 Source map

Read these files before editing:

| File | Relevant responsibility |
| --- | --- |
| [tool_module/sandbox/manager.py](tool_module/sandbox/manager.py) | E2B SDK boundary; creation, reconnect, commands, file I/O, pause, kill, slots, and cached handles. |
| [tool_module/sandbox/tools.py](tool_module/sandbox/tools.py) | Tool contracts exposed to the model; shell quoting; read-before-edit; lazy sandbox acquisition. |
| [tool_module/tools/sandbox.py](tool_module/tools/sandbox.py) | Tool exports/registration surface. |
| [harness_module/workspace.py](harness_module/workspace.py) | Folder claims, materialization, hashing, tar transfers, sentinel verification, and flush. |
| [harness_module/runner.py](harness_module/runner.py) | Slot and folder-lease acquisition, flush-before-release, pause/reap ordering, and retrying teardown. |
| [harness_module/store.py](harness_module/store.py), [harness_module/blobs.py](harness_module/blobs.py) | Durable file tree and Supabase blob storage. |
| [harness_module/api.py](harness_module/api.py) | Startup sweeps, auth, files, session commands, and read-only live filesystem inspection. |
| [config_module/loader.py](config_module/loader.py), [config_module/config.yaml](config_module/config.yaml) | YAML/environment loading and coherence checks. |
| [.env.example](.env.example), [requirements.txt](requirements.txt) | Public configuration contract and SDK dependency. |
| [db/migrations/0002_session_sandboxes.sql](db/migrations/0002_session_sandboxes.sql), [0003](db/migrations/0003_workspace_sentinel.sql), [0004](db/migrations/0004_slot_expiry.sql) | Persistent sandbox IDs, workspace nonce, and slot expiry. |
| [tests/test_sandbox_tools.py](tests/test_sandbox_tools.py), [tests/test_sandbox_pool.py](tests/test_sandbox_pool.py), [tests/test_workspace.py](tests/test_workspace.py) | Existing behavioral regression coverage. |
| [tests/test_sandbox_integration.py](tests/test_sandbox_integration.py), [tests/test_store_integration.py](tests/test_store_integration.py) | Real SDK and real blob-provider tests. |
| [tests/conftest.py](tests/conftest.py), [tests/dbgate.py](tests/dbgate.py), [.github/workflows/ci.yml](.github/workflows/ci.yml) | Test database handling, skip behavior, and required checks. |

## 3. Target architecture

Keep the existing Python E2B adapter. Move both its control-plane and sandbox file/command traffic to the local installation. Supabase remains the durable store; E2B disks remain disposable caches.

```text
User browser
    |
    | SSH tunnel to Arkos and local Supabase Auth
    v
ark.mit.edu
    Arkos API / runner
        |-- SQL + Auth + private file blobs --> existing local Supabase
        |-- model calls ---------------------> existing local SGLang
        |
        | E2B SDK: control API 127.0.0.1:13000
        | E2B SDK: command/file proxy 127.0.0.1:13002
        v
    dedicated outer Docker container (only /dev/kvm passed through)
        QEMU Ubuntu guest, user-mode networking
            E2B Embed control plane + orchestrator
                Firecracker sandbox for each Arkos session
                    /home/user/store/<claimed folder>/...

Durable bytes: Supabase -> Arkos -> sandbox -> Arkos -> Supabase
```

The outer VM contains E2B's direct kernel, network-namespace, hugepage, and firewall setup. It does **not** eliminate host impact: the outer container still uses Docker bridge/NAT state, CPU, RAM, disk, and the host KVM subsystem. Preserve that distinction in documentation.

### Endpoint contract

These are addresses as seen by the Arkos process running on the host:

| Purpose | Host address | Guest/service address |
| --- | --- | --- |
| Arkos | `http://127.0.0.1:1121` | Existing host process |
| Supabase gateway | `http://127.0.0.1:18000` | Existing Supabase Compose gateway |
| Supabase PostgreSQL | `127.0.0.1:15432` | Existing database container port 5432 |
| Local email inbox | `http://127.0.0.1:18025` | Existing Mailpit UI |
| E2B control API | `http://127.0.0.1:13000` | Outer port 3000 forwarded to guest 3000 |
| E2B dashboard | `http://127.0.0.1:13001` | Outer port 3001 forwarded to guest 3001 |
| E2B command/file proxy | `http://127.0.0.1:13002` | Outer port 3002 forwarded to guest 3002 |
| E2B guest SSH | `127.0.0.1:12222` | Outer port 2222 forwarded to guest 22 |

Do not point `E2B_API_URL` at the dashboard or put `E2B_SANDBOX_URL` on the API port. Both SDK endpoints are required. These host loopback addresses would need revisiting if Arkos itself moved into a container; containerizing Arkos is outside this migration.

All outer published ports must bind `127.0.0.1`. Do not publish the guest's databases, Redis, ClickHouse, orchestrator, or Docker socket onto the shared host's network. Dashboard access, if needed, stays behind SSH.

### Resource envelope

Reuse the recorded starting envelope unless measurements require a smaller one:

| Resource | Starting configuration |
| --- | --- |
| QEMU guest | 4 vCPUs, 8 GiB RAM |
| Outer container | 4 CPU ceiling, 9 GiB memory ceiling, 256 PID ceiling |
| Guest disk | Sparse 32 GiB qcow2 overlay; preserve its backing image |
| Guest hugepages | `HUGEPAGES=1024`, previously configured only inside the guest |
| Outer logs | `json-file`, 5 MiB × 2 files |
| Initial Arkos slots | 2 total across users; 2 per user; 2 unattended sessions per user |

Two slots are an initial upper limit, not a throughput claim. Inspect the pinned base template's actual memory/CPU requirements and measure peak usage. If two boxes do not fit with guest service headroom, start with one and lower all relevant quotas consistently. Do not enlarge the shared-host allocation silently. The outer memory ceiling is a final resource boundary, not a substitute for admission control.

A sparse virtual disk's nominal capacity is not its current host allocation. Record both. Do not reduce RAM, hugepages, or disk sizes purely to make a number look smaller without rerunning lifecycle and persistence checks.

## 4. Scope and invariants

### Required outcomes

1. Every E2B operation for this deployment uses the local endpoint pair and the local team key. Missing local configuration fails clearly without falling back to E2B Cloud.
2. `base` is passed explicitly as the selected template.
3. Session slots, user ownership, folder leases, pause/reconnect, timeouts, and cleanup retain their existing behavior.
4. Files acknowledged as saved by Arkos survive destruction of the sandbox and appear byte-identically in a fresh sandbox.
5. A failed scan, missing blob, failed extract, unavailable provider, or invalid workspace sentinel cannot replace durable files with an empty or partial tree.
6. The shared host retains working SSH, routes, existing inference, Supabase, and unrelated services.
7. Another operator can understand start, status, logs, graceful stop, restart recovery, backup, restore, and rollback from the delivered instructions.

### Preserve these application contracts

- `run_command`, `read_file`, `write_file`, `edit_file`, `grep`, and `glob` keep their schemas and session scoping.
- The synchronous SDK stays off the asyncio event loop, using the existing thread wrappers.
- No sandbox starts during module import, manifest construction, or a read-only browse/peek of a sleeping session.
- One session owns one persistent sandbox handle; sessions do not share a working directory or sandbox identity.
- `session_sandboxes` remains the slot/handle table. `workspace_nonce` remains the proof that a sandbox contains the workspace expected by a flush.
- Files under `/home/user/store/<folder>` are durable only after a successful flush. Scratch files elsewhere are not a durable-storage guarantee.
- The runner flushes before pausing/reaping or releasing folder leases. Flush failure preserves recovery information and must not be reported as successful persistence.
- User-defined folder claims, including read-only claims and subpaths, keep their current meaning.
- Sandbox processes receive no Arkos, Supabase, model-provider, connector, or E2B API credentials.

### Out of scope

- Replacing E2B with another provider, a new sandbox abstraction framework, Kubernetes, a scheduler service, or a multi-node platform.
- Rebuilding Supabase, moving its data, changing Auth semantics, setting up external email delivery, or completing unrelated password-recovery work from the earlier deployment attempt.
- Changing the model, restarting inference, changing tool approvals, enabling connectors, or installing browser tooling.
- Public endpoint exposure, new DNS/TLS infrastructure, broad firewall changes, host package upgrades, or host reboot testing.
- Live migration of Firecracker memory/disk snapshots from hosted E2B. Transfer durable application files through Supabase instead.

## 5. Required application changes

Keep SDK-specific code in `tool_module/sandbox/manager.py`. Prefer a few private helpers over a provider hierarchy. Do not rename the public manager interface to implement an endpoint change.

### M1. Explicit deployment selection and endpoint validation

Add one literal configuration field: `sandbox.deployment`, accepting `hosted` or `local`. Keep `hosted` as the default for existing installations; set the server's deployment to `local`. Document it next to the existing sandbox settings.

Use the existing environment contract for the SDK values:

```dotenv
# Endpoint values for ark.mit.edu; fill the key privately on that server.
E2B_API_URL=http://127.0.0.1:13000
E2B_SANDBOX_URL=http://127.0.0.1:13002
E2B_API_KEY=
```

The empty key above is an example, not a usable configuration. Preserve the installation's generated key. Extend `.env.example` with descriptions of both URLs; do not put secret values in YAML or an example file.

Implement local-mode validation before any provider operation, including create, reconnect, cached-handle renewal, pause, kill-by-ID, and cleanup sweeps that can contact E2B:

1. Require a nonempty local API key and both endpoint values.
2. Require valid absolute HTTP(S) URLs. For this host deployment, require loopback hosts (`127.0.0.1`, `localhost`, or `::1`), explicit ports, and no embedded credentials, query, or fragment. Normalize a trailing slash consistently. Prefer the exact IPv4 values above in deployment files.
3. Refuse unknown deployment values and conflicting SDK debug/access-token settings. Do not enable SDK debug mode as a replacement for self-hosted endpoint configuration.
4. Use one source of connection settings for all SDK entry points. The SDK already reads these environment variables; a small validation helper plus the pinned SDK's native environment behavior is sufficient if tests prove every path. If explicit keyword arguments are used, inspect the installed version's accepted arguments first and cover static `Sandbox.kill(id)` as well as instance methods.
5. A validation error must name the missing/invalid setting without exposing its value when it might contain a secret.
6. An invalid or unreachable local configuration must never cause a request to a hosted default. Test at the transport/SDK boundary, not just by asserting that an environment variable is present.

Validate the endpoint/credential selection before a cleanup path deletes a handle it would need to kill or recover. In particular, a suppressed startup-sweep exception must not silently discard IDs because configuration was invalid. Emit a sanitized diagnostic when sandbox cleanup cannot proceed; keep ordinary Auth/files operations available.

Do not add mandatory `${E2B_API_URL}` or `${E2B_SANDBOX_URL}` interpolation to the whole YAML file. `ConfigLoader` throws on an unset interpolation, including for code that never uses E2B. Preserve lazy SDK import and useful non-sandbox application behavior when E2B is intentionally stopped. A local configuration validation failure should refuse sandbox work, not silently disable unrelated file/Auth operations.

The existing loader uses `load_dotenv(..., override=False)`: inherited environment values win over `.env`. Check the effective environment of the actual tmux-launched API process and remove stale hosted E2B variables from that launch context. Keep loopback addresses in `NO_PROXY` if the process inherits HTTP proxy settings. Never print the full process environment to prove this.

The upstream Python SDK's [connection configuration source](https://github.com/e2b-dev/E2B/blob/main/packages/python-sdk/e2b/connection_config.py) confirms that both endpoint environment variables exist. Its `main` branch is reference material; inspect the pinned installed SDK before choosing signatures or exception classes.

### M2. Preserve the configured template exactly

The workstation checkout currently contains this behavior in `_template()`:

```python
return name if name and name != "base" else None
```

That discards the explicitly selected `base` template. Change the behavior to return any nonempty configured name, including `base`; only an absent/empty name selects the SDK default in hosted mode. Local mode requires a nonempty template and defaults operationally to `base`.

This fix already existed on the server in command 51. Inspect the current diff rather than applying the same textual replacement twice. Add a focused manager/tool regression asserting that `Sandbox.create` receives `template="base"`, timeout, and session metadata, and no application credential environment. Also cover an arbitrary template name and hosted-mode omission.

### M3. Protect durable files when local infrastructure fails

Port and review the two server fixes in command 51:

**Missing blob:** in `workspace.materialize`, if a tree entry references a blob that cannot be read, raise `store.StoreError`. Do not skip the file and seal a partial workspace. Resolve required blob payloads before transferring or sealing the new workspace. The durable tree remains unchanged.

**Failed scan:** `_sweep` currently uses `find ... 2>/dev/null || true`, which can turn a failed filesystem scan into an apparently empty tree. Create missing claimed directories deliberately, execute the scan without masking failure, and check the exit code before parsing output. A suitable command structure is:

```text
mkdir -p <shell-quoted claimed paths> && find <same paths> -type f -exec sha256sum {} +
```

This is a structure, not a literal command to paste. Use the existing `shlex.quote` handling. On a nonzero scan result, raise `StoreError` with bounded, sanitized diagnostic text. Partial stdout plus an error is still failure. An intentionally emptied folder after a successful scan must continue to commit its legitimate deletions.

Keep `_verify_seal` before flush, and keep the runner's existing flush-before-reap ordering. Update the old test named `test_a_tree_row_whose_blob_is_gone_skips_that_file_rather_than_guessing` to assert refusal of partial materialization. Add a regression proving that a failed scan leaves both existing tree rows and blob bytes unchanged. Preserve tests for missing/wrong sentinels, failed tar extraction, read-only claims, and legitimate deletion of every file.

### M4. Distinguish a missing sandbox from a provider outage

`get_or_create`, `_connect`, and `_resume` currently catch broad SDK exceptions and may create a replacement box. A timeout, bad key, refused proxy connection, or server error is not proof that the old box was destroyed.

Use the pinned SDK's verified not-found/expired-sandbox exception or status to permit recreation. Propagate or translate other failures into a bounded provider-unavailable result while retaining the stored handle and recovery information. Do not introduce unbounded retries. Do not interpret a template-not-found error as a vanished existing sandbox.

For shell execution, preserve ordinary nonzero command exits and their stdout/stderr as command results. Transport/auth failures must remain distinguishable from a completed shell command; a transport timeout may leave execution outcome unknown and must not trigger an automatic replay of an arbitrary shell command.

Keep all operations on local endpoints during recovery. If the pinned SDK cannot distinguish a missing box reliably, fail conservatively and explain the limitation rather than guessing. Add fault-injection regressions for connection timeout, authentication failure, server error, and confirmed missing sandbox. Do not replace a healthy box just because one health call failed.

### M5. Bound aggregate capacity on the shared host

Existing `sandbox.max_concurrent_per_user` does not cap the total across users. Add `sandbox.max_concurrent_total` using the same slot table and transaction, without adding a separate service or queue.

- Default `0` means no additional aggregate limit for existing hosted installations. Require a positive total in local mode.
- Set the initial local total and per-user limits to 2, with `quotas.max_unattended_sessions: 2`; lower them together if preflight measurements require it. The current config coherence rule requires the per-user sandbox limit to be at least the unattended-session quota.
- In `claim_slot`, serialize the aggregate count and insertion using a documented, dedicated PostgreSQL transaction advisory lock before the existing per-user lock. Use the same lock ordering on every acquisition path.
- Reclaim eligible expired/terminal slots consistently, count reservations across users, and exclude the caller when renewing its own slot. Perform remote kills after the transaction, as in the current implementation.
- Keep `claim_slot`'s boolean interface. Update the runner's capacity message so an aggregate limit does not falsely blame only the current user's usage.
- Add real database concurrency tests: multiple users competing for the last slot, repeat acquisition by the same session, capacity returned after release, and stale-slot reclamation. Hosted-mode per-user behavior must still pass.

This is a cap on recorded reservations. Failed kills can leave provider orphans temporarily alive; document and observe those, preserve finite sandbox timeouts, and keep the outer resource ceilings. Do not claim that a database count alone proves the number of physical VMs.

### M6. Keep dependencies and documentation reproducible

Keep the current `e2b_code_interpreter.Sandbox` import for this migration. Removing it in favor of another SDK package is unnecessary scope unless a verified incompatibility demands it.

Use a small deployment constraints file for the verified E2B package pair, for example `deploy/e2b/constraints.txt`, rather than upgrading all application dependencies. Recheck the recorded baseline against the retained runtime. Record exact package versions, runtime commit, image digests, guest image checksum, and relevant configuration. Do not deploy floating `latest` images or fetch `main` as an unrecorded installation input.

Update the README and config comments that currently say the MIT profile must use a hosted sandbox. Explain both supported endpoint modes, the local deployment's stopped/running state, and its resource limit. Keep credentials in environment files and endpoint/mode policy documented in one place.

## 6. Deployment deliverables

Keep this small. Suggested new repository artifacts, created by the implementing agent on the server and delivered as a reviewable patch:

```text
deploy/e2b/
    README.md             # short host/guest runbook, pins, recovery and stop behavior
    compose.yaml          # reviewed outer VM definition, no secrets
    Dockerfile            # outer QEMU runtime, only if reconciling the existing build
    cloud-init.yaml       # secret-free template for a rebuild, never applied to a live disk
    constraints.txt       # exact verified E2B SDK pair
scripts/
    verify_local_e2b.py    # finite, explicit smoke/integration entry point
```

Reuse existing files where that is clearer; do not create wrappers that merely obscure a one-line Docker command. Separate **resume an existing installation** from **rebuild from scratch**. Rendering a seed image or overwriting a qcow2 disk must never be a side effect of an ordinary start/status command.

Treat the repository's deployment files as the reviewed source and `/home/service/dev/arkos-local/e2b-vm` as the installed runtime directory. Show the exact copy/render step for each changed definition, preserving the installed `data/`, keys, known hosts, and private configuration. Keep all relative paths anchored to the installed project directory. Do not copy a directory wholesale over the running installation.

The verification script must have an explicit mode, timeout, cleanup behavior, safe database selection, and machine-readable pass/fail output. Define a small CLI with `--mode sdk|application`, `--timeout-seconds`, and `--state-file`; write a sanitized JSON report to stdout. `sdk` exercises provider lifecycle without model inference. `application` exercises section 8.3 through the existing API. Use the state file to resume/inspect an already-started check after interruption and identify resources for cleanup; require mode `0600`, and never overwrite a state file naming a live check. Both modes must clean their own resources in a normal completion/failure path where safe, and report residual IDs when cleanup fails. Keep useful recovery data if work might not have flushed.

The script must not install software, create missing production infrastructure, start E2B implicitly, print secrets, or silently skip mandatory checks. Unit/transport tests are sufficient for hosted-mode compatibility; do not create paid hosted sandboxes to prove V04.

Use a new database migration only if a demonstrated schema requirement arises. The endpoint switch, explicit template, local validation, and aggregate reservation cap can use the existing schema. Never edit already-applied migration files or rerun historical destructive migrations manually.

## 7. Ordered implementation runbook

Complete each phase's exit condition before the next dependent phase. Keep a decision record when observed state differs from this specification. Commands in this section are for the implementing agent after runtime work is authorized; none were executed while writing this file.

### Phase A — establish the current baseline

From the workstation, reuse an SSH connection:

```bash
ark_ssh() {
  ssh -o ControlMaster=auto \
      -o ControlPath=/tmp/arkos-setup-ssh-%C \
      -o ControlPersist=600 \
      -o BatchMode=yes \
      -o ConnectionAttempts=1 \
      -o ConnectTimeout=15 \
      -o StrictHostKeyChecking=yes \
      -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=3 \
      service@ark.mit.edu "$@"
}
ark_ssh 'hostname; uptime; id'
```

If this fails before authentication, record the failure and stop deployment mutations. Do not loop on new connections. Reuse existing SSH configuration and known hosts; do not suppress host-key verification.

On the server, inspect:

```bash
cd /home/service/dev/arkos
git status --short
git rev-parse HEAD
git diff --stat
tmux -L arkos list-sessions
tmux list-sessions
docker ps -a --format '{{.Names}}\t{{.Status}}\t{{.Ports}}'
docker stats --no-stream --format '{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}'
free -h
df -h /home/service/dev/arkos
ip -4 route show
ip -6 route show
ss -lnt
curl --fail --silent --show-error --max-time 10 http://127.0.0.1:1121/health
```

Also inspect applicable `AGENTS.md` files, the exact remote source diff, tmux pane commands, Compose project labels, selected container mounts/devices/limits/restart settings, the dedicated Docker networks, and the VM backing-file chain. Use selected `docker inspect --format` fields; full inspect output contains environment values. Read private configuration only in a process that reports presence/mode or selected nonsecret endpoints.

Record these facts before changing anything:

- Whether E2B is still stopped and whether the guest disk, base image, seed, SSH key, and known-hosts file are present.
- Whether any other operator changed the deployment or started a different E2B instance.
- Whether Arkos currently targets local or hosted E2B. The previous server was already partially configured for local E2B; do not assume it still uses cloud defaults just because the repository does.
- Current application sessions, slots, pending teardown, and whether any have unflushed work. Record counts and relevant owned IDs without exposing user content.
- Current Docker subnets and host/VPN routes. Reuse the existing dedicated network where sound. Do not choose a replacement subnet until non-overlap with management and routed networks is established.
- Effective credentials' presence and permissions; do not rotate them to make setup easier.
- Available host/guest resources, existing listener conflicts, and baseline health of inference and Supabase.
- SDK package versions, runtime pin, guest image checksum, and image digests. Use the existing lock file as evidence, not an instruction to install everything again.

Inspect `/dev/kvm` access and nested virtualization availability read-only. Do not load/unload host modules or change virtualization settings on this shared machine. If the prerequisite is no longer available, report the specific prerequisite rather than substituting an unsafe architecture.

**Exit condition:** current state, ownership, active work, credentials' presence, preserved edits, routes, and resource budget are recorded. The next mutation has a known target and recovery path.

### Phase B — preserve data and reconcile the source

1. Review server-only fixes against M2/M3 and the local source. Preserve the pre-existing `model_module/run.sh` change. Do not use a hard reset, a checkout of all files, or an `rsync --delete` deployment.
2. Capture a private pre-migration copy of the application configuration, relevant source diff, E2B Compose/Dockerfile/seed definitions, pin files, and container settings. Set restrictive permissions before writing backups. Keep secret backups on the server.
3. Obtain a consistent backup of the durable application data before cutover: PostgreSQL including the relevant Auth/storage metadata, Supabase file bytes, the `db-config` encryption-key volume, and private configuration. Inventory the actual named volumes using this Compose project's labels; do not assume a similarly named volume belongs to us.
4. Quiesce application writes for the backup/cutover interval. A PostgreSQL dump and a storage-volume archive taken at unrelated times are not proof of a recoverable file tree. Account for any other writers before declaring consistency.
5. Back up the stopped E2B VM's disk chain and configuration if changes could make rollback require them. The qcow2 overlay depends on its backing image; copying the overlay alone is insufficient. Preserve sparsity and check free disk before copying. Prefer a supported reflink/copy-on-write copy where available; do not manufacture a full 32 GiB allocation unnecessarily.
6. Record backup paths, sizes, checksums, permissions, and what each backup covers. Verify readability before proceeding. The restore drill is specified in section 10.
7. Implement M1–M6 on the server checkout, retaining a reviewable patch. Install only the needed runtime/test tooling into the existing venv with the verified constraints. Do not replace the host Python or install the unrelated browser dependency stack just to run these tests.

For a PostgreSQL dump, the command shape is `docker compose ... exec -T db pg_dump -U postgres -Fc -d postgres > <private dump path>` on the server. Resolve the actual project and database first; keep binary output out of the command log. Log its exit status, byte count, and checksum instead. Back up roles/grants required for the chosen restore procedure as well. Do not treat this example as the entire backup procedure.

**Exit condition:** source changes are reviewable, required backups exist, and no existing data or user edit has been discarded.

### Phase C — handle existing sandbox IDs before endpoint cutover

`session_sandboxes.sandbox_id` has no endpoint/provider namespace. A cloud sandbox ID must not be reconnected or killed against a different E2B installation. Equally, do not introduce hosted credentials to clean up IDs already proven to belong to this local installation.

Use an offline cutover rather than supporting two runtimes at once:

1. Identify each retained slot/ID's source installation from current config, deployment history, SDK metadata, and operator evidence. If ownership cannot be established, preserve it and resolve that uncertainty before deleting the row.
2. Prevent new sandbox tasks while draining current runs. Inspect the API and runner's actual lifecycle; do not invent a maintenance flag that does not exist. A controlled Arkos-only maintenance interval is acceptable after active work is drained. Keep Supabase and the shared model running.
3. While still using each old installation's valid configuration, complete/stop its sessions through normal lifecycle paths, flush claimed files, and verify their durable readback. A paused sandbox may contain scratch data; distinguish that from persisted files and record any data that would otherwise be lost.
4. Reap old sandboxes only after successful flush. Verify provider-side absence and release the corresponding slots/nonces through the manager's existing lifecycle.
5. Preserve user, session, project, file, Auth, and blob data. Do not truncate `session_sandboxes` as a shortcut. If an already-dead, verified-owned sandbox leaves a stale row, remove only the exact reviewed row after checking durable state and record why it was safe.
6. If the current deployment contains no hosted IDs, record that evidence and skip hosted cleanup. Do not request a cloud key or create a cloud account simply to perform this migration.
7. Keep the pre-cutover application config available for diagnosis, but make local mode explicit before any new sandbox work is admitted. Restart only Arkos when required to clear cached handles and inherited SDK settings.

No schema change is required for an offline cutover that proves old handles are drained. If live dual-runtime support becomes a new requirement, it needs a separate design with endpoint ownership persisted alongside IDs; do not quietly implement it here.

**Exit condition:** every remaining ID belongs to the selected local installation, or there are no retained IDs; all acknowledged durable files are preserved.

### Phase D — reconcile and start the retained local VM

Review the outer definition against these requirements:

- Compose project stays `arkos-e2b-vm`, service stays `vm`, and only the existing dedicated data directory is mounted.
- QEMU uses the intended guest disk and backing image, KVM acceleration, the reviewed CPU/RAM limits, and user-mode networking.
- The only passed-through host device is `/dev/kvm`; the outer container is not privileged and does not use host PID/network namespaces or mount the host root/Docker socket.
- Guest SSH and the three E2B ports are bound only to host loopback.
- Preserve the existing read-only outer root, dropped capabilities, `no-new-privileges`, bounded tmpfs, PID limit, and bounded logs unless a verified issue requires a documented adjustment.
- Inspect guest log limits too: an outer Docker log cap does not cap Docker logs inside the guest. Bound the E2B guest containers' logs in a small reviewed override if the retained pin does not already do so. Do not add a new telemetry stack to solve this.
- Make the **outer** VM's default restart policy explicitly `"no"` in the reviewed deployment definition for this handoff. Start/recovery is manual and documented; an intentional stop stays stopped across Docker/host restarts. Do not change all upstream guest services' restart policies indiscriminately.
- Do not rebuild a healthy existing VM, regenerate its seed identity, or overwrite its secrets. Cloud-init templates are for reconstruction, not for resetting a retained disk.

Validate the outer definition with `docker compose --project-directory /home/service/dev/arkos-local/e2b-vm config --quiet`. A successful parse is not runtime verification.

Use the dedicated tmux socket. List panes and inspect their current commands first; send a start command only into the intended idle shell. If the session does not exist, create it with the correct working directory. The intended foreground command in `arkos-e2b` is:

```bash
cd /home/service/dev/arkos-local/e2b-vm
docker compose up vm
```

Do not paste this into a pane already running QEMU/Compose, and do not create a second VM because the first one is slow to boot. Wait on the same process and inspect its status. Detach with `Ctrl-b`, then `d`.

Connect to the guest from the host:

```bash
cd /home/service/dev/arkos-local/e2b-vm
ssh -i ssh_key -o IdentitiesOnly=yes -o BatchMode=yes \
    -o UserKnownHostsFile=known_hosts -o StrictHostKeyChecking=yes \
    -p 12222 arkos@127.0.0.1
```

Inside the guest, verify the expected hostname `arkos-e2b` before these commands:

```bash
cd /opt/e2b
sudo docker compose ps --all
sudo docker compose up -d --wait --wait-timeout 900
```

These are **guest commands**. E2B Embed performs privileged setup of the machine on which it runs; that machine must be the guest. Reconcile the guest once after every guest boot. Follow the retained pin's instructions for recreating volatile namespace/firewall state. A detached `up` exit or a healthy API alone is not enough: verify the `base` template by creating a real sandbox through the local proxy.

A long-running `up` may take minutes. Observe its existing process/container state in bounded intervals, report progress, and record final completion. A tool polling timeout does not authorize rerunning initialization.

**Exit condition:** the retained VM and intended guest stack are healthy, the base template creates a sandbox, and host routes/listeners/resources remain within the recorded constraints.

### Phase E — install local credentials and activate the application configuration

Read the generated SDK environment from the guest through a server-side process that captures stdout privately. The old deployment used `/run/e2b/sdk.env` in the `ready` service. Confirm the retained pin's location. Do not run `docker compose logs ready` into a shared transcript: it prints the team key.

Parse the returned values as data, not with `eval` of remote output. Preserve the generated team key. Write only the required E2B fields to the existing application `.env`, atomically and with mode `0600`, preserving all Supabase/Auth/model settings. Override the guest-local URL values with the **host** addresses `127.0.0.1:13000` and `127.0.0.1:13002`; the guest's own `localhost:3000/3002` addresses are not the host endpoints.

Set the existing server YAML's sandbox fields to:

```yaml
sandbox:
  deployment: local
  template: base
  timeout_seconds: 300
  max_concurrent_per_user: 2
  max_concurrent_total: 2
  slot_ttl_s: 900
  browse_max_bytes: 1048576

quotas:
  max_unattended_sessions: 2
  # Preserve all other existing quota keys.
```

Merge these keys into the actual YAML. Do not replace its whole `quotas` block with this partial example. Lower the three capacity values together if the resource gate selected one slot.

Restart only the owned Arkos process in `tmux -L arkos` after active work is drained. Its recorded foreground command was:

```bash
cd /home/service/dev/arkos
.venv/bin/python -m uvicorn harness_module.api:app --host 127.0.0.1 --port 1121
```

Verify the actual command and environment before using it. Keep one API worker for this deployment unless a separately verified multi-worker design is introduced; its manager caches and runner state are process-local.

`GET /health` currently verifies the application/database, not a complete E2B operation. Keep its meaning clear. Use the dedicated verification script for E2B readiness instead of making every ordinary health request boot a sandbox.

**Exit condition:** the running Arkos process is explicitly in local mode, both E2B paths are local, normal auth/files still work, and invalid local settings cannot fall back to the cloud.

## 8. Verification plan

### 8.1 Test environments and commands

Run tests on the server. Use a dedicated disposable database such as the previously created `arkos_setup_test`. Unit fixtures truncate tables. `tests/conftest.py` protects against a DSN inherited only from `.env`, but an explicitly exported production `DB_URL` still wins: **never export the production DSN into pytest**.

Prepare a server-only test launcher that:

1. Reads credentials privately and constructs a DSN for the explicitly named test database.
2. Verifies `SELECT current_database()` matches that disposable name before migrations or pytest run.
3. Sets the fixed test Auth signing secrets expected by the tests and a nonempty dummy `OPENAI_API_KEY`. Do not mutate the deployed `.env` to make unit tests pass.
4. Uses only loopback E2B/Supabase endpoints for live provider tests and a separate test blob prefix or private bucket namespace.
5. Fails if a required database/provider is absent. A skip is not a migration pass.

Unit fixtures should isolate deployment/quota settings explicitly so a production YAML change does not alter unrelated test assumptions. Live tests must still use local mode and the intended tested resource limits. Changes to test configuration must not remove the new endpoint or aggregate-limit coverage.

After preparing that environment, the existing gates are:

```bash
cd /home/service/dev/arkos
.venv/bin/python -m pip check
.venv/bin/python db/migrate.py
.venv/bin/python -m pytest -q tests/ -m 'not integration' --timeout=120
.venv/bin/python -m pytest -q tests/test_store_integration.py tests/test_sandbox_integration.py --timeout=120
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy --follow-imports=silent --ignore-missing-imports --disable-error-code=arg-type tool_module/tools/ tool_module/browser/tool.py tool_module/sandbox/tools.py
```

The migration command above must run in the same verified **test** environment as pytest, not in an ordinary production shell. Use the exact Ruff/Mypy pins from `requirements-dev.txt`; the recorded server Ruff version differed from the repository pin. If a baseline unrelated failure exists, identify it with evidence rather than weakening checks or calling it a pass.

### 8.2 Required behavioral coverage

| ID | Scenario | Required evidence |
| --- | --- | --- |
| V01 | Configured `base` and arbitrary template | SDK create receives the exact configured template. |
| V02 | Valid local environment | Create, reconnect, timeout renewal, pause, file I/O, instance kill, and kill-by-ID reach only the selected local installation. |
| V03 | Missing key or either URL; malformed/nonlocal URL; conflicting debug settings | No SDK/network operation occurs; useful sanitized error identifies the setting. |
| V04 | Hosted compatibility | Existing hosted-mode behavior and optional SDK import remain usable; selecting local mode is deliberate. |
| V05 | Lazy operation | Import, tool manifest, browse, and peek of an unheld box do not create or wake a sandbox. |
| V06 | Capacity races | Per-user and aggregate limits hold across concurrent transactions; renewal does not double-count; released/expired slots become available. |
| V07 | Real commands and files | Text and binary bytes round-trip; directory results retain shape; ordinary failing shell command returns its actual stdout/stderr/exit code. |
| V08 | Reconnect after forgetting the handle | Stored ID reconnects to the same real sandbox and retains a marker file. |
| V09 | Pause/resume | A real pause followed by reconnect retains the box ID and marker; pause failure is not reported as verified hibernation. |
| V10 | Kill and orphan paths | Reap with and without a cached handle destroys the correct local box and frees its slot; kill failure is observable without a cloud fallback. |
| V11 | Provider outage | Timeout/401/403/5xx do not create a replacement box or erase a valid ID. Confirmed missing-box behavior is tested separately. |
| V12 | Missing blob | Materialization raises before sealing a partial tree; existing durable entries remain intact. |
| V13 | Failed or partial scan | Nonzero result, including partial stdout, cannot commit an empty/partial durable tree. |
| V14 | Sentinel and tar errors | Replacement box, wrong nonce/tree, or failed extraction/archive cannot falsely complete a flush. |
| V15 | Claims and deletes | Read-only edits do not persist; subpaths remain scoped; legitimate all-file deletion still works; resumed boxes do not resurrect stale files. |
| V16 | Credential isolation | Application/provider secret names and actual values are absent from sandbox environment and transferred archives. Assertions never print secret values. |
| V17 | Full application flow | Real Arkos auth/upload/session/tool execution/persistence succeeds against local E2B and local Supabase. |
| V18 | Fresh sandbox reconstruction | Destroy the first sandbox; a second sandbox reads the saved output byte-identically and can save another change. |
| V19 | Process/guest recovery | Controlled Arkos restart and guest stop/start preserve saved files and documented recovery works. |
| V20 | Backup/restore | A private backup restores into an isolated target and reproduces selected tree metadata and blob hashes. |
| V21 | Shared-host impact | Routes, loopback bindings, resource limits, Supabase, inference, and SSH remain sound after startup and recovery. |

Extend existing behavioral tests where their fixtures fit. A small `tests/test_sandbox_config.py` is appropriate for configuration and transport guards. Do not add tests that only assert the exact shell string used by the implementation; test the resulting data-preservation behavior.

Existing real SDK tests can prove individual provider operations but do not cover the full user-visible flow. Extend the credential-isolation list to include the Supabase legacy and modern secret fields in addition to its current keys. Avoid dumping `env` into the command record; inspect it inside the test and report only the assertion outcome.

### 8.3 Full application acceptance procedure

Use a unique temporary account, project/folder prefix, and random content marker. The old mode-0600 verification file may be reused only after confirming its ownership and actual object state. Never echo its contents. Refresh its expired login through Supabase instead of replaying old cookies blindly.

1. Authenticate through the actual local Supabase Auth endpoint. For a new test account, require email confirmation through the existing local Mailpit inbox. Do not fake an Arkos cookie or weaken Auth.
2. Exchange the Supabase bearer at `POST /auth/session`; verify `GET /auth/me` identifies that test account.
3. Upload `migration-<unique>/input.txt` through `POST /files`. Read it back through the Files API and compare exact bytes/hash.
4. Create a project linked to that folder through `POST /projects`.
5. Create an ordinary Arkos session through `POST /sessions`, asking the existing local model to read the input in its sandbox and write a deterministic output under `/home/user/store/migration-<unique>/output.txt`. Use no connectors or external services. For example, request copying the input and appending a unique marker with `run_command`.
6. Observe session events and status. If normal approval is required, inspect the exact requested test action and answer it through the ordinary approval endpoint within the already authorized test scope. Do not change the application's approval policy.
7. Verify that a real E2B command ran, the session's workspace flushed, and the Files API returns the exact expected output. A model saying it finished, a terminal session state, or an output existing only inside the sandbox is insufficient.
8. Reap/destroy the first sandbox through its normal lifecycle and verify provider absence. Start a distinct session with a distinct sandbox ID linked to the same folder. It must read the saved output correctly, write another deterministic file, flush it, and expose it through the Files API.
9. Separately verify pause/resume using the same box ID, and confirm read-only live browsing does not create an absent box.
10. Record session/sandbox IDs, endpoint mode, expected/actual hashes, status, and cleanup results without recording credentials or user content beyond the synthetic marker.

Give model execution a finite deadline. On an observation timeout, inspect the existing session instead of creating duplicates. If the local model cannot make a valid tool call, record the model/tool-call failure; deterministic manager tests may isolate the infrastructure, but they do not substitute for V17.

### 8.4 Failure verification without disrupting shared services

Use SDK/transport injection for missing endpoints, denied auth, timeout, 5xx, and partial scans. Use a temporary verifier process with an intentionally closed loopback port when a real connection-failure check is needed. Do not change the live deployment's credential file or take down Supabase to simulate failure.

Run disruptive guest recovery checks only after test jobs are drained and no other active session depends on that VM. Do not fill a disk, exhaust RAM, reboot the host, or kill arbitrary QEMU/Firecracker processes as a test. Assert the outcome of an injected failure against real durable test data where practical.

## 9. Maintenance and restart contract

The completed short runbook must include these operator tasks with their tested commands:

| Task | Required behavior |
| --- | --- |
| Status | Show owned tmux sessions, outer VM state, selected guest service health, and an explicit small E2B smoke check. |
| Debug | Attach separately to `arkos-e2b`, `arkos-supabase`, and `arkos-api` on tmux socket `arkos`; explain detach keys. |
| Logs | Show selected `api`, `orchestrator`, and `client-proxy` logs, avoiding `ready` credential output and full environment dumps. |
| Start | Reuse the retained disk, start one outer VM in its tmux session, reconcile guest Compose, and verify a base sandbox. |
| Graceful stop | Drain and flush application work, disable outer auto-restart, power off the guest, verify outer exit, and retain disks/configuration. |
| Guest restart | Repeat the reconciliation required after boot, then verify local endpoints and saved-file reconstruction. |
| Host restart recovery | Document manual recreation of missing tmux sessions and the same VM/API start order. Do not reboot the shared host to test it. |
| Upgrade | Pin new artifacts, back up, drain, review changes, and verify before discarding rollback data. |

For graceful stop, use the already demonstrated command-76 procedure: verify container ownership; `docker update --restart=no` for that exact outer VM; request `sudo shutdown -h now` **inside the guest**; wait until the outer container has exited; then record its stopped state. A disconnect from guest SSH is expected during poweroff. A slow poweroff is not permission to force-kill it or delete a stale-looking disk.

Sending Ctrl-c/SIGTERM to QEMU is not a substitute for a verified guest shutdown. Do not describe `stop_grace_period` alone as proof that the guest databases flushed cleanly.

Test a controlled guest stop/start, not a shared-host reboot. Verify the generated E2B credential is unchanged, the base template is usable, old stale handles are handled safely, and saved files reopen in a new sandbox. Test an Arkos-only process restart as well. After these checks, leave the deployment in the state authorized by the implementation request and record it explicitly; if no running-state instruction supersedes the earlier stop, leave E2B stopped.

## 10. Backup, restore, rollback, and cleanup

### Backup/restore acceptance

Keep the durable Supabase data, its file bytes, and needed configuration as one documented recovery set. Preserve the E2B guest's own database/seed state if the runbook promises identity/key/template continuity; a cold backup of its complete disk chain is a simple option for this small deployment.

Restore into a separate disposable database/storage namespace and, if validating a VM image, an isolated VM with different loopback ports. Never restore over live Supabase or boot a second E2B instance on its production ports. Keep any restored VM within the same shared-host resource budget by stopping the original first where necessary.

Verify actual recovery, not just `pg_restore --list`: selected application rows, file hashes, private bucket access, and retrieval of the synthetic saved output must work from the restored set. Show that the E2B files needed to reconstruct the environment are present and readable. Document any explicitly untested recovery component. Remove only the temporary restore target after recording the result.

Keep a small, measured backup footprint and an explicit retention decision. Do not silently create repeated whole-disk backups or delete the only recovery copy to save space.

### Rollback triggers and procedure

Rollback or hold cutover if local endpoint routing is unproven, data preservation fails, host resources/routes deteriorate, credential isolation fails, or lifecycle/recovery tests cannot pass.

1. Stop admitting new sandbox work; inspect and drain existing work with its current local configuration.
2. Preserve any newly saved Supabase data. Do not restore an older application database merely to undo an endpoint/code change.
3. Retain potentially unflushed boxes and their handles for recovery; do not clear them to make the dashboard appear clean.
4. Revert only the reviewed migration patch/configuration using the captured prior versions, preserving other edits and credentials.
5. Gracefully stop the dedicated E2B guest if required. Leave Supabase and inference intact.
6. The safe fallback is an explicitly unavailable/stopped sandbox capability. A return to hosted E2B requires the user's authorization and the same ID-drain discipline; never switch to paid cloud silently.
7. Restore from backups only for actual data/configuration recovery, with a reviewed target and consistency check. Document all remaining limitations.

No rollback command may use `docker compose down -v`, global Docker prune, a host reboot, or broad database deletes.

### Scoped cleanup

- Reap only sandboxes created by verification, identified by recorded IDs and session metadata; check paused as well as running boxes.
- Remove only the temporary test accounts, projects, sessions, mails, and private verification-state files whose ownership is established.
- Clean test blob prefixes carefully. Production content-addressed blobs may be shared by multiple tree entries; do not delete a blob merely because a test file referenced it. If reference safety is not proven, leave it and document it.
- Retain the required logs/evidence and chosen rollback backup. Remove only known temporary artifacts and restore targets, after active handles are terminal.
- Never delete the shared inference image, unrelated volumes, or caches simply because they are large.

## 11. Completion checklist and handoff

Report **implementation complete** only when all applicable items below have current evidence. Historical green tests and an installation script exiting zero do not satisfy the full scope.

- [ ] Current remote state and pre-existing edits were inspected and recorded.
- [ ] E2B runtime/SDK/image/guest pins and resource envelope are recorded and reproducible.
- [ ] Local mode validates both endpoint paths and has no cloud fallback.
- [ ] M1–M6 are implemented with small reviewed changes and relevant regression coverage.
- [ ] Old IDs are drained or proven to belong to the selected local installation; durable data is retained.
- [ ] V01–V21 have actual results, including the real agent workflow and fresh-sandbox saved-file recovery.
- [ ] Required static checks and regression/provider tests passed against the correct isolated environments; skips are explained and do not replace required live tests.
- [ ] Separate tmux debugging sessions and graceful lifecycle commands are verified.
- [ ] Backup/restore and controlled guest/API restart recovery are demonstrated within scope.
- [ ] No broad host mutation, unrelated restart, credential disclosure, or unreviewed increase in resource use occurred.
- [ ] Temporary verification resources are cleaned up or specifically accounted for.
- [ ] `ARK_COMMANDS.md` records every remote command and its outcome; `ARK_SETUP.md` reflects the final authorized service state.
- [ ] A reviewable source/deployment patch, exact install constraints, and a concise results table are delivered.

The final implementation report should state: what changed; which source/deployment versions are running; whether E2B is running or intentionally stopped; where to attach/debug; evidence for file persistence and recovery; measured resource use; and any remaining failure or unverified requirement. If a required check cannot be performed, report the migration as incomplete rather than redefining completion around the tests that passed.

### Command-record entry format

Use this after each remote command, continuing the existing record's numbering:

```text
Number and UTC timestamp:
Context / desired outcome:
Exact command attempted (no embedded secrets):
Sanitized stdout/stderr and exit code, or current process handle:
Actual changes:
Verification / whether the desired outcome occurred:
Next decision and why:
```

For long-running commands, add later output and final exit status to the same numbered operation. For credentials and binary backups, record presence, permissions, sizes/checksums, and success/failure rather than their contents. Never invent an executed command from an intended action.

## 12. Upstream references and version caveats

The repository and deployment record are the sources for Arkos-specific requirements. These upstream links explain E2B behavior and should be rechecked against the chosen pin:

- [E2B Embed Compose guide](https://github.com/e2b-dev/runtime/blob/main/embed/compose/README.md): the October 2 reference documents a dedicated mutable machine, nested virtualization, generated SDK settings, guest boot reconciliation, bounded logs, and pause/version limitations. It recommends 12 GiB RAM; the retained 8 GiB guest with reduced hugepages is a previously exercised constrained configuration that still needs capacity testing. Do not apply this guide's host-mutation commands directly to `ark.mit.edu`.
- [Previously recorded pinned guide](https://github.com/e2b-dev/runtime/blob/9dd5b727318831ebbdd84cc9f51b25c5f3af96c8/embed/compose/README.md): recorded by the earlier deployment. Retrieval of this exact page failed during specification preparation; confirm the cached source and pin on the server. That fetch failure does not establish that the pin is invalid.
- [Python SDK connection configuration](https://github.com/e2b-dev/E2B/blob/main/packages/python-sdk/e2b/connection_config.py): endpoint environment behavior; inspect the installed package for the exact versioned contract.
- [Docker bridge firewall behavior](https://docs.docker.com/engine/network/packet-filtering-firewalls/): reason to inspect the host's network state even when ports are bound to loopback.
- [Ubuntu UFW limit documentation](https://help.ubuntu.com/community/Gufw): context for avoiding bursts of new SSH connections; it is not a diagnosis of the previous sustained outage.

Do not combine a newer upstream Compose file, an older `.env` image set, and an arbitrary SDK release. Preserve a tested set or deliberately validate a new pinned set. Internet downloads during bootstrap are compatible with a local runtime; serving sandbox control and execution through E2B Cloud is not. Do not claim an air-gapped deployment unless network egress has actually been specified and verified.
