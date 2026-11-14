# Feature Spec: Selfhosted E2B and its integration into ArkOS

Status: draft. Author: Kshitij Duraphe. Last updated 2026-11-08.

## Summary

This is a spec for Arkos to give each session a private Linux machine (sandbox) where the agent runs commands and edits files using the E2B framework. Ordinarily in the cloud, E2B publishes a version of its server called Embed that runs on a single machine. The idea is to run it inside the server as a separate process. On arkos's side this takes only configuration.

## Problem

On ark today, every sandbox tool call fails. That covers `run_command` and the six file tools, so the agent can't run anything or open a file. The system prompt still tells the model it has its own computer, so it keeps trying. arkos only refuses a tool after three failures in a row, so one turn can waste up to twenty-one tool calls. This is done when a session on ark can run commands and edit files in a sandbox served from ark. The whole setup should be rebuildable from this repo and leave ark itself unchanged.

## Background

### How arkos uses sandboxes

A session gets its sandbox the first time the agent calls a sandbox tool. arkos copies the session's folders into it from our file store, and copies changes back when the session pauses or ends. Nothing in a sandbox is meant to last, so losing one costs at most the work that hasn't been copied back yet. The sandbox never receives passwords or keys.

All of this already works. `tool_module/sandbox/manager.py` is the only file that talks to E2B. The E2B library it uses reads three environment variables: an API key, the address of E2B's API, and the address of its sandbox proxy. With the two addresses unset, it uses E2B's cloud. With them set, it uses whatever server they name.

### What really is E2B's Embed?

Embed is E2B's server packaged as a set of Docker containers for one machine. We should keep a pinned copy in `e2b/` to use as reference. It brings its own databases, an API, the program that starts and stops sandboxes, a proxy that carries traffic to and from them, and a dashboard. The first time it starts, it creates the API key that arkos will use. It doesn't send usage data anywhere unless we turn that on.

Embed needs a lot of control over the machine it runs on. Some of its containers run with full root access to the host. Every time it starts, it loads kernel modules, sets aside a fixed block of memory for sandboxes, changes kernel settings and adds a firewall rule. It also opens thirteen ports, and one of them lets anyone who can reach it create and delete sandboxes without a password. E2B's own advice is to give it a machine or a VM of its own.

## Proposed approach

We'll run Embed in a VM on the server in this way: The VM's setup will live in this repo, so anyone can review or rebuild it. It will run the copy of Embed in `e2b/`, so the repo and the server agree on the version. ark gives the VM its virtualization hardware and nothing else. Everything Embed does as root happens inside the VM, so ark's own memory, kernel settings and firewall stay as they are. The VM's API and sandbox proxy are reachable only from ark itself, which is all arkos needs. Its SSH and the dashboard are reachable the same way, for whoever runs the server. The port that controls sandboxes without a password never leaves the VM.

New VM
CPUs	8
Memory	12 GiB
Memory set aside for sandboxes	4 GiB
Busy sandboxes it can hold	about 8
Disk	64 GiB

The disks are sparse, so they only take as much of ark's disk as they actually fill. i.e. there is no excess 'prepackaged stuff'.

On the arkos side, the new install's API key and the two local addresses go into arkos's `.env`.

Upgrading Embed, or rebuilding it, ends every sandbox alive at that moment. A paused sandbox loses only files outside the session's folders, because the folders were copied back when it paused. A running one also loses edits it hasn't copied back, so upgrades should wait until no session is running. If anything goes wrong, stopping the VM's container takes the sandbox server down and arkos carries on without sandboxes.

### Alternatives considered


1. Cloud costs: no

2. A VM inside the server is probably the best idea. Running Embed directly on the server would be simpler than a VM. It would also put kernel modules, a fixed memory reservation, a firewall rule and the password-free port on a server that other services depend on.

A machine of its own would isolate Embed better than a VM on ark. We don't have one. Nothing here stops us from moving to one later, because arkos only knows the sandbox server by its address.

### Decisions needed

1. Whether sandboxes may reach the internet. We propose yes, since the agent is told it can install packages. Their traffic would leave through ark's connection.
2. Whether to keep E2B's dashboard. We propose keeping it, reachable only from ark, for whoever runs the server.
3. Whether to cap sandboxes across all users. arkos limits each user to five but has no overall limit, so a busy day could fill the VM, and new sandboxes would then fail to start. We propose accepting that for now and watching for it.

### Out of scope

This plan doesn't cover a dedicated machine, custom sandbox templates, limits on sandbox internet access, an overall sandbox cap or E2B's built-in metrics.

The launch documents on the `vin/launch-docs` branch list e2b as an outside component that receives user files, in decisions D1 and D19 and in the readiness plan's data table. Those entries should come out once this ships.

## Implementation plan

| Task | Effort | Waits on |
|---|---|---|
| 1. Put the VM setup in the repo | about a day | nothing |
| 2. Start the new VM on ark | half a day | Task 1 |
| 3. Connect arkos to it | two hours | Task 2 |

### Task 1: Put the VM setup in the repo

This task puts the VM's setup under `e2b/vm/`, so it can be reviewed and rebuilt on any Linux machine with virtualization support.

The setup downloads Ubuntu 24.04's minimal cloud image and checks it against the checksum Ubuntu publishes. It creates the VM's disk on top of that image, the configuration the VM reads on first boot, and an SSH key for logging in. Then it starts QEMU inside a container. That container gets ark's virtualization device and nothing more. It runs as an ordinary user, with no Linux capabilities and no way to gain any. Its filesystem is read-only, and its CPU and memory limits sit just above the VM's own.

Inside the VM, the setup installs Docker and the tools Embed expects, and starts Embed every time the VM boots. Starting it at boot matters because a reboot drops two of Embed's settings, a firewall rule and a directory, and Embed recreates them only when it starts. The VM forwards its SSH, API, dashboard and sandbox proxy to ark's local ports 12222, 13000, 13001 and 13002, and forwards nothing else.

The new files are `e2b/vm/Dockerfile`, `e2b/vm/compose.yaml`, `e2b/vm/user-data.template` and `e2b/vm/create.sh`. The SSH key and the disk images are added to `.gitignore`. `e2b/README.md` gains short instructions for starting, stopping, rebooting, upgrading and removing the VM. Tests 1 to 4 check this task.

### Task 2: Start the new VM on ark

Build the new VM on ark from `e2b/vm/` and start Embed in it from the files in `e2b/`, with 4 GiB set aside for sandboxes. Set the VM's container to restart on its own. This task is done when E2B's own smoke test passes inside the VM and ark's memory, kernel modules, kernel settings and firewall are exactly as they were before. Tests 2 to 4 check this task.

### Task 3: Connect arkos to it

Put the new install's API key into arkos's `.env`, along with `E2B_API_URL=http://127.0.0.1:13000` and `E2B_SANDBOX_URL=http://127.0.0.1:13002`, then restart arkos. Inside the VM, running `docker compose exec ready cat /run/e2b/sdk.env` in `/opt/e2b` prints the key.

Then run arkos's live sandbox tests against the new server. After that, run one real agent task: upload a file, ask the agent to process it with `run_command`, and check that the result shows up in Files after the session ends.

Finally, add the two address variables to `.env.example`. The comment about the MIT setup in `config_module/config.yaml` and the README's "Running at MIT" section still describe the sandbox as paid cloud compute, so update both. Tests 5 to 7 check this task.

## Tests

These run on a real machine; record their results under Implementation notes.

| # | What it checks | How |
|---|---|---|
| 1 | ark gives the VM nothing but virtualization | `docker inspect` on the VM's container shows it isn't privileged, has no capabilities, can't gain new ones and has only `/dev/kvm` |
| 2 | Embed's changes stay inside the VM | After Embed starts, ark's hugepage count is unchanged, `/etc` has no new E2B files and there's no new firewall rule |
| 3 | Only the four forwarded ports are open on ark | `ss -ltn` shows 12222, 13000, 13001 and 13002 on `127.0.0.1` only, and port 5008 can't be reached from ark |
| 4 | Embed survives a VM reboot | Reboot the VM; Embed comes back on its own and the smoke test passes |
| 5 | arkos's sandbox code works with our server | `pytest -m integration tests/test_sandbox_integration.py` passes with the three E2B variables set |
| 6 | The whole path works | Upload a file, have the agent process it with `run_command`, and find the result in Files after the session ends |
| 7 | A paused session gets its sandbox back | Write a file outside `~/store`, have the session wait on a question, answer it, then read the file back |

## Open questions

1. Does the E2B library on ark work with the new Embed? If Test 5 fails, find the oldest version that passes and make it the minimum in `requirements.txt`.
2. Embed won't start more than three sandboxes at the same moment. What does an arkos user see when four or more sessions start sandboxes together? If it's a confusing failure, a retry belongs in a separate spec.

## Reference

Where things are in the code:

- `tool_module/sandbox/manager.py` talks to E2B, and `tool_module/sandbox/tools.py` defines the seven sandbox tools.
- `harness_module/workspace.py` copies folders in and out of sandboxes, and `harness_module/runner.py` decides when a sandbox starts, pauses and ends.
- Sandbox settings are under `sandbox:` in `config_module/config.yaml`, and the live tests are in `tests/test_sandbox_integration.py`.
- `e2b/` is Embed, copied unchanged from `e2b-dev/runtime` at commit `6dd5539`. E2B's guide to it is `embed/compose/README.md` in that repo.

ark is an x86-64 machine with an AMD CPU and nested virtualization enabled, running Linux 7.0. It has 32 CPUs, about 61 GiB of memory and 789 GiB of free disk.

Embed needs Linux with KVM and cgroup v2, kernel 6.8 or newer, Docker 27 or newer with Compose 2.24 or newer, and the `/dev/kvm` and `/dev/net/tun` devices. E2B recommends 12 GiB of memory and 20 GiB of free disk. Installing or upgrading needs internet access to E2B's image registries.

Embed reserves memory for sandboxes in 2 MiB pages. The `HUGEPAGES` setting controls how many, and its default of 2048 pages comes to 4 GiB. A standard sandbox uses up to 512 MiB of that while busy and about 180 MiB while idle, so the default pool holds about 8 busy sandboxes or 20 idle ones. Beyond that, new sandboxes fail with `500: Failed to place sandbox`.

## Implementation notes

Tested on ark on 2026-11-08 from commit `b5cb9b5`, copied to `/home/service/dev/arkos-local/e2b`. `create.sh` and Embed's first start took about three minutes together, most of it downloads.

| # | Result |
|---|---|
| 1 | Passed. The container isn't privileged, drops every capability and can't gain any, gets only `/dev/kvm`, has its own process namespace and network, has a read-only filesystem, and runs as `service`. |
| 2 | Passed. ark's hugepage count stayed at 0, `nbd` stayed unloaded and `/etc` gained no E2B files. The firewall wasn't checked, because that needs root on ark. |
| 3 | Passed. Only 12222, 13000, 13001 and 13002 were listening, all on `127.0.0.1`, and nothing on 5008. |
| 4 | Not run. The VM wasn't rebooted. |
| 5 | Passed after one rerun. On the first run 8 of 9 passed. The credentials test failed because ark's `OPENAI_API_KEY` is the placeholder `-`, which matches any hyphen in the sandbox's environment; the variable itself wasn't there. With a realistic stand-in key the test passed, so none of the five credentials it checks reaches a sandbox. |
| 6 | Not run. arkos's `.env` and the running arkos were left alone. |
| 7 | Not run as an agent task, but Test 5's pause and resume tests passed. |

E2B's own smoke test also passed inside the VM: a sandbox started, and its port 8080 was reached through the sandbox proxy.

While up and idle, the VM's container used 12.1 GiB of its 13 GiB memory limit and about a sixth of one CPU core, and the VM's disk held 8.1 GiB. E2B was then shut down: the VM powered off cleanly, its container is stopped with automatic restart turned off, and the four ports are closed.
