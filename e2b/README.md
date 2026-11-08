# E2B Embed

The self-hosted E2B stack as one Docker Compose project: the control-plane API
on :3000, the dashboard on :3001, the sandbox proxy on :3002, a Firecracker
orchestrator, and the `base` template. It serves the API E2B Cloud does.

Vendored unmodified from [e2b-dev/runtime](https://github.com/e2b-dev/runtime)
at `6dd55395ad530c72e7d433016c0a1c9cad7e78b9` (2026-11-08), under that
repository's Apache-2.0 license, kept here as `LICENSE`:

| here | upstream |
| --- | --- |
| `compose.yaml` | `embed/compose/compose.yaml` |
| `.env.example` | `embed/compose/.env`: the pinned image and artifact versions |
| `LICENSE` | `LICENSE` |

`.env` is renamed because this repo ignores `*.env`. Copy it to `.env` before
`docker compose up`, so local settings such as `TEAM_API_KEY` stay out of git,
as in `supabase/`. Upstream's guide to running the stack is
[embed/compose/README.md](https://github.com/e2b-dev/runtime/blob/6dd55395ad530c72e7d433016c0a1c9cad7e78b9/embed/compose/README.md)
at the same commit.

**`up` changes the machine it runs on.** Privileged containers in the host's PID
and network namespaces load kernel modules, write sysctls, reserve hugepages and
add an iptables rule. Thirteen ports bind every interface, including the
orchestrator's unauthenticated control API on 5008. Run it only on a dedicated
host or VM.

To upgrade, copy both files from a newer upstream commit and update the commit
above.
