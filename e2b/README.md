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

## Running it

`vm/` runs Embed inside a VM, so nothing it does as root touches the host. On
ark this directory lives at `/home/service/dev/arkos-local/e2b`. To put it
there from a checkout of this repo:

    git archive --prefix=e2b/ HEAD:e2b | ssh service@ark.mit.edu 'tar -x -C /home/service/dev/arkos-local'

The commands below run from that directory, and log into the VM with this
helper:

    vmssh() { ssh -i vm/ssh_key -o UserKnownHostsFile=vm/known_hosts -o StrictHostKeyChecking=accept-new -p 12222 arkos@127.0.0.1 "$@"; }

The first start builds the QEMU image, downloads Ubuntu, creates the VM's
disks and SSH key, and boots the VM, which then installs Docker and starts
Embed. On ark the whole thing took about three minutes.

    vm/create.sh

Embed is ready when this prints `active`:

    vmssh systemctl is-active e2b

To connect arkos, put the key this prints into arkos's `.env` as
`E2B_API_KEY`, along with `E2B_API_URL=http://127.0.0.1:13000` and
`E2B_SANDBOX_URL=http://127.0.0.1:13002`, then restart arkos.

    vmssh sudo docker compose --project-directory /opt/e2b exec -T ready cat /run/e2b/sdk.env

To stop it, turn off the container's restart policy so Docker doesn't boot
the VM straight back up, shut the VM down from inside, and stop the container
once it has powered off.

    docker update --restart=no arkos-e2b-vm-1
    vmssh sudo shutdown -h now
    docker wait arkos-e2b-vm-1
    docker compose -f vm/compose.yaml stop

To start it again, recreate the container, which also restores its restart
policy. The VM keeps its disk, so Embed's images, databases and API key are
still there.

    docker compose -f vm/compose.yaml up -d --force-recreate
