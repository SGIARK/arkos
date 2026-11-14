# E2B Runbook

## About

We run arkos with self-hosted E2B. But by default, E2B needs to change system settings to work and ark is shared with other services, so E2B itself runs inside a VM. Only the model's sandbox commands go to E2B. Chat messages go to the model, as before.

## Locations relative to ark

All commands below run on ark.mit.edu as the `service` user.

- `/home/service/dev/arkos-local/e2b` holds E2B and its VM, copied from the `e2b/` folder of the repo. The VM's disk is 8 GB (chosen arbitrarily) and is in `vm/data/`, and the key for logging into the VM is `vm/ssh_key`.
- `/home/service/dev/arkos` is arkos itself. Its settings file is `.env`.
- arkos runs in the tmux session `arkos-served`, on the default tmux server. It is not in `arkos-api`.
- The VM's Docker container is called `arkos-e2b-vm-1`.
- E2B is reachable only from ark itself, on these ports: `13000` (API), `13001` (dashboard), `13002` (sandbox proxy) and `12222` (SSH into the VM).


## Logging into the VM

Most commands below need this helper. I recommend defining it once per terminal, from `/home/service/dev/arkos-local/e2b`:

```bash
cd /home/service/dev/arkos-local/e2b
vmssh() { ssh -i vm/ssh_key -o UserKnownHostsFile=vm/known_hosts -o StrictHostKeyChecking=accept-new -p 12222 arkos@127.0.0.1 "$@"; }
```

## Check if E2B is running or not

```bash
docker inspect -f '{{.State.Status}}' arkos-e2b-vm-1
vmssh systemctl is-active e2b
```

The first command prints `running` when the VM is up. The second prints `active` when E2B is ready inside it.

## Stopping E2B

Stopping E2B frees about 12 GB of ark's memory. While it's off, arkos keeps working, but the sandbox commands fail.

```bash
cd /home/service/dev/arkos-local/e2b
docker update --restart=no arkos-e2b-vm-1
vmssh sudo shutdown -h now
docker wait arkos-e2b-vm-1
docker compose -f vm/compose.yaml stop
```

This turns off automatic restart first, so Docker doesn't boot the VM straight back up. Then it shuts the VM down from inside, waits for it to power off, and stops the container.

## Starting E2B again

```bash
cd /home/service/dev/arkos-local/e2b
docker compose -f vm/compose.yaml up -d --force-recreate
until vmssh systemctl is-active e2b 2>/dev/null | grep -qx active; do sleep 15; done; echo ready
```

The first command also turns automatic restart back on. The VM keeps its disk, so E2B's data and API key are unchanged, and arkos needs no changes. The first few checks fail while the VM boots, which is normal. Expect a minute or two.

## Setting it up from scratch

You only need this if the VM is gone. Copy the repo's `e2b/` folder to ark, from a checkout on your own computer:

```bash
git archive --prefix=e2b/ HEAD:e2b | ssh service@ark.mit.edu 'tar -x -C /home/service/dev/arkos-local'
```

That fails if the folder already exists. Then, on ark:

```bash
/home/service/dev/arkos-local/e2b/vm/create.sh
```

The script downloads Ubuntu, creates the VM and starts it, and the VM then installs E2B. The first time took about three minutes. A new setup creates a new API key, so continue with the next section.

## Connecting arkos to E2B

arkos needs three lines in `/home/service/dev/arkos/.env`:

```
E2B_API_URL=http://127.0.0.1:13000
E2B_SANDBOX_URL=http://127.0.0.1:13002
E2B_API_KEY=<the key>
```

To see the key:

```bash
vmssh sudo docker compose --project-directory /opt/e2b exec -T ready cat /run/e2b/sdk.env
```

Copy the value after `export E2B_API_KEY=`. Ignore the addresses it prints: they're the VM's own, not the ones arkos uses.

Keep the key out of chats and documents. Only `.env` needs it.

arkos reads `.env` only when it starts, so restart it after any change.

## Restarting arkos

First check that nobody is in the middle of a conversation, because a restart ends it. Then restart arkos inside its own tmux pane:

```bash
cd /home/service/dev/arkos
tmux respawn-pane -k -t arkos-served:0.0 -c /home/service/dev/arkos "bash -c 'set -a; source .env; set +a; /home/service/miniconda3/envs/arkos/bin/python -m uvicorn harness_module.api:app --host 127.0.0.1 --port 1121 2>&1 | tee -a api.log'"
```

This stops arkos and starts it again in the same place, with the same command as before. Don't press Ctrl-C in that pane: arkos is the pane's only program, so the pane may close.

Then check it came back:

```bash
curl http://127.0.0.1:1121/health
```

It should print `{"status":"ok","database":"ok"}`.

Don't start arkos from the `arkos-api` session as well. It fails with "address already in use", because the copy in `arkos-served` already holds port 1121.

## Testing that it works

In the app, start a new conversation in a project and ask: "Run `uname -a` and save the output to test.txt in this project's folder." It works if the chat shows a `run_command` call with real output, and `test.txt` appears in the project's files after the conversation ends.

To test only the key, without arkos:

```bash
cd /home/service/dev/arkos
KEY=$(grep '^E2B_API_KEY' .env | cut -d= -f2- | tr -d "'\"")
curl -s -o /dev/null -w '%{http_code}\n' -H "X-API-Key: $KEY" http://127.0.0.1:13000/sandboxes
```

`200` means E2B accepts the key. `401` means it doesn't.

## Known issues

1. The AI's command fails with `401: Unauthorized ... Invalid API key`:
- Most often arkos was not restarted after `.env` changed. Restart it as described above.
- Otherwise the key in `.env` is wrong. Run the key test above, and copy the key again if needed.
- Check that `.env` has only one uncommented `E2B_API_KEY` line.

2. The AI asks you for an E2B API key: don't give it one. It's usually reacting to an old error earlier in the same conversation. Decline the request and start a new conversation.

3. Sandbox commands fail with a connection error: E2B is probably stopped. Check whether it's running, and start it.

4. New sandboxes fail with `Failed to place sandbox`: the VM's memory for sandboxes is full. It holds about 8 sandboxes at once, and arkos has no overall limit across users. Wait for conversations to end, or stop old ones.

