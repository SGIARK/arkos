#!/usr/bin/env bash
# Builds the E2B VM from nothing and starts it. Run it on the host that will
# run the VM. Running it again keeps everything that already exists.
set -euo pipefail

VM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E2B_DIR="$(dirname "$VM_DIR")"
DATA="$VM_DIR/data"
IMAGE_URL="https://cloud-images.ubuntu.com/minimal/releases/noble/release"
IMAGE_NAME="ubuntu-24.04-minimal-cloudimg-amd64.img"
RUNNER_IMAGE="arkos-e2b-vm:ubuntu24.04"
DISK_SIZE="64G"

die() {
    echo "create.sh: $*" >&2
    exit 1
}

# Ubuntu's minimal cloud image, accepted only if it matches the checksum
# Ubuntu publishes beside it.
fetch_image() {
    local expected
    if [[ -f "$DATA/base.img" ]]; then
        return
    fi
    expected="$(curl -fsSL "$IMAGE_URL/SHA256SUMS" |
        awk -v name="$IMAGE_NAME" '{ file = $2; sub(/^\*/, "", file) } file == name { print $1 }')"
    [[ -n "$expected" ]] || die "Ubuntu publishes no checksum for $IMAGE_NAME"
    curl -fL --progress-bar -o "$DATA/base.img.partial" "$IMAGE_URL/$IMAGE_NAME"
    echo "$expected  $DATA/base.img.partial" | sha256sum -c --quiet - ||
        die "$IMAGE_NAME does not match its published checksum"
    mv "$DATA/base.img.partial" "$DATA/base.img"
}

# The key the operator logs into the VM with. It stays in this directory.
make_key() {
    if [[ -f "$VM_DIR/ssh_key" ]]; then
        return
    fi
    ssh-keygen -q -t ed25519 -N "" -C arkos-e2b-vm -f "$VM_DIR/ssh_key"
}

# Fill in the cloud-init template with the operator's key and Embed's two
# files from e2b/, so the guest runs exactly the version this repo pins.
render_seed() {
    local seed
    seed="$(<"$VM_DIR/user-data.template")"
    seed="${seed//@SSH_PUBLIC_KEY@/$(<"$VM_DIR/ssh_key.pub")}"
    seed="${seed//@COMPOSE_B64@/$(base64 -w0 "$E2B_DIR/compose.yaml")}"
    seed="${seed//@ENV_B64@/$(base64 -w0 "$E2B_DIR/.env.example")}"
    mkdir -p "$DATA/seed"
    printf '%s\n' "$seed" >"$DATA/seed/user-data"
    printf 'instance-id: arkos-e2b-001\nlocal-hostname: arkos-e2b\n' >"$DATA/seed/meta-data"
}

# Run one command in the runner image with no network, no capabilities and
# nothing mounted but the data directory.
in_runner() {
    docker run --rm --network none --cap-drop ALL --security-opt no-new-privileges:true \
        --user "$(id -u):$(id -g)" -v "$DATA:/data" "$RUNNER_IMAGE" "$@"
}

# The small read-only disk that cloud-init reads on the guest's first boot.
make_seed_iso() {
    if [[ -f "$DATA/seed.iso" ]]; then
        return
    fi
    render_seed
    in_runner genisoimage -quiet -output /data/seed.iso -volid cidata -joliet -rock \
        /data/seed/user-data /data/seed/meta-data
}
