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
