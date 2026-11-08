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
