# Runs the E2B VM. QEMU is all it holds, and compose.yaml gives the container
# nothing from the host but /dev/kvm.
FROM ubuntu:24.04
RUN apt-get update \
 && apt-get install -y --no-install-recommends qemu-system-x86 qemu-utils genisoimage \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /data
