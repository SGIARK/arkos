#!/usr/bin/env bash
# Deploy the landing page + reload Caddy. Run on the ark server (or via ssh).
# Assumes: repo checked out, Caddy installed, waitlist service set up (see README).
set -euo pipefail

REPO_LANDING="${REPO_LANDING:-$HOME/ark/landing}"       # where Landing.html lives
WEBROOT="${WEBROOT:-/var/www/ark-landing}"
STEPS="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> building site from $REPO_LANDING/Landing.html"
python3 "$STEPS/scripts/build-site.py" "$REPO_LANDING/Landing.html" /tmp/ark-index.html

echo "==> publishing to $WEBROOT"
sudo mkdir -p "$WEBROOT"
sudo install -m 0644 /tmp/ark-index.html "$WEBROOT/index.html"

echo "==> installing Caddyfile"
sudo install -m 0644 "$STEPS/caddy/Caddyfile" /etc/caddy/Caddyfile
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy

echo "==> done. verify: curl -I https://ark.mit.edu/landing"
