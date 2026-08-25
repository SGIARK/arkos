#!/usr/bin/env bash
# Post-deploy checks. Safe to run against production (inserts one test email).
set -euo pipefail
BASE="${BASE:-https://ark.mit.edu}"

echo "== page loads =="
curl -fsS -o /dev/null -w "landing: %{http_code}\n" "$BASE/landing"

echo "== /app is down (expect 404) =="
curl -s -o /dev/null -w "/app: %{http_code}\n" "$BASE/app" || true

echo "== waitlist happy path (expect 204) =="
curl -s -o /dev/null -w "insert: %{http_code}\n" -X POST "$BASE/landing/api/waitlist" \
  -H 'Content-Type: application/json' \
  -d '{"email":"smoke+'"$(date +%s)"'@example.com"}'

echo "== duplicate is idempotent (expect 204, not an error) =="
curl -s -o /dev/null -w "dupe: %{http_code}\n" -X POST "$BASE/landing/api/waitlist" \
  -H 'Content-Type: application/json' -d '{"email":"dupe@example.com"}'
curl -s -o /dev/null -w "dupe2: %{http_code}\n" -X POST "$BASE/landing/api/waitlist" \
  -H 'Content-Type: application/json' -d '{"email":"dupe@example.com"}'

echo "== bad email rejected (expect 400) =="
curl -s -o /dev/null -w "bad: %{http_code}\n" -X POST "$BASE/landing/api/waitlist" \
  -H 'Content-Type: application/json' -d '{"email":"not-an-email"}'

echo "== honeypot silently accepted, nothing stored (expect 204) =="
curl -s -o /dev/null -w "honeypot: %{http_code}\n" -X POST "$BASE/landing/api/waitlist" \
  -H 'Content-Type: application/json' -d '{"email":"bot@example.com","company_website":"x"}'

echo "== the list is NOT publicly readable =="
curl -s -o /dev/null -w "GET waitlist: %{http_code} (want 404/405)\n" "$BASE/landing/api/waitlist" || true
