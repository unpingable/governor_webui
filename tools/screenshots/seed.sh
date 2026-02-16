#!/usr/bin/env bash
# seed.sh — Import deterministic state for screenshot generation.
# Run against a live gov-webui research stack (default: localhost:8003).
set -euo pipefail

BASE="${SCREENSHOT_BASE_URL:-http://127.0.0.1:8003}"
FIXTURE_DIR="$(dirname "$0")/fixtures"
AUTH="${GOVERNOR_AUTH_TOKEN:+Authorization: Bearer $GOVERNOR_AUTH_TOKEN}"

echo "==> Seeding research demo state into $BASE ..."

curl -sS -X POST "$BASE/governor/import" \
  -H "Content-Type: application/json" \
  ${AUTH:+-H "$AUTH"} \
  --data-binary "@$FIXTURE_DIR/research_demo.json" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'    Imported: {d[\"imported\"]}, Skipped: {d[\"skipped\"]}')"

echo "==> Seed complete."
