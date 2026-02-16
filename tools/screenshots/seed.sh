#!/usr/bin/env bash
# seed.sh — Import deterministic state for screenshot generation.
# Run against a live gov-webui research stack (default: localhost:8003).
#
# Creates a fresh session with synthetic chat history so screenshots
# show clean state (no leftover errors from real usage).
set -euo pipefail

BASE="${SCREENSHOT_BASE_URL:-http://127.0.0.1:8003}"
FIXTURE_DIR="$(dirname "$0")/fixtures"
AUTH="${GOVERNOR_AUTH_TOKEN:+Authorization: Bearer $GOVERNOR_AUTH_TOKEN}"

echo "==> Seeding research demo state into $BASE ..."

# 1. Import research sidebar state (claims, anchors)
curl -sS -X POST "$BASE/governor/import" \
  -H "Content-Type: application/json" \
  ${AUTH:+-H "$AUTH"} \
  --data-binary "@$FIXTURE_DIR/research_demo.json" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'    Imported: {d[\"imported\"]}, Skipped: {d[\"skipped\"]}')"

# 2. Delete old sessions, create a fresh one with synthetic chat
echo "    Clearing old sessions..."
OLD_IDS=$(curl -sS "$BASE/sessions/" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for s in data.get('sessions', []):
    print(s['id'])
")
for sid in $OLD_IDS; do
  curl -sS -X DELETE "$BASE/sessions/$sid" > /dev/null 2>&1 || true
done

echo "    Creating screenshot session..."
SESSION_ID=$(curl -sS -X POST "$BASE/sessions/" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-5-20250929","title":"Scaling Laws in Deep Learning"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

# 3. Seed synthetic messages (user question + assistant reply with DOI refs)
curl -sS -X POST "$BASE/sessions/$SESSION_ID/messages" \
  -H "Content-Type: application/json" \
  -d '{
    "role": "user",
    "content": "What do we know about neural scaling laws? I have the Kaplan et al. and Hoffmann et al. papers."
  }' > /dev/null

curl -sS -X POST "$BASE/sessions/$SESSION_ID/messages" \
  -H "Content-Type: application/json" \
  -d '{
    "role": "assistant",
    "content": "Neural scaling laws describe power-law relationships between compute budget, dataset size, model parameters, and loss. The foundational work is Kaplan et al. (doi:10.48550/arXiv.2001.08361), which established that performance scales predictably with compute across several orders of magnitude.\n\nHoffmann et al. (doi:10.48550/arXiv.2203.15556) revised the optimal allocation, showing that earlier models were significantly over-parameterized relative to their training data — the \"Chinchilla\" result. Their key finding: for a fixed compute budget, you should scale data and parameters roughly equally.\n\nOpen questions:\n- Whether these laws hold at frontier scale or plateau\n- Whether mixture-of-experts architectures follow the same scaling curves\n- How data quality (not just quantity) affects the exponent\n\nCANDIDATE_SOURCE: arxiv:2401.00001\n\nThe MoE question is particularly interesting — early results suggest better compute efficiency but the scaling exponent may differ from dense transformers."
  }' > /dev/null

echo "    Session $SESSION_ID seeded with 2 messages."

echo "==> Seed complete."
