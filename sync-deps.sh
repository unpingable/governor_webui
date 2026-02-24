#!/usr/bin/env bash
# Sync local-only dependencies into Docker build context.
#
# Both start.sh and start-codex.sh source this file. Add new non-PyPI
# dependencies here — one place, both launch paths.
#
# Usage (from start scripts):  source sync-deps.sh
#
# Requires: SCRIPT_DIR set to repo root before sourcing.

set -euo pipefail

AGENT_GOV_DIR="$(cd "$SCRIPT_DIR/../agent_gov" && pwd)"

# ── agent-governor ────────────────────────────────────────────────────────
if [ ! -d "$AGENT_GOV_DIR/src/governor" ]; then
  echo "Error: agent-governor source not found at $AGENT_GOV_DIR"
  echo "Expected: ../agent_gov relative to this repo"
  exit 1
fi
rm -rf "$SCRIPT_DIR/agent-governor"
mkdir -p "$SCRIPT_DIR/agent-governor/src"
cp "$AGENT_GOV_DIR/pyproject.toml" "$SCRIPT_DIR/agent-governor/"
cp "$AGENT_GOV_DIR/README.md" "$SCRIPT_DIR/agent-governor/"
cp -r "$AGENT_GOV_DIR/src/governor" "$SCRIPT_DIR/agent-governor/src/"
echo "Synced agent-governor from $AGENT_GOV_DIR"

# ── receipt-v1 ────────────────────────────────────────────────────────────
RECEIPT_V1_DIR="$AGENT_GOV_DIR/libs/receipt_v1"
if [ -d "$RECEIPT_V1_DIR/src/receipt_v1" ]; then
  rm -rf "$SCRIPT_DIR/receipt-v1"
  mkdir -p "$SCRIPT_DIR/receipt-v1/src"
  cp "$RECEIPT_V1_DIR/pyproject.toml" "$SCRIPT_DIR/receipt-v1/"
  cp -r "$RECEIPT_V1_DIR/src/receipt_v1" "$SCRIPT_DIR/receipt-v1/src/"
  echo "Synced receipt-v1 from $RECEIPT_V1_DIR"
else
  echo "Warning: receipt_v1 not found at $RECEIPT_V1_DIR — receipt export/verify will 500"
fi

# ── Add new local deps above this line ────────────────────────────────────
