#!/usr/bin/env bash
set -euo pipefail

# Auto-detect environment and start Phosphor with Claude Code backend.
# Handles snap Docker's $HOME remapping automatically.
#
# Uses the pre-built GHCR image by default. If ../agent_gov exists,
# syncs local deps and builds from source instead (developer mode).

cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"

# ── Detect real home directory ──────────────────────────────────────────────
# Snap Docker remaps $HOME inside its sandbox. Use getent to get the truth.
REAL_HOME=$(eval echo "~$(whoami)")
if [ ! -d "$REAL_HOME" ]; then
  REAL_HOME="$HOME"
fi

# ── Detect Claude CLI version ──────────────────────────────────────────────
VERSIONS_DIR="$REAL_HOME/.local/share/claude/versions"
if [ ! -d "$VERSIONS_DIR" ]; then
  echo "Error: Claude Code CLI not found at $VERSIONS_DIR"
  echo "Install it: https://docs.anthropic.com/en/docs/claude-code"
  exit 1
fi

# Pick the latest version (highest semver)
CLAUDE_VERSION=$(ls "$VERSIONS_DIR" | sort -V | tail -1)
CLAUDE_BINARY="$VERSIONS_DIR/$CLAUDE_VERSION"

if [ ! -f "$CLAUDE_BINARY" ]; then
  echo "Error: Claude binary not found at $CLAUDE_BINARY"
  echo "Available versions:"
  ls "$VERSIONS_DIR"
  exit 1
fi

if [ ! -x "$CLAUDE_BINARY" ]; then
  echo "Warning: Claude binary not executable, fixing..."
  chmod +x "$CLAUDE_BINARY"
fi

# ── Check credentials ──────────────────────────────────────────────────────
CREDS="$REAL_HOME/.claude/.credentials.json"
if [ ! -f "$CREDS" ]; then
  echo "Error: Claude credentials not found at $CREDS"
  echo "Run 'claude' once to authenticate."
  exit 1
fi

# ── Decide: pre-built image or local build ────────────────────────────────
BUILD_FLAGS=""
BUILD_OVERRIDE=""
AGENT_GOV_DIR="$SCRIPT_DIR/../agent_gov"

if [ -d "$AGENT_GOV_DIR/src/governor" ]; then
  echo "Found agent_gov — building from source"
  source "$SCRIPT_DIR/sync-deps.sh"
  BUILD_FLAGS="--build"
  BUILD_OVERRIDE="-f docker-compose.build.yml"
else
  echo "Using pre-built image from ghcr.io/unpingable/phosphor"
fi

# ── Write .env for docker-compose ──────────────────────────────────────────
cat > .env <<EOF
REAL_HOME=$REAL_HOME
CLAUDE_VERSION=$CLAUDE_VERSION
EOF

echo "Claude Code v$CLAUDE_VERSION"
echo "Binary: $CLAUDE_BINARY"
echo "Home:   $REAL_HOME"
echo ""

# ── Start containers ───────────────────────────────────────────────────────
docker-compose \
  -f docker-compose.yml \
  $BUILD_OVERRIDE \
  -f docker-compose.claude-code.yml \
  up $BUILD_FLAGS -d "$@"

echo ""
echo "Fiction:  http://localhost:8001"
echo "Code:     http://localhost:8002"
echo "Research: http://localhost:8003"
