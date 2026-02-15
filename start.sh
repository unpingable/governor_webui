#!/usr/bin/env bash
set -euo pipefail

# Auto-detect environment and start Governor WebUI with Claude Code backend.
# Handles snap Docker's $HOME remapping automatically.

cd "$(dirname "$0")"

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

# ── Sync agent-governor source for Docker build ──────────────────────────
# agent-governor is not on PyPI — copy source into build context.
AGENT_GOV_DIR="$(cd "$(dirname "$0")/../agent_gov" && pwd)"
if [ ! -d "$AGENT_GOV_DIR/src/governor" ]; then
  echo "Error: agent-governor source not found at $AGENT_GOV_DIR"
  echo "Expected: ../agent_gov relative to this repo"
  exit 1
fi
rm -rf agent-governor
mkdir -p agent-governor/src
cp "$AGENT_GOV_DIR/pyproject.toml" agent-governor/
cp "$AGENT_GOV_DIR/README.md" agent-governor/
cp -r "$AGENT_GOV_DIR/src/governor" agent-governor/src/
echo "Synced agent-governor from $AGENT_GOV_DIR"

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
  -f docker-compose.claude-code.yml \
  up --build -d "$@"

echo ""
echo "Fiction:  http://localhost:8001"
echo "Code:     http://localhost:8002"
echo "Research: http://localhost:8003"
