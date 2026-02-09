#!/usr/bin/env bash
set -euo pipefail

# Auto-detect environment and start Governor WebUI with Codex backend.
# Handles snap Docker's $HOME remapping automatically.

cd "$(dirname "$0")"

# ── Detect real home directory ──────────────────────────────────────────────
# Snap Docker remaps $HOME inside its sandbox. Use getent to get the truth.
REAL_HOME=$(eval echo "~$(whoami)")
if [ ! -d "$REAL_HOME" ]; then
  REAL_HOME="$HOME"
fi

# ── Detect Node.js version ──────────────────────────────────────────────────
NODE_VERSION=""
if command -v node &>/dev/null; then
  NODE_VERSION=$(node --version)
elif [ -d "$REAL_HOME/.nvm/versions/node" ]; then
  NODE_VERSION=$(ls "$REAL_HOME/.nvm/versions/node/" | sort -V | tail -1)
fi

if [ -z "$NODE_VERSION" ]; then
  echo "Error: Node.js not found (checked 'node --version' and ~/.nvm/versions/node/)"
  echo "Install Node.js or nvm: https://github.com/nvm-sh/nvm"
  exit 1
fi

echo "Node.js: $NODE_VERSION"

# ── Detect architecture ─────────────────────────────────────────────────────
RAW_ARCH=$(uname -m)
case "$RAW_ARCH" in
  x86_64)   ARCH="x86_64-unknown-linux-musl" ;;
  aarch64)  ARCH="aarch64-unknown-linux-musl" ;;
  arm64)    ARCH="aarch64-unknown-linux-musl" ;;
  *)
    echo "Error: Unsupported architecture: $RAW_ARCH"
    echo "Codex supports x86_64 and aarch64."
    exit 1
    ;;
esac

# ── Verify Codex binary ─────────────────────────────────────────────────────
CODEX_BINARY="$REAL_HOME/.nvm/versions/node/$NODE_VERSION/lib/node_modules/@openai/codex/vendor/$ARCH/codex/codex"

if [ ! -f "$CODEX_BINARY" ]; then
  echo "Error: Codex binary not found at $CODEX_BINARY"
  echo "Install it: npm install -g @openai/codex"
  exit 1
fi

if [ ! -x "$CODEX_BINARY" ]; then
  echo "Warning: Codex binary not executable, fixing..."
  chmod +x "$CODEX_BINARY"
fi

# ── Check Codex config ──────────────────────────────────────────────────────
if [ ! -d "$REAL_HOME/.codex" ]; then
  echo "Error: Codex config not found at $REAL_HOME/.codex/"
  echo "Run 'codex' once to authenticate."
  exit 1
fi

# ── Write .env for docker-compose ──────────────────────────────────────────
cat > .env <<EOF
REAL_HOME=$REAL_HOME
NODE_VERSION=$NODE_VERSION
ARCH=$ARCH
EOF

echo "Codex binary: $CODEX_BINARY"
echo "Architecture: $ARCH"
echo "Home:         $REAL_HOME"
echo ""

# ── Start containers ───────────────────────────────────────────────────────
docker-compose \
  -f docker-compose.yml \
  -f docker-compose.codex.yml \
  up --build -d "$@"

echo ""
echo "Fiction:  http://localhost:8001"
echo "Code:     http://localhost:8002"
echo "Research: http://localhost:8003"
