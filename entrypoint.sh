#!/bin/bash
set -e

# Initialize governor context if needed
CONTEXTS_DIR="${GOVERNOR_CONTEXTS_DIR:-/contexts}"
CTX_ID="${GOVERNOR_CONTEXT_ID:-default}"
GOV_DIR="$CONTEXTS_DIR/$CTX_ID/.governor"

CTX_DIR="$CONTEXTS_DIR/$CTX_ID"
CTX_META="$CTX_DIR/_context.json"

if [ ! -d "$GOV_DIR" ]; then
    mkdir -p "$GOV_DIR/sessions"
    echo '{"sessions": {}, "mainline": null}' > "$GOV_DIR/sessions/index.json"
fi

# Write context metadata if missing (required by GovernorContextManager.get())
if [ ! -f "$CTX_META" ]; then
    MODE="${GOVERNOR_MODE:-general}"
    NOW=$(python3 -c "from datetime import datetime,timezone; print(datetime.now(timezone.utc).isoformat())")
    cat > "$CTX_META" <<CTXEOF
{
  "context_id": "$CTX_ID",
  "mode": "$MODE",
  "root": "$CTX_DIR",
  "governor_dir": "$GOV_DIR",
  "created_at": "$NOW",
  "metadata": {}
}
CTXEOF
    echo "Created context metadata: $CTX_META (mode=$MODE)"
fi

# Start governor daemon in background on a Unix socket
SOCKET_PATH=$(python3 -c "
from gov_webui.daemon_client import default_socket_path
from pathlib import Path
print(default_socket_path(Path('$GOV_DIR')))
")

echo "Starting governor daemon at $SOCKET_PATH"
governor serve \
    --socket "$SOCKET_PATH" \
    --mode "${GOVERNOR_MODE:-general}" \
    &
DAEMON_PID=$!

# Export socket path so the adapter can find it
export GOVERNOR_SOCKET="$SOCKET_PATH"
export GOVERNOR_DIR="$GOV_DIR"

# Wait for socket to appear (up to 5 seconds)
for i in $(seq 1 50); do
    if [ -S "$SOCKET_PATH" ]; then
        echo "Daemon ready (pid $DAEMON_PID)"
        break
    fi
    sleep 0.1
done

if [ ! -S "$SOCKET_PATH" ]; then
    echo "Warning: daemon socket not found after 5s, starting adapter anyway"
fi

# Start the web adapter in foreground
exec uvicorn gov_webui.adapter:app --host 0.0.0.0 --port 8000
