# Gov-WebUI Architecture

> Gov-webui is the browser-accessible presentation layer for Agent Governor.
> It serves a self-contained chat + governance UI and an OpenAI-compatible API.
> It has no authority of its own.

## Position in the Ecosystem

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Maude     │  │   Guvnah    │  │  gov-webui  │
│ (Python TUI)│  │ (Electron)  │  │  (FastAPI)  │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
  Unix socket      stdio pipe    Unix socket (chat)
  JSON-RPC 2.0     JSON-RPC 2.0  + direct import (state)
       │                │                │
       └────────────────┴────────┬───────┘
                                 │
                     ┌───────────┴───────────┐
                     │   Governor Daemon      │
                     │   (governor serve)     │
                     │   25 RPC methods       │
                     └───────────┬───────────┘
                                 │
                     ┌───────────┴───────────┐
                     │   60+ Governor Modules │
                     └───────────┬───────────┘
                                 │
                ┌────────┬───────┴───┬────────┐
                │        │           │        │
           Anthropic   Ollama   Claude CLI  Codex CLI
```

All three clients share the same daemon contract. Gov-webui is the only one that
also imports governor modules directly (for non-chat state queries).

## Transport: The Split-Brain Architecture

Gov-webui uses a hybrid transport model:

### Chat path: Daemon via Unix socket

```
Browser ──HTTP POST──▶ adapter.py ──Unix socket RPC──▶ governor daemon
                       DaemonChatClient                 chat.send / chat.stream
```

The `DaemonChatClient` (`daemon_client.py`) is a 5-method JSON-RPC client that
handles only the governance-relevant chat path:

| Method | RPC | Purpose |
|--------|-----|---------|
| `chat_send()` | `chat.send` | Non-streaming governed chat |
| `chat_stream()` | `chat.stream` | Streaming with `chat.delta` notifications |
| `commit_pending()` | `commit.pending` | Check for pending violations |
| `chat_models()` | `chat.models` | List available models |
| `chat_backend()` | `chat.backend` | Current backend info |

This ensures all chat goes through the daemon's governance pipeline:
augmentation, continuity checking, violation resolution, and receipt emission.

Socket path resolution: `GOVERNOR_SOCKET` env var, or auto-derived from
`$XDG_RUNTIME_DIR/governor-{sha256(gov_dir)[:12]}.sock`.

### Non-chat path: Direct governor imports

```
Browser ──HTTP GET──▶ adapter.py ──direct import──▶ governor modules
                      GovernorContextManager           SessionStore
                      ContinuityRegistry               DecisionLedger
                      build_viewmodel()                 ResearchStore
```

Non-chat endpoints (sessions, governor/status, dashboard, fiction/code panels)
import governor modules directly. This is acceptable because:
- These are read-only state queries, not governance-relevant actions
- The daemon's RPC methods for these call the same underlying code
- Direct imports avoid the socket round-trip for read-heavy sidebar polling

### Why not full daemon delegation?

The sidebar polls `/governor/status`, `/governor/now`, and mode-specific endpoints
every 3 seconds. Routing all of this through the daemon socket would add latency
to every poll cycle without improving correctness. The invariant is:

> All governance-relevant actions (chat generation, violation resolution, receipt emission)
> go through the daemon. Read-only state queries may use direct imports.

### Parity enforcement

`tests/test_parity.py` contains 5 tripwire tests that verify:
- `DaemonChatClient` is called for chat (not `ChatBridge` directly)
- Pending violations are surfaced through the daemon
- Governor footer is appended to responses

## Module Layout

```
src/gov_webui/
├── adapter.py         # FastAPI app: all HTTP routes, lazy-init singletons
├── daemon_client.py   # DaemonChatClient: Unix socket JSON-RPC (chat only)
├── summaries.py       # Pure functions: status pill, one-sentence, why-feed, history
└── static/
    ├── index.html     # Combined chat + governor sidebar (single-page app)
    ├── dashboard.html # V2 governance dashboard (runs, regime, claims)
    └── governor.html  # Standalone governor panel
```

### Lazy-initialized singletons (adapter.py)

```python
_bridge: ChatBridge | None              # Created but NOT used for chat (legacy)
_daemon_client: DaemonChatClient | None # Unix socket RPC (chat-path authority)
_context_manager: GovernorContextManager # Shared across all endpoints
_session_store: SessionStore            # Session persistence (SQLite)
```

The `ChatBridge` is still instantiated (for backend detection and model listing)
but is never used for chat generation. Chat always goes through `_daemon_client`.

## Request Flow: Chat Message

```
1. Browser POSTs to /v1/chat/completions {messages, stream: true}
2. adapter.py checks GOVERNOR_AUTH_TOKEN (if set)
3. _get_daemon_client() returns or creates DaemonChatClient
4. DaemonChatClient.commit_pending() — check for pre-existing violation
5. If violation pending → return violation prompt (no generation)
6. DaemonChatClient.chat_stream() → daemon
7. Daemon: augment_messages() (system prompt, anchors, puppet constraints)
8. Daemon: stream through backend (Anthropic/Ollama/Claude/Codex)
9. Daemon: check_response_blocking() on full content
10. Daemon: emit gate receipt (pass or block)
11. If blocking violation → daemon creates PendingViolation
12. adapter.py converts JSON-RPC notifications → SSE `data:` lines
13. Browser renders streaming response + governor footer
```

## Request Flow: Violation Resolution

```
1. Browser sends "fix" / "revise" / "proceed" as next chat message
2. adapter.py detects resolution command
3. DaemonChatClient → daemon commit.fix / commit.revise / commit.proceed
4. Daemon resolves via ViolationResolver
5. For "fix": daemon regenerates compliant response
6. For "revise": daemon updates anchor constraint
7. For "proceed": daemon logs exception, returns original content
8. Response streamed back through same SSE path
```

## Deployment

### Local (development)

```bash
# Terminal 1: start daemon
governor serve

# Terminal 2: start webui
governor-webui
# → http://127.0.0.1:8000
```

### Docker (multi-mode)

`docker-compose.yml` runs three isolated stacks:
- Fiction (`:8001`) — `GOVERNOR_MODE=fiction`
- Code (`:8002`) — `GOVERNOR_MODE=code`
- Research (`:8003`) — `GOVERNOR_MODE=research`

Each stack gets its own `GOVERNOR_CONTEXT_ID`, isolating all state.

Override files layer on backend-specific config:
- `docker-compose.claude-code.yml` — mounts Claude CLI + credentials
- `docker-compose.codex.yml` — mounts Codex CLI + Node.js config
- `docker-compose.ollama.yml` — adds Ollama container

### VM / systemd

See `agent_gov/docs/DEPLOYMENT.md` for the full systemd setup:
- `governor.service` — daemon with `BACKEND_TYPE` + `ANTHROPIC_API_KEY`
- `gov-webui.service` — adapter with `GOVERNOR_SOCKET` env var
- Secrets in `/etc/governor/secrets.env` (chmod 600)
- Caddy reverse proxy for HTTPS

Key gotcha: `GOVERNOR_SOCKET` must be set explicitly in the systemd unit —
`XDG_RUNTIME_DIR` is not inherited by systemd services.

## What Gov-WebUI Is NOT

- Not an authority — governor is the authority
- Not a daemon — it's a stateless HTTP adapter
- Not a replacement for Maude/Guvnah — different UX for different contexts
- Not where governance decisions are made — that's the daemon's job

It's a window into the governor. The glass doesn't move the gears.
