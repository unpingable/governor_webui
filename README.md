# gov-webui

Presentation layer for [Agent Governor](https://github.com/unpingable/agent_governor).

WebUI is untrusted — governor is the authority. The WebUI cannot sign receipts,
mint keys, broaden scope, or execute commits without challenge.

## Quick Start

```bash
pip install -e .
governor-webui              # Start on http://127.0.0.1:8000
```

## Docker

```bash
./start.sh                  # Claude Code backend (auto-detects CLI + credentials)
./start-codex.sh            # Codex backend (auto-detects Node + architecture)

# Or manual docker-compose:
docker-compose up -d                                           # Anthropic API (3 stacks)
docker-compose -f docker-compose.yml -f docker-compose.claude-code.yml up -d
docker-compose -f docker-compose.yml -f docker-compose.codex.yml up -d
docker-compose -f docker-compose.yml -f docker-compose.ollama.yml up -d
```

The default `docker-compose.yml` brings up three isolated stacks:
- Fiction mode on `:8001`
- Code mode on `:8002`
- Research mode on `:8003`

## Configuration

All configuration via environment variables. CLI flags are not used.

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND_TYPE` | `ollama` | Backend: `anthropic`, `ollama`, `claude-code`, `codex` |
| `ANTHROPIC_API_KEY` | — | API key (required for anthropic backend) |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama URL |
| `CLAUDE_PATH` | `claude` | Path to claude CLI |
| `CODEX_PATH` | `codex` | Path to codex CLI |
| `GOVERNOR_MODE` | `general` | Mode: `fiction`, `code`, `nonfiction`, `research`, `general` |
| `GOVERNOR_CONTEXT_ID` | `default` | Isolates state per user/project |
| `GOVERNOR_CONTEXTS_DIR` | `~/.governor-contexts` | Base dir for context state |
| `GOVERNOR_SOCKET` | auto-derived | Unix socket path to governor daemon |
| `GOVERNOR_AUTH_TOKEN` | — | Bearer token for mutating endpoints (empty = no auth) |
| `GOVERNOR_BIND_HOST` | `127.0.0.1` | Bind host (set `0.0.0.0` for network access) |
| `GOVERNOR_SHOW_OK_FOOTER` | `true` | Show `[Governor] OK` footer on clean responses |

## Backends

| Backend | Billing | Setup |
|---------|---------|-------|
| **Anthropic API** | Per-token API charges | `ANTHROPIC_API_KEY=sk-ant-...` |
| **Claude Code** | Claude Max subscription | `BACKEND_TYPE=claude-code` (CLI must be authenticated) |
| **Codex** | ChatGPT subscription | `BACKEND_TYPE=codex` (Node.js required) |
| **Ollama** | Free (local compute) | `BACKEND_TYPE=ollama` (GPU recommended) |

Runtime backend switching: `POST /v1/backends/switch` or use the sidebar dropdown.

## Modes

The mode determines which constraints are active and what the sidebar shows.

| Mode | System Prompt Focus | Sidebar Panels |
|------|---------------------|----------------|
| Fiction | Story consistency, character voice, affect regime | Characters, World Rules, Forbidden |
| Code | Tech decisions, patterns, interferometry | Decisions, Constraints, Compare |
| Nonfiction | Sources, positions, hedging | Status + Corrections |
| Research | Non-convergent exploration, hypothesis tracking | Claims, Assumptions, Uncertainties, Links |
| General | No mode-specific prompts | Status + Corrections |

## UI

Three HTML pages served as static assets:

| URL | Page | Purpose |
|-----|------|---------|
| `/` | `index.html` | Combined chat + governor sidebar |
| `/dashboard` | `dashboard.html` | V2 governance dashboard (runs, regime, claims) |
| `/governor/ui` | `governor.html` | Standalone governor panel |

The chat panel supports:
- Streaming responses (SSE)
- Intent form modal (gear icon) for structured session configuration
- Violation resolution flow (fix/revise/proceed)
- Mode-specific sidebar panels

## API Reference

### Chat (OpenAI-compatible)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (streaming via SSE supported) |
| `/v1/models` | GET | List available models |
| `/v1/backends` | GET | List backends with availability |
| `/v1/backends/switch` | POST | Switch backend at runtime |

### Sessions

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/sessions/` | GET | List all sessions |
| `/sessions/` | POST | Create new session |
| `/sessions/{id}` | GET | Get session with messages |
| `/sessions/{id}` | DELETE | Delete session |
| `/sessions/{id}` | PATCH | Update session title |
| `/sessions/{id}/messages` | POST | Append message |

### Governor State

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/governor/status` | GET | Full state + ViewModel v2 |
| `/governor/now` | GET | Glanceable status pill |
| `/governor/why` | GET | Decision/violation/claim feed |
| `/governor/history` | GET | Events grouped by day |
| `/governor/detail/{id}` | GET | Drill-down by item ID |
| `/governor/corrections` | GET | Resolution history |
| `/governor/export` | GET | Export all state as JSON |
| `/governor/import` | POST | Import anchors + corrections |

### Mode-Specific

| Endpoint | Method | Modes |
|----------|--------|-------|
| `/governor/fiction/{characters,world-rules,forbidden}` | GET/POST/DELETE | Fiction |
| `/governor/code/{decisions,constraints}` | GET/POST/DELETE | Code |
| `/governor/code/compare` | POST | Code (interferometry) |
| `/governor/research/{claims,assumptions,uncertainties,links}` | GET/POST/DELETE | Research |

### V2 Dashboard (Run-Centric)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v2/runs` | GET/POST | List/create runs |
| `/v2/runs/{id}` | GET | Run detail + manifest |
| `/v2/runs/{id}/events` | GET | Events (SSE via `?stream=true`) |
| `/v2/runs/{id}/claims` | GET | Extracted claims |
| `/v2/runs/{id}/violations` | GET | Violations only |
| `/v2/runs/{id}/report` | GET | Generated report |
| `/v2/runs/{id}/cancel` | POST | Cancel active run |
| `/v2/artifacts/{hash}` | GET | Content-addressed blob |
| `/v2/dashboard/summary` | GET | Aggregate statistics |
| `/v2/dashboard/regime` | GET | Current regime state |

### Intent Compiler

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v2/intent/templates` | GET | List form templates |
| `/v2/intent/schema/{name}` | GET | Build form schema for current mode |
| `/v2/intent/validate` | POST | Validate response against schema |
| `/v2/intent/compile` | POST | Compile to intent + constraints (emits receipt) |
| `/v2/intent/policy` | GET | Current form policy |

### Health

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Backend + governor connectivity |
| `/api/info` | GET | Version, endpoints, capabilities |

Auth: mutating endpoints (POST/PUT/DELETE/PATCH) require `Authorization: Bearer <token>` when `GOVERNOR_AUTH_TOKEN` is set.

## Violation Resolution

When the AI generates content that violates a constraint:

```
[Governor] Blocked — 1 violation(s) detected.

  - no_eval_anchor: Cannot use eval() in production code

How would you like to handle this?

1. fix — Rewrite to comply with the constraint
2. revise — Update the constraint to permit this
3. proceed — Allow this once and log an exception
```

While a violation is pending, normal messages are blocked. Only resolution commands are accepted.

## Tests

```bash
pip install -e ".[dev]"
python3 -m pytest tests/ -v
```

239 tests across 5 files:
- `test_adapter.py` (108) — chat, sessions, governor endpoints, auth, backends
- `test_dashboard_v2_api.py` (39) — runs, artifacts, dashboard summary
- `test_intent_api.py` (25) — form schema, validation, compilation
- `test_parity.py` (5) — split-brain tripwire (daemon RPC used, not ChatBridge)
- `test_summaries.py` (62) — status pill, history, detail queries

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the transport story, split-brain fix, and how gov-webui fits in the ecosystem.

## Security

- The WebUI is an **untrusted cockpit** — it renders governor state but cannot override it
- All governance decisions happen in the daemon/core
- Set `GOVERNOR_AUTH_TOKEN` for shared/network deployments
- Set `GOVERNOR_BIND_HOST=127.0.0.1` (default) to prevent network exposure
- API keys belong in environment variables, not code
- Exception logs contain partial responses — review before sharing
