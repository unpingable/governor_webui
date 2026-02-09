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
./start.sh                  # Claude Code backend (auto-detects CLI)
./start-codex.sh            # Codex backend (auto-detects Node)

# Or manual docker-compose:
docker-compose up -d                                           # Anthropic API
docker-compose -f docker-compose.yml -f docker-compose.claude-code.yml up -d
```

## Architecture

```
gov-webui → agent-governor (pip dependency)
```

All governance logic lives in `agent-governor`. This package provides:
- FastAPI HTTP routing (`adapter.py`)
- ViewModel presentation formatting (`summaries.py`)
- Static HTML/JS chat + governor sidebar UI
