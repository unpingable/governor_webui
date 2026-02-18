# Compatibility

This repo is the web UI for Agent Governor.

## Version coupling

gov-webui is independently versioned. It targets Governor contract versions
listed below, not Governor semver directly.

## Compatible Governor versions

- Tested against: `>=2.3.0`
- Delegates chat to daemon via Unix socket RPC (GOVERNOR_SOCKET env var)
- Non-chat endpoints use direct Python imports from `governor` package

## Contract versions (wire / JSON)

| Contract | Version | Used For |
|----------|---------|----------|
| RPC protocol | 1.0 | Daemon chat delegation (chat.send, chat.stream) |
| StatusRollup schema | 1 | Dashboard display |
| ViewModel schema | v2 | State endpoint (`/api/state`) |
| Receipt schema | 2 | Receipt listing |
| Intent schema | 1 | Intent compiler modal (`/v2/intent/*`) |

## Feature negotiation

If the governor daemon socket is unavailable, chat endpoints return 503.
Non-chat endpoints (sessions, governor/status, dashboard) use direct imports
and work without a running daemon.
