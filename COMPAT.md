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

## Phase 0 contracts (v0.4.0+)

| Contract | Version | Used For |
|----------|---------|----------|
| ProjectState schema | 1 | Code/Research builder project persistence (`project.json`) |
| Contract.config | 1 | Constraints Wizard typed config (artifact type, length, voice, bans, etc.) |
| Config hash | SHA-256 canonical | `config_hash` (16 hex) + `config_hash_full` (64 hex) — server always recomputes |
| `[CONSTRAINTS]` block | 1 | System message injected into chat path; machine-parseable, no prose |
| Validation findings | 1 | Research validator output: weasel words, ban matches, length bands, format checks |

### Config hash cross-language invariance

The canonical form is: recursive dict key sort → sort known-set lists (`voice`, `bans`, `structure`) → strip strings → `JSON.stringify` / `json.dumps(separators=(",",":"), ensure_ascii=False)` → SHA-256.

Test vector:
```
Input:     {"artifact_type":"essay","bans":["studies show","experts agree"],"length":"medium","voice":["wry","dry"]}
Canonical: {"artifact_type":"essay","bans":["experts agree","studies show"],"length":"medium","voice":["dry","wry"]}
SHA-256:   a5e80158a5636c553943b23fb8559db9f1f0cf250a8d5f6bb914afe720be1cc8
Short:     a5e80158a5636c55
```

## Feature negotiation

If the governor daemon socket is unavailable, chat endpoints return 503.
Non-chat endpoints (sessions, governor/status, dashboard) use direct imports
and work without a running daemon.
