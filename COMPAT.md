# Compatibility

Phosphor is the user-facing governed agent client for Agent Governor.

## Version coupling

Phosphor is independently versioned. It targets Governor contract versions
listed below, not Governor semver directly.

## Compatible Governor versions

- Tested against: `2.8.x` (daemon at 99 RPC methods as of 2026-07-05)
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

## Shell-contract methods (desk-mode lane)

The following daemon RPC methods are part of the governed-shell contract
(GS-2b/3/4, shell-contract §1-§4). They are present in the daemon as of
2.8.x but **not yet consumed by Phosphor** — they are the target of the
U3 desk-mode lane:

| Method | Kind | Notes |
|--------|------|-------|
| `operator.decisions.list` | read | Unified operator decision feed |
| `operator.decisions.resolve` | mutating | The ONE mutation door (GS-3) |
| `operator.watch` | streaming | Bounded poll loop, re-subscribe on return |
| `runtime.session.list` | read | List supervised sessions |
| `runtime.session.create` | mutating | Create a new session |
| `runtime.session.launch` | mutating | Launch a supervised session |
| `runtime.session.kill` | mutating | Terminate a session |
| `runtime.intervention.list` | read | List pending tool-call approvals |
| `runtime.intervention.resolve` | mutating | Approve / deny a tool call |
| `runtime.promotion.get` | read | Pending workspace changes |
| `runtime.promotion.diff` | read | Unified diff of changes |
| `runtime.promotion.resolve` | mutating | Accept / reject workspace changes |

`DaemonShellClient` (U3-A) wraps these over the same Content-Length
JSON-RPC framing used by `DaemonChatClient`.

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
