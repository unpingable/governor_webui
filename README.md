# Governor WebUI

**The cockpit for governed AI.** Chat with any model — governance happens underneath. Every turn gets a receipt. Every constraint is visible. Nothing writes to canon without proof.

```text
You:       "Write chapter 3 — Elena finds the letter."
Model:     [drafts chapter 3]
Governor:  constraints_hash ✓  config_hash ✓  mode=fiction  turn_seq=7
Receipt:   embedded in response, hash-chained, exportable
Artifact:  promote draft → revisable artifact with version history
```

400+ tests. Four backends. The model proposes — the governor constrains — the UI makes it legible.

> *This is the presentation layer for [Agent Governor](https://github.com/unpingable/agent_governor). The governor daemon is the authority. The UI can render state and submit requests, but it cannot override policy, mint receipts, or write to canon.*

---

## What You Get

### Governed chat with receipts

Every assistant turn carries a **governance receipt** — constraints hash, config hash, turn sequence, forced flag. Not decorative metadata. Tamper with the chain and the hash breaks.

The receipt strip appears below each response. Click a hash to copy. Expand for details. The trust trail is always visible, never hidden behind a settings page.

### Artifact engine

Promote any assistant output to a **revisable artifact** with version history, kind inference, and provenance tracking. Artifacts are file-backed, versioned, and conflict-safe (optimistic concurrency with 409 handling).

- **Promote from chat** — select text, promote to new or existing artifact
- **Inline editing** — full editor with edit/preview/history tabs
- **Kind inference** — auto-detects code, prose, markdown, config
- **Version history** — every save is a version, load any prior state
- **List management** — filter by title/kind, inline rename, copy artifact ID
- **Dirty guards** — unsaved changes warn before navigation, Ctrl+S saves

### Effective config panel

See exactly what policy the governor is enforcing *right now*. The config panel shows resolved fields (defaults + session overrides + system clamps), mode, and diagnostic info. Filter to non-defaults only. Copy the full config envelope as JSON.

No more guessing what constraints are active. Open the panel, read the contract.

### Receipt export and verification

Export the full receipt chain as canonical JSONL. Verify chain integrity — structure checks, hash verification, chain continuity — in one click. Upload a previously exported chain to verify it independently.

The trust loop closes: generate receipts → export → verify → the math either checks out or it doesn't.

---

## Two Loops, One Cockpit

Every mode runs the same two loops. The sidebar changes; the enforcement doesn't.

### 1) Capture: draft > pending > accept

The assistant says something. The UI detects "definition/citation"-shaped content. Detections show up as **chips** and collect in a **Pending** drawer. Nothing auto-promotes. **Accept** writes to the canonical store *with a receipt*.

```
1. Assistant outputs: "...scaling exponent is 0.76 (doi:10.1234/example.2020)."
   → Chip appears: citation: doi:10.1234/example.2020
2. Click ACCEPT
   → Ledger now has source_ref + claim + receipt
3. Next turn
   → Assistant can cite the accepted source without re-pasting
   → Why overlay shows: injected / matched / floating counts
```

Citations are typed (`doi:/arxiv:/rfc:/cve:/pypi:`) and audited per-turn. Receipts come from the governor daemon, not the UI.

### 2) Violations: block > resolve > continue

When output violates a constraint, chat is blocked until you choose:

```
Assistant outputs: "Use eval() here..."
Governor sees:     constraint violation (no_eval_anchor)
Governor acts:     BLOCK + resolution options
UI enforces:       Fix / Revise / Proceed — nothing else accepted
```

This is not "chat with tabs." It's a cockpit for a write-gated system.

---

## Modes

Modes are **policy bundles + sidepanels**. Same core loop, different constraints.

| Mode | Focus | Sidebar | Port |
|------|-------|---------|------|
| **Fiction** | Continuity + canon | Characters, World Rules, Forbidden | `:8001` |
| **Code** | Decisions + constraints | Decisions, Constraints, Compare | `:8002` |
| **Research** | Claims + provenance | Claims, Assumptions, Uncertainties, Links, Why overlay | `:8003` |
| General | No mode-specific policy | Status + Corrections | — |

Research mode is where the capture loop is most visible: DOI/arXiv/CVE/RFC/PyPI references get extracted, accepted sources constrain the next turn, and the Why overlay shows exactly which sources were injected vs referenced vs floating.

Fiction mode is the proof that governance isn't just for crisp ground truth. If it works where facts are fuzzy — tracking canon, tone, consent — it's not compliance middleware. It generalizes.

---

## Structured Builders

Code and Research modes include a **structured iteration loop** that replaces "chat until it works" with a mechanically gated workflow:

```
Intent → Contract → Plan → Accept → Run/Validate → Done
```

Every step requires explicit user action. Chat never mutates project state. The builder sidebar shows where you are, what's locked, and what's next.

### Code Builder

```
1. Set intent         "Parse CSV files and output JSON"
2. Lock intent        (prevents drift — you said what you meant)
3. Open wizard        Set artifact=tool, length=medium, voice=dry
   → Save constraints (server hashes config, injects [CONSTRAINTS] block)
4. Add plan phases    "Implementation" → "Testing"
5. Chat               Ask the model to write the parser
   → It sees your [CONSTRAINTS] block automatically
6. Accept code block  Click Accept on the code → saved to workspace
7. Advance plan       proposed → accepted → in_progress → completed
8. Run                POST /governor/code/run — preflight check, then execute
```

### Research Builder

```
1. Set thesis         "How does cognitive load affect code review quality?"
2. Lock thesis        (your question is your question)
3. Open wizard        Set artifact=lit_review, voice=academic, citations=required,
                      bans=["studies show", "experts agree"]
   → Save constraints (model will see length band, voice, citation requirements)
4. Chat               Ask the model to draft — constraints are live
5. Accept draft       Click Accept on markdown → saved as .md to workspace
6. Validate           POST /governor/research/project/validate
   → Catches: weasel words without citations, banned phrases,
     scope constraint violations, length band mismatches
```

### Constraints Wizard

Both builders share a **Constraints Wizard** that replaces freeform contract modals:

| Field | Widget | Effect |
|-------|--------|--------|
| Artifact type | Dropdown | `tool`, `essay`, `lit_review`, etc. |
| Length | Segmented | `short` / `medium` / `long` → word count bands |
| Voice | Chips | `dry`, `wry`, `academic`, `spicy`, etc. |
| Citations | Segmented | `none` / `light` / `required` |
| Bans | Tag input | Literal phrases the model must avoid |
| Format | Toggles | Tables, bullets, headings on/off |
| Strict mode | Toggle | Warnings vs hard fails in validation |

The wizard computes a canonical SHA-256 hash of the config. The server always recomputes — never trusts client hashes. A live preview shows the exact `[CONSTRAINTS]` block the model will see as you adjust widgets.

---

## Try It

### Docker (recommended)

Brings up **three isolated stacks** with persistent state volumes:

```bash
./start.sh              # Claude Code backend (auto-detects CLI + credentials)
./start-codex.sh        # Codex backend (auto-detects Node + architecture)

# Or direct:
docker compose up -d
```

```
Fiction:  http://localhost:8001
Code:     http://localhost:8002
Research: http://localhost:8003
```

Quick sanity: `curl -s http://localhost:8003/health`

### Local dev

```bash
pip install -e .
governor-webui          # http://127.0.0.1:8000
```

---

## Pick a Backend

The UI talks to multiple inference backends. The governor enforces constraints regardless.

| Backend | Billing | Setup |
|---------|---------|-------|
| **Anthropic API** | Per-token | `BACKEND_TYPE=anthropic` + `ANTHROPIC_API_KEY=sk-ant-...` |
| **Claude Code CLI** | Claude Max subscription | `BACKEND_TYPE=claude-code` (CLI must be authenticated) |
| **Codex CLI** | ChatGPT subscription | `BACKEND_TYPE=codex` (Node.js required) |
| **Ollama** | Free (local) | `BACKEND_TYPE=ollama` + `OLLAMA_HOST=http://localhost:11434` |

Switch at runtime via the sidebar dropdown or `POST /v1/backends/switch`.

---

## Security Model

The WebUI is a **non-authoritative client**. Governance decisions happen in the daemon.

- The UI cannot sign receipts, broaden scope, or mint keys
- Set `GOVERNOR_AUTH_TOKEN` to lock mutating endpoints on shared deployments
- Default bind is loopback only (`127.0.0.1`) — set `0.0.0.0` only intentionally
- API keys belong in env vars, not code
- Exception logs may contain partial responses — review before sharing

---

## Configuration

All configuration is env vars.

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND_TYPE` | `ollama` | `anthropic`, `ollama`, `claude-code`, `codex` |
| `ANTHROPIC_API_KEY` | — | Required for anthropic backend |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama URL |
| `GOVERNOR_MODE` | `general` | `fiction`, `code`, `research`, `nonfiction`, `general` |
| `GOVERNOR_CONTEXT_ID` | `default` | Isolates state per user/project |
| `GOVERNOR_CONTEXTS_DIR` | `~/.governor-contexts` | Where state lives |
| `GOVERNOR_SOCKET` | auto-derived | Unix socket path to governor daemon |
| `GOVERNOR_AUTH_TOKEN` | — | Bearer token for mutating endpoints |
| `GOVERNOR_BIND_HOST` | `127.0.0.1` | Bind host |

---

## Pages

| URL | Purpose |
|-----|---------|
| `/` | Chat + governor sidebar |
| `/dashboard` | Governance dashboard (runs, regime, claims) |
| `/governor/ui` | Standalone governor panel |

---

## Development

```bash
pip install -e ".[dev]"
python3 -m pytest tests/ -v    # 400+ tests
```

### Screenshots

`npm run screenshots` regenerates `docs/img/*` via Playwright with seeded fixture state. See `tools/screenshots/` for fixtures and specs.

```bash
npm install && npx playwright install    # one-time setup
npm run screenshots                      # seed + capture golden shots
```

---

## Docs

- [ARCHITECTURE.md](ARCHITECTURE.md) — transport story, split-brain fix, ecosystem fit, Phase 0 boundaries
- [COMPAT.md](COMPAT.md) — version coupling, contract versions
- [docs/API.md](docs/API.md) — full endpoint reference

---

## What This Repo Is Not

- Not the governor kernel (that's [Agent Governor](https://github.com/unpingable/agent_governor))
- Not an IDE integration (that's the [VS Code extension](https://github.com/unpingable/vscode-governor))
- Not a place where chat becomes canon by accident

It's a cockpit for a system where language is a proposal and only receipts earn writes.
