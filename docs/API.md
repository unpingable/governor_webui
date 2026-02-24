# API Reference

Full endpoint reference for Phosphor. All endpoints return JSON unless noted.

Auth: mutating endpoints (POST/PUT/DELETE/PATCH) require `Authorization: Bearer <token>` when `GOVERNOR_AUTH_TOKEN` is set. Read-only endpoints are always open.

## Chat (OpenAI-compatible)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (streaming via SSE supported) |
| `/v1/models` | GET | List available models |
| `/v1/backends` | GET | List backends with availability |
| `/v1/backends/switch` | POST | Switch backend at runtime |

## Sessions

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/sessions/` | GET | List all sessions |
| `/sessions/` | POST | Create new session |
| `/sessions/{id}` | GET | Get session with messages |
| `/sessions/{id}` | DELETE | Delete session |
| `/sessions/{id}` | PATCH | Update session title |
| `/sessions/{id}/messages` | POST | Append message |

## Governor State

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

## Fiction Mode

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/governor/fiction/characters` | GET | List characters |
| `/governor/fiction/characters` | POST | Add character |
| `/governor/fiction/world-rules` | GET | List world rules |
| `/governor/fiction/world-rules` | POST | Add world rule |
| `/governor/fiction/forbidden` | GET | List forbidden things |
| `/governor/fiction/forbidden` | POST | Add forbidden thing |
| `/governor/fiction/capture/scan` | POST | Scan text for canon captures |
| `/governor/fiction/captures` | GET | List pending captures |
| `/governor/fiction/capture/{id}/accept` | POST | Promote capture to canon |
| `/governor/fiction/capture/{id}/reject` | POST | Dismiss capture |

## Code Mode

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/governor/code/decisions` | GET | List decisions |
| `/governor/code/decisions` | POST | Add decision |
| `/governor/code/constraints` | GET | List constraints |
| `/governor/code/constraints` | POST | Add constraint |
| `/governor/code/compare` | POST | Run code interferometry |

## Research Mode

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/governor/research/state` | GET | Full research state + ED score |
| `/governor/research/claims` | POST | Add claim |
| `/governor/research/claims/{id}` | DELETE | Delete claim |
| `/governor/research/claims/{id}/status` | PATCH | Update claim status |
| `/governor/research/assumptions` | POST | Add assumption |
| `/governor/research/assumptions/{id}` | DELETE | Delete assumption |
| `/governor/research/assumptions/{id}/status` | PATCH | Update assumption status |
| `/governor/research/uncertainties` | POST | Add uncertainty |
| `/governor/research/uncertainties/{id}` | DELETE | Delete uncertainty |
| `/governor/research/uncertainties/{id}/status` | PATCH | Update uncertainty status |
| `/governor/research/links` | POST | Add typed link between items |
| `/governor/research/links/{id}` | DELETE | Delete link |
| `/governor/research/capture/scan` | POST | Scan text for claims + source refs |
| `/governor/research/captures` | GET | List pending research captures |
| `/governor/research/capture/{id}/accept` | POST | Promote capture to claim ledger |
| `/governor/research/capture/{id}/reject` | POST | Dismiss capture |

## V2 Dashboard (Run-Centric)

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

## Intent Compiler

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v2/intent/templates` | GET | List form templates |
| `/v2/intent/schema/{name}` | GET | Build form schema for current mode |
| `/v2/intent/validate` | POST | Validate response against schema |
| `/v2/intent/compile` | POST | Compile to intent + constraints (emits receipt) |
| `/v2/intent/policy` | GET | Current form policy |

## Health

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Backend + governor connectivity |
| `/api/info` | GET | Version, endpoints, capabilities |
