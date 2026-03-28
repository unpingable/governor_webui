# Screenshot Generation

Regenerates `docs/img/*` via Playwright with seeded fixture state.

## Setup (one-time)

```bash
npm install && npx playwright install
```

## Usage

```bash
npm run screenshots          # seed + capture golden shots
```

Or step by step:

```bash
npm run shots:seed           # import fixture state into running stack
npm run shots:run            # capture screenshots
```

Requires a running research stack (default `http://127.0.0.1:8003`).
Override with `SCREENSHOT_BASE_URL`.

## Files

- `seed.sh` — imports deterministic fixture state
- `fixtures/research_demo.json` — sidebar state (claims, anchors)
- `shots.spec.ts` — Playwright test specs for each screenshot
- `playwright.config.ts` — viewport, timeouts, base URL
