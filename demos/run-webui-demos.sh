#!/usr/bin/env bash
# run-webui-demos.sh — Generate Playwright specs from Python demo scenarios and run them.
#
# Usage:
#   bash demos/run-webui-demos.sh           # Generate specs + run tests
#   bash demos/run-webui-demos.sh --gen     # Generate specs only (no Playwright run)
#
# The Python model (webui_demo.py) is the single source of truth.
# This script generates .spec.ts files, then runs Playwright.

set -euo pipefail
cd "$(dirname "$0")"

GEN_ONLY=false
if [[ "${1:-}" == "--gen" ]]; then
  GEN_ONLY=true
fi

echo "==> Generating Playwright specs from demo scenarios..."

python3 -c "
from governor.webui_demo import BUILTIN_DEMOS, generate_playwright_spec
import pathlib

out_dir = pathlib.Path('.')
for demo in BUILTIN_DEMOS:
    if demo.surface.value != 'webui':
        continue
    spec = generate_playwright_spec(demo)
    path = out_dir / f'{demo.name}.spec.ts'
    path.write_text(spec)
    print(f'  Generated: {path}')
"

echo "==> Generated $(ls -1 *.spec.ts 2>/dev/null | wc -l) spec file(s)."

if $GEN_ONLY; then
  echo "==> --gen mode: skipping Playwright execution."
  exit 0
fi

echo "==> Running Playwright tests..."
npx playwright test --config=playwright.config.ts
