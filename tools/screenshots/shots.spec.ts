/**
 * Screenshot generation for README/docs.
 *
 * Golden flows only — expand only when you must.
 * Requires: docker-compose up + seed.sh before running.
 *
 * Usage:
 *   npm run shots:seed && npm run shots:run
 *   # or: npm run screenshots
 */
import { test, expect } from '@playwright/test';

// Kill animations so screenshots are deterministic
const disableAnimations = async (page: any) => {
  await page.addStyleTag({
    content: `*, *::before, *::after {
      animation: none !important;
      transition: none !important;
    }`,
  });
};

test.describe('README screenshots', () => {

  test('01 — research home: chips + Why overlay + sidebar', async ({ page }) => {
    await page.goto('/');
    await disableAnimations(page);

    // Wait for sidebar to populate (governor polling) + capture scan to complete
    await page.waitForTimeout(2500);

    await page.screenshot({
      path: 'docs/img/01_research_home.png',
      fullPage: false,
    });
  });

  // Pending drawer and Why overlay shots deferred — both need testid wiring
  // or deterministic state that the conditional guards can actually trigger.

  test('02 — violation modal (requires seeded violation)', async ({ page }) => {
    // This shot requires either:
    // (a) a pending violation seeded into .governor/pending_violations.json, or
    // (b) a demo backend that triggers a violation on a known input.
    // Without either, the test passes but produces no screenshot.
    await page.goto('/');
    await disableAnimations(page);

    const modal = page.getByTestId('violation-modal');
    if (await modal.isVisible({ timeout: 2000 }).catch(() => false)) {
      await page.screenshot({
        path: 'docs/img/02_violation_modal.png',
        fullPage: false,
      });
    }
  });

  test('03 — dashboard overview (dark mode)', async ({ page }) => {
    await page.emulateMedia({ colorScheme: 'dark' });
    await page.goto('/dashboard');
    await disableAnimations(page);
    await page.waitForTimeout(1500);

    await page.screenshot({
      path: 'docs/img/03_dashboard.png',
      fullPage: false,
    });
  });

});
