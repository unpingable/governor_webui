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

  test('01 — research home with sidebar', async ({ page }) => {
    await page.goto('/');
    await disableAnimations(page);

    // Wait for sidebar to populate (governor polling)
    await page.waitForTimeout(2000);

    await page.screenshot({
      path: 'docs/img/01_research_home.png',
      fullPage: false,
    });
  });

  test('02 — capture chip + pending drawer', async ({ page }) => {
    await page.goto('/');
    await disableAnimations(page);

    // If there are pending captures, click the pill to open drawer
    const pill = page.getByTestId('research-pending-pill');
    if (await pill.isVisible({ timeout: 3000 }).catch(() => false)) {
      await pill.click();
      await expect(page.getByTestId('capture-drawer')).toBeVisible();
    }

    await page.screenshot({
      path: 'docs/img/02_pending_drawer.png',
      fullPage: false,
    });
  });

  test('03 — why overlay expanded', async ({ page }) => {
    await page.goto('/');
    await disableAnimations(page);

    // Wait for seeded chat to render (assistant message with DOI refs)
    await page.waitForTimeout(2000);

    // Click the Why toggle if present on an assistant message
    const whyToggle = page.getByTestId('why-toggle').first();
    if (await whyToggle.isVisible({ timeout: 5000 }).catch(() => false)) {
      await whyToggle.click();
      await expect(page.getByTestId('why-body').first()).toBeVisible();
    }

    await page.screenshot({
      path: 'docs/img/03_why_overlay.png',
      fullPage: false,
    });
  });

  test('04 — violation modal (requires demo backend or seeded violation)', async ({ page }) => {
    // This shot requires either:
    // (a) a pending violation seeded into .governor/pending_violations.json, or
    // (b) a demo backend that triggers a violation on a known input.
    // Without either, the test passes but produces no screenshot.
    await page.goto('/');
    await disableAnimations(page);

    const modal = page.getByTestId('violation-modal');
    if (await modal.isVisible({ timeout: 2000 }).catch(() => false)) {
      await page.screenshot({
        path: 'docs/img/04_violation_modal.png',
        fullPage: false,
      });
    }
  });

  test('05 — dashboard overview (dark mode)', async ({ page }) => {
    // Dashboard in dark mode for visual variety
    await page.emulateMedia({ colorScheme: 'dark' });
    await page.goto('/dashboard');
    await disableAnimations(page);
    await page.waitForTimeout(1500);

    await page.screenshot({
      path: 'docs/img/05_dashboard.png',
      fullPage: false,
    });
  });

});
