import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: '.',
  timeout: 30_000,
  use: {
    baseURL: process.env.SCREENSHOT_BASE_URL || 'http://127.0.0.1:8003',
    screenshot: 'on',
    trace: 'retain-on-failure',
    viewport: { width: 1400, height: 900 },
  },
  retries: 0,
  reporter: [['list']],
});
