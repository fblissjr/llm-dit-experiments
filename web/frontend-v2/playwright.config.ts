/**
 * Playwright E2E Test Configuration
 *
 * last updated: 2026-03-02
 *
 * Runs against the Vite dev server which proxies API requests to the backend.
 * Requires the backend (web/server.py) to be running for schema-driven tests.
 *
 * Usage:
 *   cd web/frontend-v2
 *   bunx playwright test                    # Run all tests
 *   bunx playwright test --headed           # Run with visible browser
 *   bunx playwright test --ui               # Interactive UI mode
 *   bunx playwright test tests/e2e/schema   # Run specific test dir
 */

import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'list',
  timeout: 30_000,

  use: {
    baseURL: 'http://localhost:5175',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],

  // Start Vite dev server before tests (backend must already be running)
  webServer: {
    command: 'bun run dev',
    port: 5175,
    reuseExistingServer: !process.env.CI,
    timeout: 15_000,
  },
});
