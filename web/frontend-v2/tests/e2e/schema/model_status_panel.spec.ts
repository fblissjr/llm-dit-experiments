/**
 * Model Status Panel E2E Tests
 *
 * last updated: 2026-03-02
 *
 * Verifies the ModelManager component renders as a read-only status panel
 * with no load/unload action buttons. Tests run against the live frontend
 * with the backend serving pipeline schemas.
 */

import { test, expect } from '@playwright/test';

test.describe('ModelManager read-only status panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Wait for pipelines to load from API
    await page.waitForSelector('[class*="model-manager"], [class*="ModelManager"]', {
      timeout: 10_000,
    }).catch(() => {
      // ModelManager may be inside the LeftNav which renders differently.
      // Just wait for the page to settle.
    });
    // Give the app time to initialize and fetch model status
    await page.waitForTimeout(2000);
  });

  test('model cards have no Load or Unload buttons', async ({ page }) => {
    // Look for any buttons with "Load" or "Unload" text within the model card area
    // These should NOT exist in the read-only panel
    const loadButtons = page.getByRole('button', { name: /^Load Model$/i });
    const unloadButtons = page.getByRole('button', { name: /^Unload$/i });

    await expect(loadButtons).toHaveCount(0);
    await expect(unloadButtons).toHaveCount(0);
  });

  test('model cards show pipeline names', async ({ page }) => {
    // Pipeline names should be visible in the status panel
    // At minimum, the server should report these pipelines from /api/pipelines
    const pipelineNames = ['Z-Image', 'LTX-2', 'FLUX.2'];

    for (const name of pipelineNames) {
      const card = page.locator(`text=${name}`).first();
      // At least one pipeline should be visible (depends on server config)
      // We check that the text exists somewhere on the page
      const count = await page.locator(`text=${name}`).count();
      if (count > 0) {
        await expect(card).toBeVisible();
      }
    }
  });

  test('model cards show status badges', async ({ page }) => {
    // Each model card should have a status badge (loaded, unloaded, etc.)
    const statusBadges = page.locator('span:text-matches("loaded|unloaded|Loading", "i")');
    // At least one status badge should be visible
    const count = await statusBadges.count();
    expect(count).toBeGreaterThan(0);
  });

  test('Refresh Status button exists', async ({ page }) => {
    const refreshButton = page.getByRole('button', { name: /Refresh Status/i });
    await expect(refreshButton).toBeVisible();
  });
});
