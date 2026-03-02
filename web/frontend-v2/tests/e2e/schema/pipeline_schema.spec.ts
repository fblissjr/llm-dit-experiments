/**
 * Pipeline Schema E2E Tests
 *
 * last updated: 2026-03-02
 *
 * Verifies schema-driven form rendering: SLG controls visible for Z-Image,
 * compile checkboxes removed from LTX-2 and Z-Image. Tests run against
 * the live frontend with the backend serving pipeline schemas.
 */

import { test, expect, type Page } from '@playwright/test';

/**
 * Select a pipeline tab and pipeline within the app.
 * Clicks the tab (Image/Video) then the pipeline name.
 */
async function selectPipeline(page: Page, tab: 'Image' | 'Video', pipelineName: string) {
  // Click the tab
  const tabButton = page.getByRole('button', { name: new RegExp(`^${tab}$`, 'i') });
  if (await tabButton.isVisible()) {
    await tabButton.click();
  }

  // Click the pipeline name in the tab bar
  const pipelineButton = page.locator(`button:has-text("${pipelineName}")`).first();
  if (await pipelineButton.isVisible().catch(() => false)) {
    await pipelineButton.click();
    await page.waitForTimeout(500); // Wait for form to render
  }
}

/**
 * Expand a collapsible parameter group by clicking its header.
 */
async function expandGroup(page: Page, groupName: string) {
  const groupHeader = page.locator(`button.section-header:has-text("${groupName}")`);
  if (await groupHeader.isVisible().catch(() => false)) {
    // Check if already expanded by looking at the content sibling
    const parent = groupHeader.locator('..');
    const content = parent.locator('.section-content');
    const maxHeight = await content.evaluate((el) =>
      window.getComputedStyle(el).maxHeight
    ).catch(() => '0px');

    if (maxHeight === '0px') {
      await groupHeader.click();
      await page.waitForTimeout(300);
    }
  }
}

/**
 * Check if a parameter label exists in the form (visible after expanding its group).
 */
async function hasParamLabel(page: Page, labelText: string): Promise<boolean> {
  const label = page.locator(`label.form-label:has-text("${labelText}")`);
  return (await label.count()) > 0;
}

test.describe('Z-Image schema', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForTimeout(2000);
    await selectPipeline(page, 'Image', 'Z-Image');
  });

  test('SLG start and stop sliders are visible in Advanced group', async ({ page }) => {
    await expandGroup(page, 'Advanced');

    // SLG Scale should be visible
    const slgScale = await hasParamLabel(page, 'SLG Scale');
    expect(slgScale).toBe(true);

    // SLG Start and SLG Stop should ALSO be visible (no broken conditional hiding them)
    const slgStart = await hasParamLabel(page, 'SLG Start');
    const slgStop = await hasParamLabel(page, 'SLG Stop');
    expect(slgStart).toBe(true);
    expect(slgStop).toBe(true);
  });

  test('compile checkbox is NOT present in Optimization group', async ({ page }) => {
    await expandGroup(page, 'Optimization');

    // "Torch Compile" checkbox should NOT exist
    const hasCompile = await hasParamLabel(page, 'Torch Compile');
    expect(hasCompile).toBe(false);
  });

  test('FBCache controls are present in Optimization group', async ({ page }) => {
    await expandGroup(page, 'Optimization');

    // These should still exist (not removed)
    const hasFBCache = await hasParamLabel(page, 'Enable FBCache');
    const hasCPUOffload = await hasParamLabel(page, 'CPU Offload');
    expect(hasFBCache).toBe(true);
    expect(hasCPUOffload).toBe(true);
  });
});

test.describe('LTX-2 schema', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForTimeout(2000);
    await selectPipeline(page, 'Video', 'LTX-2');
  });

  test('compile checkbox is NOT present in Optimization group', async ({ page }) => {
    await expandGroup(page, 'Optimization');

    // "Torch Compile" checkbox should NOT exist
    const hasCompile = await hasParamLabel(page, 'Torch Compile');
    expect(hasCompile).toBe(false);
  });

  test('FBCache and FP8 controls are present in Optimization group', async ({ page }) => {
    await expandGroup(page, 'Optimization');

    const hasFBCache = await hasParamLabel(page, 'FBCache Threshold');
    const hasFP8 = await hasParamLabel(page, 'Use FP8');
    const hasOffload = await hasParamLabel(page, 'Offload Strategy');
    expect(hasFBCache).toBe(true);
    expect(hasFP8).toBe(true);
    expect(hasOffload).toBe(true);
  });
});
