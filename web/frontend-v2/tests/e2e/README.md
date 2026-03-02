# playwright browser e2e tests

*last updated: 2026-03-02*

Browser-based end-to-end tests for the React frontend using Playwright. These validate UI behavior against a running backend + frontend -- they are NOT API tests (those live in `tests/e2e/api/` and use `TestClient`).

## prerequisites

1. **Backend running:** `uv run web/server.py --config config.toml` from the project root
2. **Playwright installed:** `cd web/frontend-v2 && bunx playwright install chromium`

The Vite dev server starts automatically via `playwright.config.ts` `webServer` config.

## running tests

All commands from `web/frontend-v2/`:

```bash
# Headless (default, CI-friendly)
bun run test:e2e

# Visible browser (for debugging)
bun run test:e2e:headed

# Interactive Playwright UI (best for development)
bun run test:e2e:ui

# Run a specific test file
bunx playwright test tests/e2e/schema/model_status_panel.spec.ts

# Run a specific test by name
bunx playwright test -g "SLG start and stop sliders"
```

## configuration

Config file: `web/frontend-v2/playwright.config.ts`

| Setting | Value | Notes |
|---------|-------|-------|
| Browser | Chromium only | Add Firefox/WebKit projects if needed |
| Base URL | `http://localhost:5175` | Vite dev server |
| Timeout (action) | 30s | Per-action timeout |
| Retries | 0 (local), 2 (CI) | Controlled by `CI` env var |
| Traces | On first retry | For debugging failures |
| Screenshots | On failure only | Saved to `test-results/` |
| Web server | `bun run dev` on :5175 | Auto-starts, reuses existing in local |

## test structure

```
tests/e2e/
├── README.md              # This file
└── schema/                # Schema-driven UI behavior tests
    ├── model_status_panel.spec.ts    # ModelManager read-only panel
    ├── pipeline_schema.spec.ts       # ParamSchema rendering (SLG, compile, etc.)
    └── settings_menu.spec.ts         # SettingsMenu server actions
```

### current test suites

**model_status_panel.spec.ts** -- Verifies ModelManager component is read-only:
- No Load/Unload buttons exist
- Pipeline names are visible
- Status badges (loaded/unloaded) are visible
- Refresh Status button exists

**pipeline_schema.spec.ts** -- Verifies schema changes render correctly:
- Z-Image: SLG start/stop sliders visible in Advanced group (not hidden by broken conditionals)
- Z-Image: compile checkbox removed from Optimization group
- LTX-2: compile checkbox removed from Optimization group
- Both: FBCache/FP8/offload controls still present

**settings_menu.spec.ts** -- Verifies SettingsMenu server actions:
- Unload All Models button exists
- Confirmation dialog appears on click
- Cancel closes the dialog
- Clear CUDA Cache and Restart Server buttons exist

## writing new tests

### file naming

- `tests/e2e/<category>/<feature>.spec.ts`
- Categories: `schema/` (form rendering), add others as needed (e.g., `generation/`, `history/`)

### template

```typescript
/**
 * Feature Name E2E Tests
 *
 * last updated: YYYY-MM-DD
 *
 * Brief description of what these tests verify.
 */

import { test, expect } from '@playwright/test';

test.describe('Feature Name', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Wait for app to initialize
    await page.waitForTimeout(2000);
  });

  test('descriptive test name', async ({ page }) => {
    // Arrange: navigate to the right state
    // Act: interact with the UI
    // Assert: verify expected behavior
    const element = page.getByRole('button', { name: /Expected Text/i });
    await expect(element).toBeVisible();
  });
});
```

### helper patterns

**Select a pipeline:**
```typescript
async function selectPipeline(page: Page, tab: 'Image' | 'Video', name: string) {
  const tabButton = page.getByRole('button', { name: new RegExp(`^${tab}$`, 'i') });
  if (await tabButton.isVisible()) await tabButton.click();

  const pipeline = page.locator(`button:has-text("${name}")`).first();
  if (await pipeline.isVisible().catch(() => false)) {
    await pipeline.click();
    await page.waitForTimeout(500);
  }
}
```

**Expand a collapsible group:**
```typescript
async function expandGroup(page: Page, groupName: string) {
  const header = page.locator(`button.section-header:has-text("${groupName}")`);
  if (await header.isVisible().catch(() => false)) {
    const parent = header.locator('..');
    const content = parent.locator('.section-content');
    const maxHeight = await content.evaluate((el) =>
      window.getComputedStyle(el).maxHeight
    ).catch(() => '0px');
    if (maxHeight === '0px') {
      await header.click();
      await page.waitForTimeout(300);
    }
  }
}
```

**Check if a form label exists:**
```typescript
async function hasParamLabel(page: Page, label: string): Promise<boolean> {
  return (await page.locator(`label.form-label:has-text("${label}")`).count()) > 0;
}
```

### selector best practices

Prefer stable selectors in this order:
1. **Roles:** `page.getByRole('button', { name: /Text/i })` -- accessible, stable
2. **Text:** `page.locator('text=Exact Text')` -- visible content
3. **Test IDs:** `page.getByTestId('my-id')` -- requires `data-testid` in source
4. **CSS class patterns:** `page.locator('[class*="model-manager"]')` -- fragile, last resort

Avoid:
- Exact class names (change with Tailwind updates)
- DOM structure assumptions (fragile across refactors)
- `nth()` selectors without context (order-dependent)

### what to test vs not test

**Good candidates for Playwright tests:**
- Schema rendering: controls visible/hidden based on ParamSchema changes
- Component state: buttons present/absent, dialogs open/close
- Navigation: pipeline selection, tab switching, settings menu
- Visual regressions: layout changes, responsive behavior

**Use pytest API E2E tests instead for:**
- Backend logic (parameter resolution, model loading, generation)
- API response validation
- Error handling for invalid requests

**Use Vitest unit tests instead for:**
- Store logic (Zustand actions, selectors)
- Utility functions
- Component rendering in isolation

## debugging

```bash
# Run with visible browser and pause on failure
bunx playwright test --headed --debug

# Run with trace viewer
bunx playwright test --trace on
bunx playwright show-trace test-results/*/trace.zip

# Generate test from interactions
bunx playwright codegen http://localhost:5175
```

## ci integration

The config supports CI via environment variable:

```bash
CI=true bun run test:e2e
```

In CI mode:
- Retries: 2 (vs 0 local)
- Workers: 1 (sequential)
- Does NOT reuse existing server (starts fresh)
- `forbidOnly: true` (fails if `test.only()` is left in)
