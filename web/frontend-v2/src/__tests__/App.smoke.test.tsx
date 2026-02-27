/**
 * Smoke test -- verifies App renders without throwing.
 *
 * Catches import failures, runtime crashes, and major
 * incompatibilities after dependency upgrades.
 */

import { enableMapSet } from 'immer';
enableMapSet();

import { describe, it, expect, vi, beforeAll, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import App from '../App';

// Route-based fetch mock for all API calls made during App initialization
const MOCK_RESPONSES: Record<string, unknown> = {
  '/api/pipelines': { pipelines: {}, loaded_pipeline: null },
  '/api/vram/status': { used_gb: 0, total_gb: 24, free_gb: 24 },
  '/api/context': {
    loaded_pipeline: null,
    encoder_only_mode: false,
    rewriter_backend: null,
    vramUsedGb: 0,
    vramTotalGb: 24,
    pendingRestartFields: [],
  },
};

beforeAll(() => {
  vi.stubGlobal('fetch', vi.fn((url: string) => {
    // Match against known routes (strip query params)
    const path = url.split('?')[0];
    const body = MOCK_RESPONSES[path] ?? {};
    return Promise.resolve({
      ok: true,
      json: () => Promise.resolve(body),
    });
  }));
});

afterEach(() => cleanup());

describe('App', () => {
  it('renders without crashing', () => {
    render(<App />);
    // App starts in loading state before pipelines resolve
    expect(screen.getByText('Loading pipelines...')).toBeDefined();
  });
});
