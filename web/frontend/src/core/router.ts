/**
 * Simple Router
 *
 * Handles URL state for pipeline switching.
 * Uses browser history API for SPA-like navigation.
 *
 * last updated: 2026-01-25
 */

import { currentPipeline, pipelineSchemas, selectPipeline } from "./state.ts";

/**
 * Initialize router - read initial state from URL
 */
export function initRouter(): void {
  // Read pipeline from URL on initial load
  const url = new URL(window.location.href);
  const pipelineId = url.searchParams.get("pipeline");

  if (pipelineId) {
    // Wait for schemas to load, then select the pipeline
    const unsubscribe = pipelineSchemas.subscribe((schemas) => {
      if (Object.keys(schemas).length > 0) {
        if (pipelineId in schemas) {
          selectPipeline(pipelineId);
        } else {
          // Invalid pipeline ID, select first available
          const firstPipeline = Object.keys(schemas)[0];
          if (firstPipeline) {
            selectPipeline(firstPipeline);
          }
        }
        unsubscribe();
      }
    });
  }

  // Handle browser back/forward buttons
  window.addEventListener("popstate", handlePopState);
}

/**
 * Handle popstate events (browser back/forward)
 */
function handlePopState(): void {
  const url = new URL(window.location.href);
  const pipelineId = url.searchParams.get("pipeline");

  if (pipelineId && pipelineId !== currentPipeline.get()) {
    const schemas = pipelineSchemas.get();
    if (pipelineId in schemas) {
      selectPipeline(pipelineId);
    }
  }
}

/**
 * Navigate to a pipeline (updates URL)
 */
export function navigateToPipeline(pipelineId: string): void {
  const schemas = pipelineSchemas.get();
  if (!(pipelineId in schemas)) {
    console.error(`Pipeline "${pipelineId}" not found`);
    return;
  }

  // Update URL
  const url = new URL(window.location.href);
  url.searchParams.set("pipeline", pipelineId);
  window.history.pushState({ pipeline: pipelineId }, "", url.toString());

  // Update state
  selectPipeline(pipelineId);
}

/**
 * Get current URL parameters as object
 */
export function getURLParams(): Record<string, string> {
  const url = new URL(window.location.href);
  const params: Record<string, string> = {};
  url.searchParams.forEach((value, key) => {
    params[key] = value;
  });
  return params;
}

/**
 * Update URL parameters without navigation
 */
export function updateURLParams(params: Record<string, string>): void {
  const url = new URL(window.location.href);
  for (const [key, value] of Object.entries(params)) {
    url.searchParams.set(key, value);
  }
  window.history.replaceState({}, "", url.toString());
}
