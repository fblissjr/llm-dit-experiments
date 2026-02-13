/**
 * App Root Component
 *
 * Main application component that initializes stores and renders
 * the two-column layout with form and result display.
 */

import { useEffect } from 'react';
import { useAppStore, useFormStore, useSessionStore } from '@/stores';
import { useAppShortcuts, useIsDesktop, useIsMobile } from '@/hooks';
import { AppShell } from '@/components/layout';
import { PipelineForm } from '@/components/pipeline';
import { GenerateButton, ProgressDisplay, ResultDisplay } from '@/components/generation';
import { ErrorBoundary } from '@/components/common/ErrorBoundary';

function PipelineView() {
  const selectedPipelineId = useAppStore((s) => s.selectedPipelineId);
  const getSelectedPipeline = useAppStore((s) => s.getSelectedPipeline);
  const getPipelineColor = useAppStore((s) => s.getPipelineColor);
  const toggleHistory = useAppStore((s) => s.toggleHistory);
  const toggleLeftNav = useAppStore((s) => s.toggleLeftNav);
  const isDesktop = useIsDesktop();
  const isMobile = useIsMobile();

  const resetPipeline = useFormStore((s) => s.resetPipeline);
  const validate = useFormStore((s) => s.validate);

  const startGeneration = useSessionStore((s) => s.startGeneration);

  const selectedPipeline = getSelectedPipeline();

  // Register keyboard shortcuts
  useAppShortcuts({
    onGenerate: () => {
      if (selectedPipelineId) {
        const errors = validate(selectedPipelineId);
        if (errors.length === 0) {
          startGeneration(selectedPipelineId);
        }
      }
    },
    onToggleHistory: toggleHistory,
    onToggleNav: isDesktop ? toggleLeftNav : undefined,
    onReset: () => {
      if (selectedPipelineId) {
        resetPipeline(selectedPipelineId);
      }
    },
  });

  return (
    <div className={isDesktop ? 'grid grid-cols-2 gap-8' : 'space-y-6'}>
      {/* Left column: Form */}
      <div className="space-y-6">
        {/* Pipeline header - show on mobile or when no left nav */}
        {isMobile && selectedPipeline && (
          <div>
            <h2
              className="text-xl font-semibold"
              style={{ color: getPipelineColor(selectedPipeline.id) }}
            >
              {selectedPipeline.name}
            </h2>
            <p className="text-sm text-gray-400 mt-1">{selectedPipeline.description}</p>
          </div>
        )}

        <PipelineForm />

        {/* Progress display */}
        <ProgressDisplay />

        {/* Generate button */}
        <GenerateButton />

        {/* Keyboard hints */}
        <div className="text-xs text-gray-500 text-center space-x-2">
          <span>
            <kbd className="px-1.5 py-0.5 bg-gray-800 rounded-sm">Ctrl+Enter</kbd> generate
          </span>
          <span>
            <kbd className="px-1.5 py-0.5 bg-gray-800 rounded-sm">Ctrl+H</kbd> history
          </span>
          {isDesktop && (
            <span>
              <kbd className="px-1.5 py-0.5 bg-gray-800 rounded-sm">Ctrl+B</kbd> nav
            </span>
          )}
        </div>
      </div>

      {/* Right column: Result */}
      <div>
        <ResultDisplay />
      </div>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="flex items-center justify-center min-h-[400px]">
      <div className="text-center">
        <div className="w-8 h-8 border-2 border-gray-400 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
        <p className="text-gray-400">Loading pipelines...</p>
      </div>
    </div>
  );
}

function ErrorState({ error }: { error: string }) {
  const initialize = useAppStore((s) => s.initialize);

  return (
    <div className="flex items-center justify-center min-h-[400px]">
      <div className="text-center max-w-md">
        <svg
          className="w-12 h-12 mx-auto mb-4 text-red-400"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
          />
        </svg>
        <h2 className="text-lg font-medium text-red-400 mb-2">Failed to load</h2>
        <p className="text-gray-400 mb-4">{error}</p>
        <button
          onClick={initialize}
          className="btn-secondary"
        >
          Retry
        </button>
      </div>
    </div>
  );
}

export default function App() {
  const isLoading = useAppStore((s) => s.isLoading);
  const error = useAppStore((s) => s.error);
  const initialize = useAppStore((s) => s.initialize);

  // Initialize on mount
  useEffect(() => {
    initialize();
  }, [initialize]);

  // Poll generation context (includes VRAM data) with setTimeout chaining.
  // setTimeout chaining ensures the next poll fires only after the previous
  // completes, preventing request pile-up when the server is busy (e.g.,
  // during generation).
  useEffect(() => {
    const { refreshContext, refreshVRAM } = useAppStore.getState();
    let contextTimeout: ReturnType<typeof setTimeout>;
    let vramTimeout: ReturnType<typeof setTimeout>;
    let cancelled = false;

    const pollContext = async () => {
      try { await refreshContext(); } catch { /* ignore */ }
      if (!cancelled) contextTimeout = setTimeout(pollContext, 15000);
    };
    const pollVRAM = async () => {
      try { await refreshVRAM(); } catch { /* ignore */ }
      if (!cancelled) vramTimeout = setTimeout(pollVRAM, 30000);
    };

    // Initial fetch then start chaining
    pollContext();
    pollVRAM();

    return () => {
      cancelled = true;
      clearTimeout(contextTimeout);
      clearTimeout(vramTimeout);
    };
  }, []);

  return (
    <AppShell>
      {isLoading ? (
        <LoadingState />
      ) : error ? (
        <ErrorState error={error} />
      ) : (
        <ErrorBoundary>
          <PipelineView />
        </ErrorBoundary>
      )}
    </AppShell>
  );
}
