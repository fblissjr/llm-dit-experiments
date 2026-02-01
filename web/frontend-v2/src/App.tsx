/**
 * App Root Component
 *
 * Main application component that initializes stores and renders
 * the two-column layout with form and result display.
 */

import { useEffect } from 'react';
import { useAppStore, useFormStore, useSessionStore } from '@/stores';
import { useAppShortcuts, useIsDesktop } from '@/hooks';
import { AppShell } from '@/components/layout';
import { PipelineSelector, PipelineForm } from '@/components/pipeline';
import { GenerateButton, ProgressDisplay, ResultDisplay } from '@/components/generation';

function PipelineView() {
  const selectedPipelineId = useAppStore((s) => s.selectedPipelineId);
  const toggleHistory = useAppStore((s) => s.toggleHistory);
  const isDesktop = useIsDesktop();

  const resetPipeline = useFormStore((s) => s.resetPipeline);
  const validate = useFormStore((s) => s.validate);

  const startGeneration = useSessionStore((s) => s.startGeneration);

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
        <PipelineSelector />
        <PipelineForm />

        {/* Progress display */}
        <ProgressDisplay />

        {/* Generate button */}
        <GenerateButton />

        {/* Keyboard hints */}
        <div className="text-xs text-gray-500 text-center">
          <kbd className="px-1.5 py-0.5 bg-gray-800 rounded">Ctrl+Enter</kbd> to generate
          {' | '}
          <kbd className="px-1.5 py-0.5 bg-gray-800 rounded">Ctrl+H</kbd> for history
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

  // Refresh VRAM periodically
  useEffect(() => {
    const refreshVRAM = useAppStore.getState().refreshVRAM;
    const interval = setInterval(refreshVRAM, 30000); // Every 30s
    return () => clearInterval(interval);
  }, []);

  return (
    <AppShell>
      {isLoading ? (
        <LoadingState />
      ) : error ? (
        <ErrorState error={error} />
      ) : (
        <PipelineView />
      )}
    </AppShell>
  );
}
