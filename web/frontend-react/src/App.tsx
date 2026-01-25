import { useEffect } from 'react';
import { usePipelineStore } from './stores/pipelineStore';
import { useModelStore } from './stores/modelStore';
import { AppShell } from './components/layout/AppShell';
import { PipelineForm } from './components/pipeline/PipelineForm';
import { ResultDisplay } from './components/generation/ResultDisplay';
import { HistoryPanel } from './components/history/HistoryPanel';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';

export default function App() {
  const { fetchPipelines, isLoading: pipelinesLoading, error: pipelinesError } = usePipelineStore();
  const { fetchVRAMStatus } = useModelStore();

  // Global keyboard shortcuts
  useKeyboardShortcuts();

  // Fetch initial data on mount
  useEffect(() => {
    fetchPipelines();
    fetchVRAMStatus();

    // Poll VRAM status every 5 seconds when generating
    const interval = setInterval(() => {
      fetchVRAMStatus();
    }, 5000);

    return () => clearInterval(interval);
  }, [fetchPipelines, fetchVRAMStatus]);

  if (pipelinesLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Loading pipelines...</p>
        </div>
      </div>
    );
  }

  if (pipelinesError) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="card max-w-md text-center">
          <p className="text-red-400 mb-4">Failed to load pipelines</p>
          <p className="text-gray-500 text-sm mb-4">{pipelinesError}</p>
          <button
            onClick={() => fetchPipelines()}
            className="btn-primary"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <AppShell
      main={
        <div className="flex flex-col lg:flex-row gap-6">
          {/* Form Section */}
          <div className="flex-1 min-w-0">
            <PipelineForm />
          </div>

          {/* Result Section */}
          <div className="flex-1 min-w-0">
            <ResultDisplay />
          </div>
        </div>
      }
      sidebar={<HistoryPanel />}
    />
  );
}
