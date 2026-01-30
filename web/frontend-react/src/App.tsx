import { useEffect, useCallback, useRef } from 'react';
import { usePipelineStore } from './stores/pipelineStore';
import { useModelStore } from './stores/modelStore';
import { useGenerationStore } from './stores/generationStore';
import { initResponsiveState, setupResizeListener, useUIStore } from './stores/uiStore';
import { AppShell } from './components/layout/AppShell';
import { PipelineForm } from './components/pipeline/PipelineForm';
import { ResultDisplay } from './components/generation/ResultDisplay';
import { HistoryPanel } from './components/history/HistoryPanel';
import { SettingsPage } from './components/settings/SettingsPage';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';

export default function App() {
  const { fetchPipelines, pipelines, isLoading: pipelinesLoading, error: pipelinesError } = usePipelineStore();
  const { fetchVRAMStatus, fetchModelStatus } = useModelStore();
  const { currentView } = useUIStore();

  // Global keyboard shortcuts
  useKeyboardShortcuts();

  // Set up responsive state listener (with proper cleanup)
  useEffect(() => {
    initResponsiveState();
    return setupResizeListener();
  }, []);

  // Track polling interval ref for cleanup
  const vramPollingInterval = useRef<ReturnType<typeof setInterval> | null>(null);

  // Memoized polling start/stop functions
  const startVRAMPolling = useCallback(() => {
    if (vramPollingInterval.current) return; // Already polling
    fetchVRAMStatus(); // Immediate fetch
    vramPollingInterval.current = setInterval(fetchVRAMStatus, 3000);
  }, [fetchVRAMStatus]);

  const stopVRAMPolling = useCallback(() => {
    if (vramPollingInterval.current) {
      clearInterval(vramPollingInterval.current);
      vramPollingInterval.current = null;
    }
  }, []);

  // Fetch initial data on mount
  useEffect(() => {
    fetchPipelines();
    fetchVRAMStatus();
  }, [fetchPipelines, fetchVRAMStatus]);

  // Subscribe to generation status for VRAM polling
  // Only poll during generation to reduce unnecessary API calls
  useEffect(() => {
    let previousStatus: string | null = null;

    const unsubscribe = useGenerationStore.subscribe((state) => {
      const currentStatus = state.status;

      // Only react to status changes
      if (currentStatus === previousStatus) return;
      previousStatus = currentStatus;

      if (currentStatus === 'generating' || currentStatus === 'loading') {
        startVRAMPolling();
      } else {
        stopVRAMPolling();
        // One final fetch when generation completes
        if (currentStatus === 'completed' || currentStatus === 'error') {
          fetchVRAMStatus();
        }
      }
    });

    return () => {
      unsubscribe();
      stopVRAMPolling();
    };
  }, [startVRAMPolling, stopVRAMPolling, fetchVRAMStatus]);

  // Fetch model status for each pipeline after pipelines load
  useEffect(() => {
    const pipelineIds = Object.keys(pipelines);
    if (pipelineIds.length > 0) {
      // Check status for each pipeline
      pipelineIds.forEach((id) => {
        fetchModelStatus(id);
      });
    }
  }, [pipelines, fetchModelStatus]);

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

  // Render settings page or studio based on current view
  if (currentView === 'settings') {
    return <SettingsPage />;
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
