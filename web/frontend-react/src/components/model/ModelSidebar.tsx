/**
 * Model Sidebar
 *
 * Collapsible right sidebar showing VRAM status and model management.
 * Desktop: slides in from right, persists open state.
 * Mobile: delegates to existing modal behavior.
 *
 * Features:
 * - Keyboard shortcut: M (to toggle)
 * - VRAM estimation warning before loading
 */

import { useState, useCallback } from 'react';
import { useModelStore } from '@/stores/modelStore';
import { usePipelineStore } from '@/stores/pipelineStore';
import { useUIStore } from '@/stores/uiStore';
import { formatVRAM, getUtilizationLevel } from '@/types/model';
import type { PipelineColor, VRAMEstimate } from '@/types';
import { PIPELINE_COLOR_CLASSES } from '@/types';
import { VRAMWarningDialog } from './VRAMWarningDialog';

export function ModelSidebar() {
  const {
    vram,
    modelStatus,
    isLoadingModel,
    loadingPipelineId,
    loadModel,
    unloadModel,
    unloadAll,
    estimateVRAM,
  } = useModelStore();
  const { pipelines } = usePipelineStore();
  const { isModelPanelOpen, toggleModelPanel, isDesktop } = useUIStore();

  // VRAM warning dialog state
  const [warningState, setWarningState] = useState<{
    isOpen: boolean;
    pipelineId: string | null;
    pipelineName: string;
    estimate: VRAMEstimate | null;
  }>({
    isOpen: false,
    pipelineId: null,
    pipelineName: '',
    estimate: null,
  });

  // Handle load with VRAM check
  const handleLoadModel = useCallback(async (pipelineId: string, pipelineName: string) => {
    // First, estimate VRAM requirements
    const estimate = await estimateVRAM(pipelineId);

    if (estimate && !estimate.wouldFit) {
      // Show warning dialog
      setWarningState({
        isOpen: true,
        pipelineId,
        pipelineName,
        estimate,
      });
    } else {
      // Fits or no estimate available - proceed with load
      loadModel(pipelineId);
    }
  }, [estimateVRAM, loadModel]);

  // Confirm load despite warning
  const handleConfirmLoad = useCallback(() => {
    if (warningState.pipelineId) {
      loadModel(warningState.pipelineId);
    }
    setWarningState({ isOpen: false, pipelineId: null, pipelineName: '', estimate: null });
  }, [warningState.pipelineId, loadModel]);

  // Cancel load
  const handleCancelLoad = useCallback(() => {
    setWarningState({ isOpen: false, pipelineId: null, pipelineName: '', estimate: null });
  }, []);

  // Desktop only - mobile uses the existing ModelManagementPanel modal
  if (!isDesktop) return null;
  if (!isModelPanelOpen) return null;

  const level = vram ? getUtilizationLevel(vram.utilizationPercent) : 'low';
  const levelColors = {
    low: 'text-green-500',
    medium: 'text-yellow-500',
    high: 'text-orange-500',
    critical: 'text-red-500',
  };

  const loadedPipelines = Object.keys(modelStatus).filter(
    (id) => modelStatus[id].status === 'loaded'
  );

  const unloadedPipelines = Object.values(pipelines).filter(
    (p) => !loadedPipelines.includes(p.id)
  );

  return (
    <aside
      className={`
        fixed right-0 top-[calc(theme(spacing.16)+1px)] bottom-0 z-30
        w-80 bg-gray-800 border-l border-gray-700
        overflow-y-auto
        transform transition-transform duration-200 ease-in-out
        ${isModelPanelOpen ? 'translate-x-0' : 'translate-x-full'}
      `}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-700 sticky top-0 bg-gray-800 z-10">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-medium text-gray-200 uppercase tracking-wide">
            Models
          </h2>
          <kbd className="px-1.5 py-0.5 text-xs bg-gray-700 rounded text-gray-400">
            M
          </kbd>
        </div>
        <button
          onClick={toggleModelPanel}
          className="p-1 text-gray-400 hover:text-gray-200 transition-colors"
          title="Close sidebar (M)"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Content */}
      <div className="p-4 space-y-5">
        {/* VRAM Overview */}
        {vram && (
          <section className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-gray-400 uppercase tracking-wide">VRAM</span>
              <span className={`text-sm font-medium ${levelColors[level]}`}>
                {formatVRAM(vram.usedMB)} / {formatVRAM(vram.totalMB)}
              </span>
            </div>

            {/* Segmented bar showing breakdown */}
            <div className="h-3 bg-gray-700 rounded-full overflow-hidden flex">
              {vram.breakdown.map((segment, i) => (
                <div
                  key={i}
                  className="h-full transition-all duration-300"
                  style={{
                    width: `${(segment.sizeMB / vram.totalMB) * 100}%`,
                    backgroundColor: segment.color,
                  }}
                  title={`${segment.label}: ${formatVRAM(segment.sizeMB)}`}
                />
              ))}
            </div>

            {/* Compact legend */}
            <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs">
              {vram.breakdown.map((segment, i) => (
                <div key={i} className="flex items-center gap-1">
                  <span
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: segment.color }}
                  />
                  <span className="text-gray-400">{segment.label}</span>
                </div>
              ))}
              <div className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-gray-600" />
                <span className="text-gray-400">Free {formatVRAM(vram.freeMB)}</span>
              </div>
            </div>
          </section>
        )}

        {/* Loaded Models */}
        <section className="space-y-2">
          <h3 className="text-xs font-medium text-gray-400 uppercase tracking-wide">
            Loaded ({loadedPipelines.length})
          </h3>

          {loadedPipelines.length === 0 ? (
            <p className="text-sm text-gray-500 py-2">No models loaded</p>
          ) : (
            <div className="space-y-2">
              {loadedPipelines.map((pipelineId) => {
                const pipeline = pipelines[pipelineId];
                const status = modelStatus[pipelineId];
                const color = (pipeline?.color ?? 'blue') as PipelineColor;

                return (
                  <div
                    key={pipelineId}
                    className="p-3 bg-gray-900 rounded-lg"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        {pipeline?.icon && <span>{pipeline.icon}</span>}
                        <span className={`font-medium text-sm ${PIPELINE_COLOR_CLASSES.text[color]}`}>
                          {pipeline?.name ?? pipelineId}
                        </span>
                      </div>
                      <button
                        onClick={() => unloadModel(pipelineId)}
                        className="text-xs px-2 py-1 rounded bg-gray-700 text-gray-400 hover:bg-red-900/50 hover:text-red-400 transition-colors"
                      >
                        Unload
                      </button>
                    </div>

                    {/* Component breakdown - compact */}
                    {status?.components && status.components.length > 0 && (
                      <div className="flex flex-wrap gap-1 text-xs">
                        {status.components.map((comp) => (
                          <span
                            key={comp.name}
                            className="px-1.5 py-0.5 bg-gray-800 rounded text-gray-400"
                            title={`${comp.name}: ${formatVRAM(comp.vramMB)}`}
                          >
                            {comp.name}
                          </span>
                        ))}
                      </div>
                    )}

                    {status?.totalVramMB && (
                      <div className="text-xs text-gray-500 mt-1">
                        {formatVRAM(status.totalVramMB)}
                        {status.loadTimeMs && ` in ${(status.loadTimeMs / 1000).toFixed(1)}s`}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </section>

        {/* Available Models */}
        {unloadedPipelines.length > 0 && (
          <section className="space-y-2">
            <h3 className="text-xs font-medium text-gray-400 uppercase tracking-wide">
              Available
            </h3>

            <div className="space-y-1">
              {unloadedPipelines.map((pipeline) => {
                const isLoading = loadingPipelineId === pipeline.id;

                return (
                  <button
                    key={pipeline.id}
                    onClick={() => handleLoadModel(pipeline.id, pipeline.name)}
                    disabled={isLoadingModel}
                    className={`
                      w-full p-2 text-left rounded-lg border border-gray-700
                      hover:border-gray-600 hover:bg-gray-700/50
                      disabled:opacity-50 disabled:cursor-not-allowed
                      transition-colors flex items-center justify-between
                    `}
                  >
                    <div className="flex items-center gap-2">
                      {pipeline.icon && <span>{pipeline.icon}</span>}
                      <span className="font-medium text-sm">{pipeline.name}</span>
                    </div>
                    {isLoading ? (
                      <span className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                    ) : (
                      <span className="text-xs text-gray-500">Load</span>
                    )}
                  </button>
                );
              })}
            </div>
          </section>
        )}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-gray-700 sticky bottom-0 bg-gray-800">
        <button
          onClick={unloadAll}
          disabled={loadedPipelines.length === 0}
          className="w-full py-2 text-sm rounded-lg border border-gray-600 text-gray-400
                     hover:bg-gray-700 hover:text-gray-200
                     disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          Unload All
        </button>
      </div>

      {/* VRAM Warning Dialog */}
      {warningState.estimate && (
        <VRAMWarningDialog
          isOpen={warningState.isOpen}
          estimate={warningState.estimate}
          pipelineName={warningState.pipelineName}
          onConfirm={handleConfirmLoad}
          onCancel={handleCancelLoad}
        />
      )}
    </aside>
  );
}
