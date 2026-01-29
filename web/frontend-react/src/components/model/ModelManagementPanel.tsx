/**
 * Model Management Panel
 *
 * Expanded view showing detailed VRAM breakdown and model controls.
 */

import { useModelStore } from '@/stores/modelStore';
import { usePipelineStore } from '@/stores/pipelineStore';
import { useUIStore } from '@/stores/uiStore';
import { formatVRAM, getUtilizationLevel } from '@/types/model';

export function ModelManagementPanel() {
  const {
    vram,
    modelStatus,
    isLoadingModel,
    loadingPipelineId,
    loadModel,
    unloadModel,
    unloadAll,
  } = useModelStore();
  const { pipelines } = usePipelineStore();
  const { isModelPanelOpen, toggleModelPanel, isMobile, isDesktop } = useUIStore();

  // Desktop uses ModelSidebar instead
  if (isDesktop) return null;
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

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 z-40"
        onClick={toggleModelPanel}
      />

      {/* Panel */}
      <div
        className={`
          fixed z-50 bg-gray-800 border border-gray-700 shadow-2xl
          ${isMobile
            ? 'inset-x-4 bottom-4 rounded-2xl max-h-[80vh]'
            : 'top-20 right-4 w-96 rounded-xl'
          }
        `}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <h2 className="text-lg font-medium">Model Management</h2>
          <button
            onClick={toggleModelPanel}
            className="p-1 text-gray-400 hover:text-gray-200 transition-colors"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4 overflow-y-auto max-h-[60vh]">
          {/* VRAM Overview */}
          {vram && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">VRAM Usage</span>
                <span className={`text-sm font-medium ${levelColors[level]}`}>
                  {formatVRAM(vram.usedMB)} / {formatVRAM(vram.totalMB)}
                </span>
              </div>

              {/* Segmented bar showing breakdown */}
              <div className="h-4 bg-gray-700 rounded-full overflow-hidden flex">
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

              {/* Legend */}
              <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-gray-400">
                {vram.breakdown.map((segment, i) => (
                  <div key={i} className="flex items-center gap-1.5">
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: segment.color }}
                    />
                    <span>{segment.label}</span>
                    <span className="text-gray-500">{formatVRAM(segment.sizeMB)}</span>
                  </div>
                ))}
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-gray-600" />
                  <span>Free</span>
                  <span className="text-gray-500">{formatVRAM(vram.freeMB)}</span>
                </div>
              </div>
            </div>
          )}

          {/* Loaded Pipelines */}
          <div className="space-y-2">
            <h3 className="text-sm font-medium text-gray-300">Loaded Pipelines</h3>

            {loadedPipelines.length === 0 ? (
              <p className="text-sm text-gray-500 py-2">No models loaded</p>
            ) : (
              loadedPipelines.map((pipelineId) => {
                const pipeline = pipelines[pipelineId];
                const status = modelStatus[pipelineId];

                return (
                  <div
                    key={pipelineId}
                    className="p-3 bg-gray-900 rounded-lg space-y-2"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        {pipeline?.icon && <span>{pipeline.icon}</span>}
                        <span className="font-medium">
                          {pipeline?.name ?? pipelineId}
                        </span>
                        <span className="px-2 py-0.5 text-xs bg-green-500/20 text-green-400 rounded-full">
                          Loaded
                        </span>
                      </div>
                      <button
                        onClick={() => unloadModel(pipelineId)}
                        className="text-sm text-gray-400 hover:text-red-400 transition-colors"
                      >
                        Unload
                      </button>
                    </div>

                    {/* Component breakdown */}
                    {status?.components && status.components.length > 0 && (
                      <div className="grid grid-cols-2 gap-1 text-xs">
                        {status.components.map((comp) => (
                          <div
                            key={comp.name}
                            className="flex items-center justify-between px-2 py-1 bg-gray-800 rounded"
                          >
                            <span className="text-gray-400 capitalize">{comp.name}</span>
                            <span className="text-gray-300">
                              {formatVRAM(comp.vramMB)}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}

                    {status?.totalVramMB && (
                      <div className="text-xs text-gray-500">
                        Total: {formatVRAM(status.totalVramMB)}
                        {status.loadTimeMs && ` · Loaded in ${(status.loadTimeMs / 1000).toFixed(1)}s`}
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>

          {/* Available Pipelines */}
          <div className="space-y-2">
            <h3 className="text-sm font-medium text-gray-300">Available Pipelines</h3>

            <div className="grid grid-cols-2 gap-2">
              {Object.values(pipelines)
                .filter((p) => !loadedPipelines.includes(p.id))
                .map((pipeline) => (
                  <button
                    key={pipeline.id}
                    onClick={() => loadModel(pipeline.id)}
                    disabled={isLoadingModel}
                    className={`
                      p-3 text-left rounded-lg border border-gray-700
                      hover:border-gray-600 hover:bg-gray-700/50
                      disabled:opacity-50 disabled:cursor-not-allowed
                      transition-colors
                    `}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      {pipeline.icon && <span>{pipeline.icon}</span>}
                      <span className="font-medium text-sm">{pipeline.name}</span>
                    </div>
                    <span className="text-xs text-gray-500 line-clamp-1">
                      {pipeline.description}
                    </span>
                    {isLoadingModel && loadingPipelineId === pipeline.id && (
                      <div className="mt-2">
                        <div className="w-full h-1 bg-gray-700 rounded-full overflow-hidden">
                          <div className="h-full bg-blue-500 animate-pulse w-1/2" />
                        </div>
                      </div>
                    )}
                  </button>
                ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-700">
          <button
            onClick={unloadAll}
            disabled={loadedPipelines.length === 0}
            className="w-full btn-secondary text-sm disabled:opacity-50"
          >
            Unload All Models
          </button>
        </div>
      </div>
    </>
  );
}
