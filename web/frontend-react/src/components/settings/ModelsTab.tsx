/**
 * ModelsTab Component
 *
 * Full model management UI with:
 * - VRAM breakdown visualization
 * - All pipelines with load status
 * - Load/Unload controls with VRAM estimation
 * - Component-level detail (transformer, VAE, etc.)
 */

import { useState, useCallback } from 'react';
import { usePipelineStore } from '@/stores/pipelineStore';
import { useModelStore } from '@/stores/modelStore';
import { formatVRAM, getUtilizationLevel } from '@/types';
import type { LoadStatus, PipelineColor, VRAMStatus, PipelineModelStatus, VRAMEstimate, ComponentStatus } from '@/types';
import { PIPELINE_COLOR_CLASSES } from '@/types';

export function ModelsTab() {
  const { pipelines } = usePipelineStore();
  const {
    vram,
    modelStatus,
    loadModel,
    unloadModel,
    unloadAll,
    estimateVRAM,
    loadingPipelineId,
    isLoadingModel,
  } = useModelStore();

  const pipelineList = Object.values(pipelines);
  const hasLoadedModels = Object.values(modelStatus).some((s) => s.status === 'loaded');

  return (
    <div className="space-y-8">
      {/* VRAM Overview Section */}
      <section>
        <h2 className="text-lg font-semibold text-gray-100 mb-4">VRAM Overview</h2>
        <VRAMOverview vram={vram} />
      </section>

      {/* Quick Actions */}
      <section className="flex items-center gap-4">
        <button
          onClick={() => unloadAll()}
          disabled={!hasLoadedModels || isLoadingModel}
          className={`
            px-4 py-2 text-sm font-medium rounded-lg transition-colors
            ${
              hasLoadedModels && !isLoadingModel
                ? 'bg-red-600/20 text-red-400 hover:bg-red-600/30 border border-red-600/30'
                : 'bg-gray-700/50 text-gray-500 cursor-not-allowed border border-gray-700'
            }
          `}
        >
          Unload All Models
        </button>

        {hasLoadedModels && (
          <span className="text-sm text-gray-400">
            {Object.values(modelStatus).filter((s) => s.status === 'loaded').length} model(s) loaded
          </span>
        )}
      </section>

      {/* Pipeline List */}
      <section>
        <h2 className="text-lg font-semibold text-gray-100 mb-4">Pipelines</h2>
        <div className="space-y-4">
          {pipelineList.map((pipeline) => (
            <PipelineCard
              key={pipeline.id}
              id={pipeline.id}
              name={pipeline.name}
              description={pipeline.description}
              icon={pipeline.icon}
              color={pipeline.color}
              status={modelStatus[pipeline.id]}
              isLoading={loadingPipelineId === pipeline.id}
              onLoad={() => loadModel(pipeline.id)}
              onUnload={() => unloadModel(pipeline.id)}
              onEstimate={() => estimateVRAM(pipeline.id)}
              currentVRAM={vram}
            />
          ))}
        </div>
      </section>
    </div>
  );
}

// VRAM Overview with breakdown
function VRAMOverview({ vram }: { vram: VRAMStatus | null }) {
  if (!vram) {
    return (
      <div className="bg-gray-800 rounded-lg p-4 text-gray-400">
        Loading VRAM status...
      </div>
    );
  }

  const level = getUtilizationLevel(vram.utilizationPercent);
  const levelColors = {
    low: 'text-green-400',
    medium: 'text-yellow-400',
    high: 'text-orange-400',
    critical: 'text-red-400',
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6 space-y-4">
      {/* Usage bar */}
      <div>
        <div className="flex justify-between text-sm mb-2">
          <span className="text-gray-400">VRAM Usage</span>
          <span className={levelColors[level]}>
            {formatVRAM(vram.usedMB)} / {formatVRAM(vram.totalMB)} ({vram.utilizationPercent.toFixed(1)}%)
          </span>
        </div>
        <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all duration-300 ${
              level === 'critical' ? 'bg-red-500' :
              level === 'high' ? 'bg-orange-500' :
              level === 'medium' ? 'bg-yellow-500' :
              'bg-green-500'
            }`}
            style={{ width: `${Math.min(vram.utilizationPercent, 100)}%` }}
          />
        </div>
      </div>

      {/* Breakdown */}
      {vram.breakdown.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-gray-300 mb-2">Breakdown</h4>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
            {vram.breakdown.map((item, idx) => (
              <div
                key={idx}
                className="flex items-center gap-2 text-sm"
              >
                <span
                  className="w-3 h-3 rounded-full flex-shrink-0"
                  style={{ backgroundColor: item.color }}
                />
                <span className="text-gray-400 truncate">{item.label}</span>
                <span className="text-gray-300 ml-auto">{formatVRAM(item.sizeMB)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Free space */}
      <div className="pt-2 border-t border-gray-700 flex justify-between text-sm">
        <span className="text-gray-400">Free VRAM</span>
        <span className="text-green-400">{formatVRAM(vram.freeMB)}</span>
      </div>
    </div>
  );
}

// Individual pipeline card
interface PipelineCardProps {
  id: string;
  name: string;
  description?: string;
  icon?: string;
  color: PipelineColor;
  status?: PipelineModelStatus;
  isLoading: boolean;
  onLoad: () => void;
  onUnload: () => void;
  onEstimate: () => Promise<VRAMEstimate | null>;
  currentVRAM: VRAMStatus | null;
}

function PipelineCard({
  name,
  description,
  icon,
  color,
  status,
  isLoading,
  onLoad,
  onUnload,
  onEstimate,
  currentVRAM,
}: PipelineCardProps) {
  const [estimate, setEstimate] = useState<{
    requiredMB: number;
    wouldFit: boolean;
  } | null>(null);
  const [showEstimate, setShowEstimate] = useState(false);

  const loadStatus: LoadStatus = isLoading ? 'loading' : (status?.status ?? 'unloaded');
  const isLoaded = loadStatus === 'loaded';
  const isUnloaded = loadStatus === 'unloaded' || loadStatus === 'error';

  const handleLoadClick = useCallback(async () => {
    if (isLoaded || isLoading) return;

    // Get estimate first
    const est = await onEstimate();
    if (est && !est.wouldFit) {
      setEstimate(est);
      setShowEstimate(true);
      return;
    }

    onLoad();
  }, [isLoaded, isLoading, onEstimate, onLoad]);

  const confirmLoad = useCallback(() => {
    setShowEstimate(false);
    onLoad();
  }, [onLoad]);

  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <div className="flex items-start justify-between gap-4">
        {/* Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            {icon && <span className="text-xl">{icon}</span>}
            <h3 className={`font-medium ${PIPELINE_COLOR_CLASSES.text[color]}`}>
              {name}
            </h3>
            <StatusBadge status={loadStatus} />
          </div>
          {description && (
            <p className="text-sm text-gray-400 mt-1">{description}</p>
          )}

          {/* Components detail when loaded */}
          {status?.components && status.components.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {status.components.map((comp: ComponentStatus) => (
                <span
                  key={comp.name}
                  className="px-2 py-1 bg-gray-700/50 rounded text-xs text-gray-300"
                >
                  {comp.name}: {formatVRAM(comp.vramMB)}
                  {comp.quantization !== 'fp16' && ` (${comp.quantization})`}
                </span>
              ))}
            </div>
          )}

          {/* VRAM usage when loaded */}
          {status?.totalVramMB && status.totalVramMB > 0 && (
            <p className="text-sm text-gray-500 mt-2">
              Using {formatVRAM(status.totalVramMB)}
              {status.loadTimeMs && ` · Loaded in ${(status.loadTimeMs / 1000).toFixed(1)}s`}
            </p>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2 flex-shrink-0">
          {isLoaded && (
            <button
              onClick={onUnload}
              className="px-3 py-1.5 text-sm font-medium rounded-lg
                bg-gray-700 text-gray-300 hover:bg-gray-600 transition-colors"
            >
              Unload
            </button>
          )}
          {isUnloaded && (
            <button
              onClick={handleLoadClick}
              disabled={isLoading}
              className="px-3 py-1.5 text-sm font-medium rounded-lg
                bg-blue-600 text-white hover:bg-blue-500 transition-colors
                disabled:bg-gray-700 disabled:text-gray-500 disabled:cursor-not-allowed"
            >
              {isLoading ? 'Loading...' : 'Load'}
            </button>
          )}
          {loadStatus === 'loading' && (
            <div className="flex items-center gap-2 text-yellow-400">
              <span className="w-4 h-4 border-2 border-yellow-400 border-t-transparent rounded-full animate-spin" />
              <span className="text-sm">Loading...</span>
            </div>
          )}
        </div>
      </div>

      {/* VRAM warning dialog */}
      {showEstimate && estimate && (
        <div className="mt-4 p-3 bg-yellow-900/20 border border-yellow-600/30 rounded-lg">
          <div className="flex items-start gap-3">
            <span className="text-yellow-500 text-lg">⚠️</span>
            <div className="flex-1">
              <p className="text-sm text-yellow-300">
                This model requires <strong>{formatVRAM(estimate.requiredMB)}</strong>
                {currentVRAM && (
                  <> but only <strong>{formatVRAM(currentVRAM.freeMB)}</strong> is available</>
                )}.
              </p>
              <p className="text-xs text-yellow-400/70 mt-1">
                Loading may fail or cause other models to be evicted.
              </p>
              <div className="flex gap-2 mt-3">
                <button
                  onClick={confirmLoad}
                  className="px-3 py-1 text-sm font-medium rounded bg-yellow-600 text-white hover:bg-yellow-500"
                >
                  Load Anyway
                </button>
                <button
                  onClick={() => setShowEstimate(false)}
                  className="px-3 py-1 text-sm font-medium rounded bg-gray-700 text-gray-300 hover:bg-gray-600"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Status badge component
function StatusBadge({ status }: { status: LoadStatus }) {
  const styles: Record<LoadStatus, { bg: string; text: string; label: string }> = {
    loaded: { bg: 'bg-green-500/20', text: 'text-green-400', label: 'Loaded' },
    loading: { bg: 'bg-yellow-500/20', text: 'text-yellow-400', label: 'Loading' },
    unloaded: { bg: 'bg-gray-500/20', text: 'text-gray-400', label: 'Not Loaded' },
    error: { bg: 'bg-red-500/20', text: 'text-red-400', label: 'Error' },
  };

  const style = styles[status];

  return (
    <span className={`px-2 py-0.5 text-xs font-medium rounded ${style.bg} ${style.text}`}>
      {style.label}
    </span>
  );
}
