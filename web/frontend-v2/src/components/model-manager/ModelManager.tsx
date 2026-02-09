/**
 * ModelManager Component
 *
 * Displays model loading status for all pipelines and allows
 * loading/unloading models. Designed to work in both desktop
 * sidebar and mobile bottom sheet contexts.
 */

import { useEffect } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { cn } from '@/utils';
import { useAppStore } from '@/stores';

interface LoRABadgeInfo {
  name: string;
  scale: number;
}

interface ModelCardProps {
  pipelineId: string;
  name: string;
  color: string;
  status: 'unloaded' | 'loading' | 'loaded' | 'error';
  vramMb?: number;
  error?: string;
  modelVariant?: string | null;
  loras?: LoRABadgeInfo[];
  configTags?: { key: string; label: string; color: string }[];
  onLoad: () => void;
  onUnload: () => void;
}

function ModelCard({
  name,
  color,
  status,
  vramMb,
  error,
  modelVariant,
  loras,
  configTags,
  onLoad,
  onUnload,
}: ModelCardProps) {
  const isLoading = status === 'loading';
  const isLoaded = status === 'loaded';
  const hasError = status === 'error';

  return (
    <div
      className={cn(
        'p-3 rounded-lg border transition-colors',
        isLoaded
          ? 'bg-gray-800/80 border-gray-600'
          : 'bg-gray-800/40 border-gray-700'
      )}
    >
      {/* Header row */}
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="flex items-center gap-2 min-w-0">
          <div
            className="w-2 h-2 rounded-full flex-shrink-0"
            style={{
              backgroundColor: isLoaded ? color : '#6b7280',
            }}
          />
          <span className="text-sm font-medium truncate">{name}</span>
        </div>

        {/* Status badge */}
        <span
          className={cn(
            'text-xs px-2 py-0.5 rounded-full flex-shrink-0',
            isLoaded && 'bg-green-500/20 text-green-400',
            isLoading && 'bg-blue-500/20 text-blue-400',
            hasError && 'bg-red-500/20 text-red-400',
            status === 'unloaded' && 'bg-gray-500/20 text-gray-400'
          )}
        >
          {isLoading ? 'Loading...' : status}
        </span>
      </div>

      {/* Model variant */}
      {isLoaded && modelVariant && (
        <p className="text-xs text-gray-500 mb-1">{modelVariant}</p>
      )}

      {/* LoRA badges */}
      {isLoaded && loras && loras.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-1.5">
          {loras.map((lora) => (
            <span
              key={lora.name}
              className="inline-flex items-center gap-0.5 px-1.5 py-0.5 text-xs bg-purple-500/20 text-purple-300 rounded"
            >
              {lora.name}
              <span className="text-purple-400/70">@{lora.scale.toFixed(2)}</span>
            </span>
          ))}
        </div>
      )}

      {/* Config tags */}
      {isLoaded && configTags && configTags.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-1.5">
          {configTags.map((tag) => {
            const colorMap: Record<string, string> = {
              purple: 'bg-purple-500/20 text-purple-300',
              blue: 'bg-blue-500/20 text-blue-300',
              orange: 'bg-orange-500/20 text-orange-300',
              green: 'bg-green-500/20 text-green-300',
            };
            return (
              <span
                key={tag.key}
                className={cn(
                  'inline-flex items-center px-1.5 py-0.5 text-xs rounded',
                  colorMap[tag.color] ?? 'bg-gray-500/20 text-gray-300'
                )}
              >
                {tag.label}
              </span>
            );
          })}
        </div>
      )}

      {/* VRAM info */}
      {isLoaded && vramMb != null && vramMb > 0 && (
        <div className="text-xs text-gray-500 mb-2">
          <span>Using {(vramMb / 1024).toFixed(1)} GB VRAM</span>
        </div>
      )}

      {/* Error message */}
      {hasError && error && (
        <p className="text-xs text-red-400 mb-2 line-clamp-2">{error}</p>
      )}

      {/* Action button */}
      <button
        onClick={isLoaded ? onUnload : onLoad}
        disabled={isLoading}
        className={cn(
          'w-full py-2 px-3 text-sm font-medium rounded-lg transition-colors',
          'flex items-center justify-center gap-2',
          isLoaded
            ? 'bg-gray-700 hover:bg-gray-600 text-gray-200'
            : 'bg-blue-600 hover:bg-blue-500 text-white',
          isLoading && 'opacity-50 cursor-not-allowed'
        )}
      >
        {isLoading ? (
          <>
            <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24">
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
                fill="none"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
            Loading...
          </>
        ) : isLoaded ? (
          <>
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
              />
            </svg>
            Unload
          </>
        ) : (
          <>
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
              />
            </svg>
            Load Model
          </>
        )}
      </button>
    </div>
  );
}

export function ModelManager() {
  const pipelines = useAppStore(
    useShallow((s) => Object.values(s.pipelines))
  );
  const modelStatus = useAppStore((s) => s.modelStatus);
  const generationContext = useAppStore((s) => s.generationContext);
  const getPipelineColor = useAppStore((s) => s.getPipelineColor);
  const refreshAllModelStatus = useAppStore((s) => s.refreshAllModelStatus);
  const loadPipelineModel = useAppStore((s) => s.loadPipelineModel);
  const unloadPipelineModel = useAppStore((s) => s.unloadPipelineModel);
  const vram = useAppStore((s) => s.vram);

  // Refresh model status on mount
  useEffect(() => {
    refreshAllModelStatus();
  }, [refreshAllModelStatus]);

  if (pipelines.length === 0) {
    return (
      <div className="text-center text-gray-500 py-4">
        <p className="text-sm">No pipelines available</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* VRAM status bar */}
      {vram && (
        <div className="mb-4">
          <div className="flex items-center justify-between text-xs text-gray-400 mb-1">
            <span>VRAM Usage</span>
            <span>
              {(vram.usedMb / 1024).toFixed(1)} / {(vram.totalMb / 1024).toFixed(1)} GB
            </span>
          </div>
          <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
            <div
              className={cn(
                'h-full transition-all duration-300',
                vram.utilizationPercent > 90
                  ? 'bg-red-500'
                  : vram.utilizationPercent > 70
                    ? 'bg-yellow-500'
                    : 'bg-blue-500'
              )}
              style={{ width: `${Math.min(100, vram.utilizationPercent)}%` }}
            />
          </div>
        </div>
      )}

      {/* Pipeline model cards */}
      <div className="space-y-2">
        {pipelines.map((pipeline) => {
          const status = modelStatus[pipeline.id] ?? { status: 'unloaded' };
          const color = getPipelineColor(pipeline.id);

          // Enrich with context data if this is the active pipeline
          const isActivePipeline = generationContext?.activePipeline === pipeline.id;
          const variant = isActivePipeline ? generationContext?.modelVariant : undefined;
          const lorasForCard = isActivePipeline
            ? generationContext?.loras?.map((l) => ({ name: l.name, scale: l.scale }))
            : undefined;

          return (
            <ModelCard
              key={pipeline.id}
              pipelineId={pipeline.id}
              name={pipeline.name}
              color={color}
              status={status.status as ModelCardProps['status']}
              vramMb={status.vramMb}
              error={status.error}
              modelVariant={variant}
              loras={lorasForCard}
              configTags={status.configTags}
              onLoad={() => loadPipelineModel(pipeline.id)}
              onUnload={() => unloadPipelineModel(pipeline.id)}
            />
          );
        })}
      </div>

      {/* Refresh button */}
      <button
        onClick={() => refreshAllModelStatus()}
        className="w-full py-2 text-sm text-gray-400 hover:text-gray-300 transition-colors"
      >
        Refresh Status
      </button>
    </div>
  );
}
