/**
 * ModelManager Component
 *
 * Read-only status panel showing what models are loaded in memory.
 * Auto-load happens at generation time -- no manual load/unload buttons.
 * Unload All is in SettingsMenu for the rare case of freeing VRAM.
 */

import { useEffect } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { cn } from '@/utils';
import { useAppStore } from '@/stores';
import { VRAMBar } from '@/components/common/VRAMBar';

interface LoRABadgeInfo {
  name: string;
  scale: number;
}

interface ModelCardProps {
  name: string;
  color: string;
  status: 'unloaded' | 'loading' | 'loaded' | 'error';
  vramMb?: number;
  error?: string;
  modelVariant?: string | null;
  loras?: LoRABadgeInfo[];
  configTags?: { key: string; label: string; color: string }[];
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
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <div
            className="w-2 h-2 rounded-full shrink-0"
            style={{
              backgroundColor: isLoaded ? color : '#6b7280',
            }}
          />
          <span className="text-sm font-medium truncate">{name}</span>
        </div>

        {/* Status badge */}
        <span
          className={cn(
            'text-xs px-2 py-0.5 rounded-full shrink-0',
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
        <p className="text-xs text-gray-500 mt-1.5">{modelVariant}</p>
      )}

      {/* LoRA badges */}
      {isLoaded && loras && loras.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-1.5">
          {loras.map((lora) => (
            <span
              key={lora.name}
              className="inline-flex items-center gap-0.5 px-1.5 py-0.5 text-xs bg-purple-500/20 text-purple-300 rounded-sm"
            >
              {lora.name}
              <span className="text-purple-400/70">@{lora.scale.toFixed(2)}</span>
            </span>
          ))}
        </div>
      )}

      {/* Config tags */}
      {isLoaded && configTags && configTags.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-1.5">
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
                  'inline-flex items-center px-1.5 py-0.5 text-xs rounded-sm',
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
        <div className="text-xs text-gray-500 mt-1.5">
          <span>Using {(vramMb / 1024).toFixed(1)} GB VRAM</span>
        </div>
      )}

      {/* Error message */}
      {hasError && error && (
        <p className="text-xs text-red-400 mt-1.5 line-clamp-2">{error}</p>
      )}
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
          </div>
          <VRAMBar
            usedGb={vram.usedMb / 1024}
            totalGb={vram.totalMb / 1024}
            height="h-2"
          />
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
              name={pipeline.name}
              color={color}
              status={status.status as ModelCardProps['status']}
              vramMb={status.vramMb}
              error={status.error}
              modelVariant={variant}
              loras={lorasForCard}
              configTags={status.configTags}
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
