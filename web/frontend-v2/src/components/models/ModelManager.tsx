/**
 * ModelManager Component
 *
 * Displays all pipelines with load/unload controls and VRAM status.
 * This is shown at the top of the left navigation sidebar.
 */

import { useEffect } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useAppStore } from '@/stores';
import { ModelCard } from './ModelCard';
import { cn } from '@/utils';

interface ModelManagerProps {
  className?: string;
  compact?: boolean;
}

export function ModelManager({ className, compact = false }: ModelManagerProps) {
  const pipelines = useAppStore(useShallow((s) => Object.values(s.pipelines)));
  const modelStatus = useAppStore((s) => s.modelStatus);
  const getPipelineColor = useAppStore((s) => s.getPipelineColor);
  const loadPipelineModel = useAppStore((s) => s.loadPipelineModel);
  const unloadPipelineModel = useAppStore((s) => s.unloadPipelineModel);
  const refreshAllModelStatus = useAppStore((s) => s.refreshAllModelStatus);
  const vram = useAppStore((s) => s.vram);

  // Refresh model status on mount and periodically
  useEffect(() => {
    refreshAllModelStatus();
    const interval = setInterval(refreshAllModelStatus, 10000); // Every 10s
    return () => clearInterval(interval);
  }, [refreshAllModelStatus]);

  // Group pipelines by category
  const imagePipelines = pipelines.filter((p) => p.category === 'image');
  const videoPipelines = pipelines.filter((p) => p.category === 'video');

  return (
    <div className={cn('space-y-4', className)}>
      {/* VRAM Status */}
      {vram && !compact && (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs text-gray-400">
            <span>VRAM Usage</span>
            <span className="font-mono">
              {(vram.usedMB / 1024).toFixed(1)}/{(vram.totalMB / 1024).toFixed(0)}GB
            </span>
          </div>
          <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
            <div
              className={cn(
                'h-full rounded-full transition-all',
                vram.utilizationPercent < 50 && 'bg-green-500',
                vram.utilizationPercent >= 50 && vram.utilizationPercent < 75 && 'bg-yellow-500',
                vram.utilizationPercent >= 75 && vram.utilizationPercent < 90 && 'bg-orange-500',
                vram.utilizationPercent >= 90 && 'bg-red-500'
              )}
              style={{ width: `${vram.utilizationPercent}%` }}
            />
          </div>
        </div>
      )}

      {/* Image Models */}
      {imagePipelines.length > 0 && (
        <div className="space-y-2">
          {!compact && (
            <h3 className="text-xs font-medium text-gray-400 uppercase tracking-wider">
              Image Models
            </h3>
          )}
          <div className="space-y-2">
            {imagePipelines.map((pipeline) => (
              <ModelCard
                key={pipeline.id}
                pipeline={pipeline}
                status={modelStatus[pipeline.id] ?? null}
                color={getPipelineColor(pipeline.id)}
                onLoad={() => loadPipelineModel(pipeline.id)}
                onUnload={() => unloadPipelineModel(pipeline.id)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Video Models */}
      {videoPipelines.length > 0 && (
        <div className="space-y-2">
          {!compact && (
            <h3 className="text-xs font-medium text-gray-400 uppercase tracking-wider">
              Video Models
            </h3>
          )}
          <div className="space-y-2">
            {videoPipelines.map((pipeline) => (
              <ModelCard
                key={pipeline.id}
                pipeline={pipeline}
                status={modelStatus[pipeline.id] ?? null}
                color={getPipelineColor(pipeline.id)}
                onLoad={() => loadPipelineModel(pipeline.id)}
                onUnload={() => unloadPipelineModel(pipeline.id)}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
