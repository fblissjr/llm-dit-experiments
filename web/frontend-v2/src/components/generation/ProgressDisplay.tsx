/**
 * ProgressDisplay Component
 *
 * Shows generation progress with step count and percentage.
 */

import { cn } from '@/utils';
import { useAppStore, useSessionStore } from '@/stores';

export function ProgressDisplay() {
  const selectedPipelineId = useAppStore((s) => s.selectedPipelineId);
  const getPipelineColor = useAppStore((s) => s.getPipelineColor);

  const status = useSessionStore((s) => s.status);
  const progress = useSessionStore((s) => s.progress);
  const error = useSessionStore((s) => s.error);

  if (status === 'idle' && !error) {
    return null;
  }

  const pipelineColor = selectedPipelineId
    ? getPipelineColor(selectedPipelineId)
    : '#3b82f6';

  // Calculate percentage
  const percent = progress
    ? Math.round((progress.step / progress.total) * 100)
    : 0;

  return (
    <div className="space-y-2">
      {/* Status text */}
      <div className="flex items-center justify-between text-sm">
        <span className={cn(
          status === 'error' && 'text-red-400',
          status === 'completed' && 'text-green-400',
          status === 'generating' && 'text-gray-300'
        )}>
          {status === 'generating' && progress && (
            <>Step {progress.step} of {progress.total}</>
          )}
          {status === 'generating' && !progress && 'Starting...'}
          {status === 'completed' && 'Complete!'}
          {status === 'error' && (error ?? 'Generation failed')}
        </span>
        {progress && status === 'generating' && (
          <span className="text-gray-400 font-mono">{percent}%</span>
        )}
      </div>

      {/* Progress bar */}
      {status === 'generating' && (
        <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-300"
            style={{
              width: `${percent}%`,
              backgroundColor: pipelineColor,
            }}
          />
        </div>
      )}

      {/* Progress message */}
      {progress?.message && (
        <p className="text-xs text-gray-500">{progress.message}</p>
      )}
    </div>
  );
}
