/**
 * Progress Display
 *
 * Shows generation progress with step count and time estimate.
 */

import { useGenerationStore } from '@/stores/generationStore';
import { usePipelineStore } from '@/stores/pipelineStore';
import { formatDuration } from '@/types/generation';
import { PIPELINE_COLOR_CLASSES, type PipelineColor } from '@/types/pipeline';

export function ProgressDisplay() {
  const { status, progress } = useGenerationStore();
  const { selectedPipelineId, pipelines } = usePipelineStore();
  const pipeline = selectedPipelineId ? pipelines[selectedPipelineId] : null;

  if (status !== 'generating' || !progress) {
    return null;
  }

  const pipelineColor = (pipeline?.color ?? 'blue') as PipelineColor;
  const barColor = PIPELINE_COLOR_CLASSES.bg[pipelineColor];

  return (
    <div className="card space-y-3">
      <div className="flex items-center justify-between text-sm">
        <span className="text-gray-300">
          {progress.message ?? 'Generating...'}
        </span>
        <span className="text-gray-500">
          Step {progress.step}/{progress.totalSteps}
        </span>
      </div>

      {/* Progress bar */}
      <div className="progress-bar">
        <div
          className={`progress-fill ${barColor}`}
          style={{ width: `${progress.percent}%` }}
        />
      </div>

      {/* Time info */}
      <div className="flex items-center justify-between text-xs text-gray-500">
        <span>Elapsed: {formatDuration(progress.elapsedMs)}</span>
        {progress.estimatedRemainingMs !== undefined && (
          <span>Remaining: ~{formatDuration(progress.estimatedRemainingMs)}</span>
        )}
      </div>
    </div>
  );
}
