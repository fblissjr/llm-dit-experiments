/**
 * PipelineSelector Component
 *
 * Segmented control for selecting between pipelines within a tab.
 * Uses pills for desktop, dropdown for mobile when >3 pipelines.
 */

import { cn } from '@/utils';
import { useAppStore } from '@/stores';
import { useIsMobile } from '@/hooks';

export function PipelineSelector() {
  const isMobile = useIsMobile();
  const activeTab = useAppStore((s) => s.activeTab);
  const selectedPipelineId = useAppStore((s) => s.selectedPipelineId);
  const selectPipeline = useAppStore((s) => s.selectPipeline);
  const getPipelinesForTab = useAppStore((s) => s.getPipelinesForTab);
  const getPipelineColor = useAppStore((s) => s.getPipelineColor);

  const pipelines = getPipelinesForTab(activeTab);

  if (pipelines.length === 0) {
    return null;
  }

  if (pipelines.length === 1) {
    // Single pipeline - show as title
    return (
      <div className="mb-6">
        <h2 className="text-xl font-semibold" style={{ color: getPipelineColor(pipelines[0].id) }}>
          {pipelines[0].name}
        </h2>
        <p className="text-sm text-gray-400 mt-1">{pipelines[0].description}</p>
      </div>
    );
  }

  // Mobile with many pipelines - use dropdown
  if (isMobile && pipelines.length > 3) {
    return (
      <div className="mb-6">
        <select
          value={selectedPipelineId ?? ''}
          onChange={(e) => selectPipeline(e.target.value)}
          className="form-select w-full"
          style={
            {
              '--pipeline-color': getPipelineColor(selectedPipelineId ?? ''),
            } as React.CSSProperties
          }
        >
          {pipelines.map((pipeline) => (
            <option key={pipeline.id} value={pipeline.id}>
              {pipeline.name}
            </option>
          ))}
        </select>
      </div>
    );
  }

  // Pills for desktop or mobile with few pipelines
  return (
    <div className="mb-6">
      <div className="flex flex-wrap gap-2">
        {pipelines.map((pipeline) => {
          const isSelected = pipeline.id === selectedPipelineId;
          const color = getPipelineColor(pipeline.id);

          return (
            <button
              key={pipeline.id}
              onClick={() => selectPipeline(pipeline.id)}
              className={cn(
                'pipeline-pill',
                isSelected ? 'pipeline-pill-active' : 'pipeline-pill-inactive'
              )}
              style={
                isSelected
                  ? ({ '--pipeline-color': color, backgroundColor: color, borderColor: color } as React.CSSProperties)
                  : undefined
              }
            >
              {pipeline.name}
            </button>
          );
        })}
      </div>

      {/* Description of selected pipeline */}
      {selectedPipelineId && (
        <p className="text-sm text-gray-400 mt-3">
          {pipelines.find((p) => p.id === selectedPipelineId)?.description}
        </p>
      )}
    </div>
  );
}
