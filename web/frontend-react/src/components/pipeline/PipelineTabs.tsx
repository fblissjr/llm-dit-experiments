/**
 * Pipeline Tabs
 *
 * Tab bar for switching between pipelines.
 * Shows icon and name, with active indicator.
 */

import { usePipelineStore } from '@/stores/pipelineStore';
import type { PipelineColor } from '@/types';

export function PipelineTabs() {
  const { pipelines, selectedPipelineId, selectPipeline } = usePipelineStore();
  const pipelineList = Object.values(pipelines);

  if (pipelineList.length === 0) {
    return null;
  }

  return (
    <nav className="flex gap-1 overflow-x-auto scrollbar-hide pb-2">
      {pipelineList.map((pipeline) => (
        <PipelineTab
          key={pipeline.id}
          id={pipeline.id}
          name={pipeline.name}
          icon={pipeline.icon}
          color={pipeline.color}
          isSelected={pipeline.id === selectedPipelineId}
          onClick={() => selectPipeline(pipeline.id)}
        />
      ))}
    </nav>
  );
}

interface PipelineTabProps {
  id: string;
  name: string;
  icon?: string;
  color: PipelineColor;
  isSelected: boolean;
  onClick: () => void;
}

function PipelineTab({
  name,
  icon,
  color,
  isSelected,
  onClick,
}: PipelineTabProps) {
  const colorMap: Record<PipelineColor, string> = {
    blue: 'bg-blue-500/20 text-blue-400 border-blue-500',
    purple: 'bg-purple-500/20 text-purple-400 border-purple-500',
    orange: 'bg-orange-500/20 text-orange-400 border-orange-500',
    teal: 'bg-teal-500/20 text-teal-400 border-teal-500',
    green: 'bg-green-500/20 text-green-400 border-green-500',
    pink: 'bg-pink-500/20 text-pink-400 border-pink-500',
  };

  return (
    <button
      onClick={onClick}
      className={`
        px-4 py-2 rounded-lg text-sm font-medium whitespace-nowrap
        border transition-all duration-200
        ${
          isSelected
            ? colorMap[color]
            : 'border-transparent text-gray-400 hover:text-gray-200 hover:bg-gray-800'
        }
      `}
    >
      {icon && <span className="mr-1.5">{icon}</span>}
      {name}
    </button>
  );
}
