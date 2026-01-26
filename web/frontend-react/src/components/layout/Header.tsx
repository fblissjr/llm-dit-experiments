/**
 * Header Component
 *
 * Application header with pipeline tabs, VRAM bar, and model status.
 */

import { usePipelineStore } from '@/stores/pipelineStore';
import { useModelStore } from '@/stores/modelStore';
import { useUIStore } from '@/stores/uiStore';
import { VRAMBar } from '../model/VRAMBar';
import type { PipelineColor } from '@/types';
import { PIPELINE_COLOR_CLASSES } from '@/types';

export function Header() {
  const { pipelines, selectedPipelineId, selectPipeline } = usePipelineStore();
  const { vram, modelStatus } = useModelStore();
  const { isMobile, toggleModelPanel } = useUIStore();

  const pipelineList = Object.values(pipelines);
  const loadedPipeline = Object.keys(modelStatus).find(
    (id) => modelStatus[id].status === 'loaded'
  );

  return (
    <header className="bg-gray-800 border-b border-gray-700 sticky top-0 z-40">
      <div className="px-4 py-3">
        <div className="flex items-center justify-between">
          {/* Logo and title */}
          <div className="flex items-center gap-3">
            <h1 className="text-lg font-semibold text-gray-100">
              LLM-DiT Studio
            </h1>
          </div>

          {/* VRAM bar (desktop) */}
          {!isMobile && vram && (
            <div className="flex items-center gap-4">
              <VRAMBar
                usedMB={vram.usedMB}
                totalMB={vram.totalMB}
                onClick={toggleModelPanel}
              />
              {loadedPipeline && (
                <div className="flex items-center gap-2 text-sm">
                  <span className="text-gray-400">Loaded:</span>
                  <span className={PIPELINE_COLOR_CLASSES.text[(pipelines[loadedPipeline]?.color ?? 'blue') as PipelineColor]}>
                    {pipelines[loadedPipeline]?.name ?? loadedPipeline}
                  </span>
                  <span className="w-2 h-2 rounded-full bg-green-500" title="Ready" />
                </div>
              )}
            </div>
          )}
        </div>

        {/* Pipeline tabs */}
        <nav className="flex gap-1 mt-3 -mb-3 overflow-x-auto scrollbar-hide">
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
      </div>

      {/* Mobile VRAM indicator */}
      {isMobile && vram && (
        <div
          className="px-4 py-2 border-t border-gray-700 flex items-center justify-between cursor-pointer"
          onClick={toggleModelPanel}
        >
          <VRAMBar
            usedMB={vram.usedMB}
            totalMB={vram.totalMB}
            compact
          />
          {loadedPipeline && (
            <span className="text-sm text-gray-400">
              {pipelines[loadedPipeline]?.name} · Ready
            </span>
          )}
        </div>
      )}
    </header>
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
  return (
    <button
      onClick={onClick}
      className={`
        px-4 py-2 text-sm font-medium rounded-t-lg transition-colors whitespace-nowrap
        border-b-2 -mb-px
        ${
          isSelected
            ? `${PIPELINE_COLOR_CLASSES.borderAndText[color]} bg-gray-900`
            : 'border-transparent text-gray-400 hover:text-gray-200 hover:bg-gray-700/50'
        }
      `}
    >
      {icon && <span className="mr-1.5">{icon}</span>}
      {name}
    </button>
  );
}
