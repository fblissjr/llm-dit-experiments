/**
 * Header Component
 *
 * Application header with pipeline tabs, VRAM bar, and model status.
 */

import { useState, useCallback } from 'react';
import { usePipelineStore } from '@/stores/pipelineStore';
import { useModelStore } from '@/stores/modelStore';
import { useUIStore } from '@/stores/uiStore';
import { VRAMBar } from '../model/VRAMBar';
import type { PipelineColor, LoadStatus } from '@/types';
import { PIPELINE_COLOR_CLASSES } from '@/types';

export function Header() {
  const { pipelines, selectedPipelineId, selectPipeline } = usePipelineStore();
  const { vram, modelStatus, loadModel, loadingPipelineId } = useModelStore();
  const { isMobile, toggleModelPanel } = useUIStore();

  const pipelineList = Object.values(pipelines);
  const loadedPipeline = Object.keys(modelStatus).find(
    (id) => modelStatus[id].status === 'loaded'
  );

  // Get status for a pipeline
  const getStatus = useCallback((pipelineId: string): LoadStatus => {
    if (loadingPipelineId === pipelineId) return 'loading';
    return modelStatus[pipelineId]?.status ?? 'unloaded';
  }, [modelStatus, loadingPipelineId]);

  // Handle model load from tab
  const handleLoadModel = useCallback((pipelineId: string, e: React.MouseEvent) => {
    e.stopPropagation(); // Don't trigger tab selection
    loadModel(pipelineId);
  }, [loadModel]);

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
              status={getStatus(pipeline.id)}
              onClick={() => selectPipeline(pipeline.id)}
              onLoadClick={(e) => handleLoadModel(pipeline.id, e)}
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
  status: LoadStatus;
  onClick: () => void;
  onLoadClick: (e: React.MouseEvent) => void;
}

function PipelineTab({
  name,
  icon,
  color,
  isSelected,
  status,
  onClick,
  onLoadClick,
}: PipelineTabProps) {
  const [isHovered, setIsHovered] = useState(false);
  const isLoaded = status === 'loaded';
  const isLoading = status === 'loading';
  const canLoad = status === 'unloaded' || status === 'error';

  // Status indicator styling
  const getStatusIndicator = () => {
    if (isLoading) {
      return (
        <span
          className="w-2 h-2 rounded-full bg-yellow-500 animate-pulse"
          title="Loading model..."
        />
      );
    }
    if (isLoaded) {
      return (
        <span
          className="w-2 h-2 rounded-full bg-green-500"
          title="Model loaded"
        />
      );
    }
    // Show gray dot for unloaded on hover, or when selected
    if (isHovered || isSelected) {
      return (
        <span
          className="w-2 h-2 rounded-full bg-gray-500"
          title="Model not loaded"
        />
      );
    }
    return null;
  };

  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className={`
        relative px-4 py-2 text-sm font-medium rounded-t-lg transition-colors whitespace-nowrap
        border-b-2 -mb-px flex items-center gap-2 group
        ${
          isSelected
            ? `${PIPELINE_COLOR_CLASSES.borderAndText[color]} bg-gray-900`
            : 'border-transparent text-gray-400 hover:text-gray-200 hover:bg-gray-700/50'
        }
      `}
    >
      {icon && <span>{icon}</span>}
      {name}
      {getStatusIndicator()}

      {/* Load button on hover - only show if unloaded and hovered */}
      {canLoad && isHovered && !isLoading && (
        <span
          onClick={onLoadClick}
          className={`
            ml-1 px-1.5 py-0.5 text-xs rounded
            bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-white
            transition-colors cursor-pointer
          `}
          title="Load model"
        >
          Load
        </span>
      )}
    </button>
  );
}
