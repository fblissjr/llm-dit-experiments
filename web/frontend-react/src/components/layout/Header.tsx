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

// Gear icon SVG component
function GearIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

export function Header() {
  const { pipelines, selectedPipelineId, selectPipeline } = usePipelineStore();
  const { vram, modelStatus, loadModel, loadingPipelineId } = useModelStore();
  const { isMobile, toggleModelPanel, setView } = useUIStore();

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
            <button
              onClick={() => setView('settings')}
              className="p-1.5 text-gray-400 hover:text-gray-200 hover:bg-gray-700 rounded-lg transition-colors"
              title="Settings (,)"
            >
              <GearIcon className="w-5 h-5" />
            </button>
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
