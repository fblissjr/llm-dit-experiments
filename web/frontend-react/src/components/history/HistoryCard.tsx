/**
 * History Card
 *
 * Thumbnail card for a single history item.
 * Shows preview, prompt snippet, and key params.
 */

import type { HistoryItem, PipelineColor } from '@/types';
import { PIPELINE_COLOR_CLASSES } from '@/types';

interface HistoryCardProps {
  item: HistoryItem;
  isSelected?: boolean;
  isCompareMode?: boolean;
  onClick: () => void;
  onUseAsInput?: () => void;
  onDelete?: () => void;
}

export function HistoryCard({
  item,
  isSelected = false,
  isCompareMode = false,
  onClick,
  onUseAsInput,
  onDelete,
}: HistoryCardProps) {
  const pipelineColor = item.pipelineColor as PipelineColor;
  const accentBorder = isSelected
    ? PIPELINE_COLOR_CLASSES.border[pipelineColor] ?? 'border-blue-500'
    : 'border-gray-700';

  return (
    <div
      className={`
        history-item relative group
        border-2 ${accentBorder}
        ${isCompareMode ? 'cursor-pointer' : ''}
      `}
      onClick={onClick}
    >
      {/* Thumbnail */}
      <div className="aspect-square overflow-hidden bg-gray-800">
        <img
          src={item.thumbnailUrl}
          alt={item.shortPrompt}
          className="w-full h-full object-cover"
          loading="lazy"
        />
      </div>

      {/* Overlay with info */}
      <div className="history-item-overlay">
        <div className="absolute inset-x-0 bottom-0 p-2 text-white">
          <p className="text-xs line-clamp-2 mb-1">{item.shortPrompt}</p>
          <div className="flex items-center justify-between text-[10px] text-gray-300">
            <span>{item.keyParams}</span>
            <span>{item.relativeTime}</span>
          </div>
        </div>
      </div>

      {/* Pipeline indicator */}
      <div
        className={`
          absolute top-2 left-2 px-1.5 py-0.5 rounded text-[10px] font-medium text-white
          ${PIPELINE_COLOR_CLASSES.bgSubtle[pipelineColor] ?? 'bg-blue-500/80'}
        `}
      >
        {item.pipelineName}
      </div>

      {/* Compare mode checkbox */}
      {isCompareMode && (
        <div className="absolute top-2 right-2">
          <div
            className={`
              w-5 h-5 rounded border-2 flex items-center justify-center
              ${isSelected ? 'bg-blue-500 border-blue-500' : 'border-white/50 bg-black/30'}
            `}
          >
            {isSelected && (
              <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
              </svg>
            )}
          </div>
        </div>
      )}

      {/* Action buttons (visible on hover) */}
      {!isCompareMode && (
        <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity flex gap-1">
          {onUseAsInput && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onUseAsInput();
              }}
              className="p-1.5 bg-gray-900/80 rounded-lg text-gray-300 hover:text-white hover:bg-gray-900 transition-colors"
              title="Use as input"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
              </svg>
            </button>
          )}
          {onDelete && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onDelete();
              }}
              className="p-1.5 bg-gray-900/80 rounded-lg text-gray-300 hover:text-red-400 hover:bg-gray-900 transition-colors"
              title="Delete"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </button>
          )}
        </div>
      )}
    </div>
  );
}
