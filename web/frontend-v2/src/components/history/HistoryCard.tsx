/**
 * HistoryCard Component
 *
 * Individual history item showing thumbnail, prompt preview, and key params.
 */

import { useCallback, useState } from 'react';
import { cn } from '@/utils';
import { useSessionStore } from '@/stores';
import type { HistoryItem } from '@/api/types';
import { ImageViewer } from '@/components/viewer/ImageViewer';

// Map color names to CSS values
const colorMap: Record<string, string> = {
  blue: '#3b82f6',
  purple: '#a855f7',
  orange: '#f97316',
  teal: '#14b8a6',
  green: '#22c55e',
  pink: '#ec4899',
};

interface HistoryCardProps {
  item: HistoryItem;
}

export function HistoryCard({ item }: HistoryCardProps) {
  const loadHistoryParams = useSessionStore((s) => s.loadHistoryParams);
  const removeHistoryItem = useSessionStore((s) => s.removeHistoryItem);
  const [showViewer, setShowViewer] = useState(false);

  const handleClick = useCallback(() => {
    loadHistoryParams(item);
  }, [item, loadHistoryParams]);

  const handleRemove = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      removeHistoryItem(item.id);
    },
    [item.id, removeHistoryItem]
  );

  const handleThumbnailClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      // Only show viewer if we have a valid image URL
      const imageUrl = item.fullImageUrl || item.thumbnailUrl;
      if (imageUrl && imageUrl !== '') {
        setShowViewer(true);
      }
    },
    [item.fullImageUrl, item.thumbnailUrl]
  );

  // Format relative time
  const relativeTime = formatRelativeTime(item.timestamp);
  const pipelineColor = colorMap[item.pipelineColor] ?? colorMap.blue;

  // Determine which URL to show in viewer (prefer full, fallback to thumbnail)
  const viewerUrl = item.fullImageUrl || item.thumbnailUrl;

  return (
    <>
      <div
        onClick={handleClick}
        className={cn(
          'card p-2 cursor-pointer transition-all',
          'hover:bg-gray-700/50 hover:border-gray-600',
          'group'
        )}
      >
        <div className="flex gap-3">
          {/* Thumbnail */}
          <div
            className="relative w-16 h-16 flex-shrink-0 rounded overflow-hidden bg-gray-700 cursor-pointer"
            onClick={handleThumbnailClick}
            title="Click to view full image"
          >
            {item.thumbnailUrl ? (
              <img
                src={item.thumbnailUrl}
                alt=""
                className="w-full h-full object-cover hover:opacity-80 transition-opacity"
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center">
                <svg
                  className="w-6 h-6 text-gray-500"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                  />
                </svg>
              </div>
            )}

            {/* Remove button (on hover) */}
            <button
              onClick={handleRemove}
              className="absolute top-0.5 right-0.5 p-1 bg-gray-900/80 rounded
                         opacity-0 group-hover:opacity-100 transition-opacity
                         hover:bg-red-600 z-10"
              title="Remove from history"
            >
              <svg
                className="w-3 h-3"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            {/* Pipeline badge and time */}
            <div className="flex items-center justify-between mb-1">
              <span
                className="text-xs font-medium px-1.5 py-0.5 rounded"
                style={{ backgroundColor: `${pipelineColor}20`, color: pipelineColor }}
              >
                {item.pipelineName}
              </span>
              <span className="text-xs text-gray-500">{relativeTime}</span>
            </div>

            {/* Prompt preview */}
            <p className="text-sm text-gray-300 line-clamp-2 mb-1">
              {item.shortPrompt || 'No prompt'}
            </p>

            {/* Key params */}
            <p className="text-xs text-gray-500 truncate">
              {item.keyParams}
            </p>
          </div>
        </div>
      </div>

      {/* Image viewer modal */}
      {showViewer && viewerUrl && (
        <ImageViewer
          url={viewerUrl}
          alt={item.shortPrompt}
          onClose={() => setShowViewer(false)}
        />
      )}
    </>
  );
}

/**
 * Format timestamp to relative time string
 */
function formatRelativeTime(timestamp: number): string {
  const now = Date.now();
  const diff = now - timestamp;

  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) return `${days}d ago`;
  if (hours > 0) return `${hours}h ago`;
  if (minutes > 0) return `${minutes}m ago`;
  return 'just now';
}
