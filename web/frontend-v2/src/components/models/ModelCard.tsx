/**
 * ModelCard Component
 *
 * Displays a single model with its status and load/unload controls.
 */

import { cn } from '@/utils';
import type { PipelineSchema, ModelStatusResponse } from '@/api/types';

interface ModelCardProps {
  pipeline: PipelineSchema;
  status: ModelStatusResponse | null;
  color: string;
  onLoad: () => void;
  onUnload: () => void;
}

const tagColors: Record<string, string> = {
  purple: 'bg-purple-500/15 text-purple-400 border-purple-500/30',
  blue: 'bg-blue-500/15 text-blue-400 border-blue-500/30',
  orange: 'bg-orange-500/15 text-orange-400 border-orange-500/30',
  green: 'bg-green-500/15 text-green-400 border-green-500/30',
};

export function ModelCard({ pipeline, status, color, onLoad, onUnload }: ModelCardProps) {
  const isLoaded = status?.status === 'loaded';
  const isLoading = status?.status === 'loading';
  const hasError = status?.status === 'error';

  const vramText = status?.vramMb
    ? `${(status.vramMb / 1024).toFixed(1)}GB`
    : null;

  return (
    <div
      className={cn(
        'p-3 rounded-lg border transition-colors',
        isLoaded ? 'border-green-500/50 bg-green-500/5' : 'border-gray-700 bg-gray-800/50'
      )}
    >
      <div className="flex items-start justify-between gap-3">
        {/* Left: Name and status */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            {/* Status indicator dot */}
            <div
              className={cn(
                'w-2 h-2 rounded-full',
                isLoaded && 'bg-green-500',
                isLoading && 'bg-yellow-500 animate-pulse',
                hasError && 'bg-red-500',
                !status && 'bg-gray-500'
              )}
            />
            <h3
              className="text-sm font-medium truncate"
              style={{ color: isLoaded ? color : undefined }}
            >
              {pipeline.name}
            </h3>
          </div>

          {/* VRAM usage */}
          {vramText && (
            <div className="text-xs text-gray-400 mt-1 ml-4">
              {vramText}
            </div>
          )}

          {/* Config tags */}
          {status?.configTags && status.configTags.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-1 ml-4">
              {status.configTags.map((tag) => (
                <span
                  key={tag.key}
                  className={cn(
                    'px-1.5 py-0.5 text-[10px] font-medium rounded-sm border',
                    tagColors[tag.color] ?? tagColors.blue
                  )}
                >
                  {tag.label}
                </span>
              ))}
            </div>
          )}

          {/* Error message */}
          {hasError && status?.error && (
            <div className="text-xs text-red-400 mt-1 ml-4 truncate" title={status.error}>
              {status.error}
            </div>
          )}

          {/* Config incompatibility warnings (shown only when unloaded) */}
          {!isLoaded && !isLoading && status?.configWarnings?.map((warn, i) => (
            <div
              key={i}
              className={cn(
                'text-xs mt-1 ml-4 truncate',
                warn.severity === 'error' ? 'text-red-400' : 'text-yellow-400'
              )}
              title={warn.message}
            >
              {warn.message}
            </div>
          ))}
        </div>

        {/* Right: Load/Unload button */}
        <button
          onClick={isLoaded ? onUnload : onLoad}
          disabled={isLoading}
          className={cn(
            'px-2.5 py-1 text-xs font-medium rounded-sm transition-colors',
            'disabled:opacity-50 disabled:cursor-not-allowed',
            isLoaded
              ? 'bg-red-500/10 text-red-400 hover:bg-red-500/20 border border-red-500/30'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600 border border-gray-600'
          )}
        >
          {isLoading ? (
            <span className="flex items-center gap-1.5">
              <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              Loading
            </span>
          ) : isLoaded ? (
            'Unload'
          ) : (
            'Load'
          )}
        </button>
      </div>
    </div>
  );
}
