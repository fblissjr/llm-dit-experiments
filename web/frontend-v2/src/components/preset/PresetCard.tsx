/**
 * PresetCard - A single preset rendered as a compact visual card.
 *
 * Shows preset name, description (truncated), variant badge, and param count.
 * Three visual states: default, hover (CSS), active (checkmark + pipeline color border).
 */

import type { GenerationPreset } from '@/api/types';

interface PresetCardProps {
  preset: GenerationPreset;
  isActive: boolean;
  onClick: () => void;
}

export function PresetCard({ preset, isActive, onClick }: PresetCardProps) {
  const paramCount = Object.keys(preset.params).length;

  return (
    <button
      type="button"
      onClick={onClick}
      className={`preset-card text-left relative flex flex-col justify-between ${
        isActive ? 'preset-card-active' : ''
      }`}
    >
      {/* Active checkmark */}
      {isActive && (
        <span
          className="absolute top-2 right-2 w-5 h-5 rounded-full flex items-center justify-center text-xs text-white"
          style={{ backgroundColor: 'var(--pipeline-color)' }}
        >
          <svg width="10" height="10" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path d="M3 8.5l3.5 3.5L13 4" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </span>
      )}

      {/* Name + description */}
      <div className="space-y-1 min-w-0">
        <div className="text-sm font-medium text-gray-100 truncate pr-6">
          {preset.name}
        </div>
        {preset.description && (
          <div className="text-xs text-gray-400 line-clamp-2 leading-relaxed">
            {preset.description}
          </div>
        )}
      </div>

      {/* Bottom badges */}
      <div className="flex items-center gap-2 mt-2">
        {preset.variant && (
          <span className="text-[10px] px-1.5 py-0.5 rounded-sm bg-gray-700 text-gray-300 truncate">
            {preset.variant}
          </span>
        )}
        <span className="text-[10px] px-1.5 py-0.5 rounded-sm bg-gray-700/50 text-gray-400 ml-auto whitespace-nowrap">
          {paramCount} param{paramCount !== 1 ? 's' : ''}
        </span>
      </div>
    </button>
  );
}
