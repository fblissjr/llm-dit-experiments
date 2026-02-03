/**
 * LoRAItem Component
 *
 * Individual LoRA entry with dropdown selection and scale slider.
 * Outputs "path:scale" format string.
 */

import { useId } from 'react';
import type { LoRAFile } from '@/api/client';
import { cn } from '@/utils';

interface LoRAItemProps {
  path: string;
  scale: number;
  scaleMin: number;
  scaleMax: number;
  availableLoras: LoRAFile[];
  onPathChange: (path: string) => void;
  onScaleChange: (scale: number) => void;
  onRemove: () => void;
  disabled?: boolean;
  index: number;
}

export function LoRAItem({
  path,
  scale,
  scaleMin,
  scaleMax,
  availableLoras,
  onPathChange,
  onScaleChange,
  onRemove,
  disabled = false,
  index,
}: LoRAItemProps) {
  const selectId = useId();
  const scaleId = useId();

  // Calculate fill percentage for slider visual
  const fillPercent = ((scale - scaleMin) / (scaleMax - scaleMin)) * 100;

  // Find current selection in available loras
  const selectedLora = availableLoras.find((l) => l.path === path);

  // Group LoRAs by directory for better organization
  const lorasByDirectory = availableLoras.reduce(
    (acc, lora) => {
      const dir = lora.directory || 'loras';
      if (!acc[dir]) acc[dir] = [];
      acc[dir].push(lora);
      return acc;
    },
    {} as Record<string, LoRAFile[]>
  );

  const directories = Object.keys(lorasByDirectory).sort();

  return (
    <div className="flex items-center gap-2 p-2 bg-gray-800/50 rounded-lg border border-gray-700">
      {/* Index badge */}
      <span className="text-xs text-gray-500 w-4 text-center">{index + 1}</span>

      {/* LoRA selector dropdown */}
      <div className="flex-1 min-w-0">
        <select
          id={selectId}
          value={path}
          onChange={(e) => onPathChange(e.target.value)}
          disabled={disabled}
          className={cn(
            'w-full px-2 py-1.5 text-sm bg-gray-900 border border-gray-600 rounded',
            'text-gray-200 cursor-pointer',
            'focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500',
            disabled && 'opacity-50 cursor-not-allowed',
            !path && 'text-gray-500'
          )}
        >
          <option value="">Select a LoRA...</option>
          {directories.length === 1 ? (
            // Single directory - flat list
            availableLoras.map((lora) => (
              <option key={lora.path} value={lora.path}>
                {lora.name} ({lora.size_mb}MB)
              </option>
            ))
          ) : (
            // Multiple directories - grouped
            directories.map((dir) => (
              <optgroup key={dir} label={dir}>
                {lorasByDirectory[dir].map((lora) => (
                  <option key={lora.path} value={lora.path}>
                    {lora.name} ({lora.size_mb}MB)
                  </option>
                ))}
              </optgroup>
            ))
          )}
        </select>
      </div>

      {/* Scale slider */}
      <div className="flex items-center gap-2 w-32">
        <input
          id={scaleId}
          type="range"
          value={scale}
          onChange={(e) => onScaleChange(parseFloat(e.target.value))}
          min={scaleMin}
          max={scaleMax}
          step={0.05}
          disabled={disabled || !path}
          className={cn(
            'w-16 h-1.5 cursor-pointer rounded-full appearance-none',
            (!path || disabled) && 'opacity-50 cursor-not-allowed'
          )}
          style={{
            background: `linear-gradient(to right, var(--pipeline-color, #3b82f6) 0%, var(--pipeline-color, #3b82f6) ${fillPercent}%, #374151 ${fillPercent}%, #374151 100%)`,
          }}
        />
        <span className="text-xs text-gray-400 w-10 text-right tabular-nums">
          {scale.toFixed(2)}
        </span>
      </div>

      {/* Remove button */}
      <button
        type="button"
        onClick={onRemove}
        disabled={disabled}
        className={cn(
          'p-1 text-gray-400 hover:text-red-400 hover:bg-red-500/10 rounded transition-colors',
          disabled && 'opacity-50 cursor-not-allowed'
        )}
        title="Remove LoRA"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  );
}
