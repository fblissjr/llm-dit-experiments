/**
 * LoRAList Component
 *
 * Container for multiple LoRA entries. Manages add/remove and
 * converts between UI state and "path:scale" string format.
 */

import { useCallback } from 'react';
import type { ParamSchema } from '@/api/types';
import { cn } from '@/utils';
import { LoRAItem } from './LoRAItem';

interface LoRAListProps {
  param: ParamSchema;
  value: string[];
  onChange: (value: string[]) => void;
  disabled?: boolean;
}

// Parse "path:scale" format into components
function parseLoraSpec(spec: string): { path: string; scale: number } {
  if (!spec.includes(':')) {
    return { path: spec, scale: 0.8 };
  }
  const lastColon = spec.lastIndexOf(':');
  const path = spec.slice(0, lastColon);
  const scaleStr = spec.slice(lastColon + 1);
  const scale = parseFloat(scaleStr);
  // If scale parsing fails, treat whole string as path
  if (isNaN(scale)) {
    return { path: spec, scale: 0.8 };
  }
  return { path, scale };
}

// Format components back to "path:scale" string
function formatLoraSpec(path: string, scale: number): string {
  return `${path}:${scale.toFixed(2)}`;
}

export function LoRAList({ param, value, onChange, disabled = false }: LoRAListProps) {
  const scaleMin = param.scale_min ?? -2.0;
  const scaleMax = param.scale_max ?? 2.0;
  const maxCount = param.max_count ?? 5;

  // Parse all specs into structured form for editing
  const items = (value || []).map(parseLoraSpec);

  const handlePathChange = useCallback(
    (index: number, newPath: string) => {
      const newItems = [...items];
      newItems[index] = { ...newItems[index], path: newPath };
      onChange(newItems.map((item) => formatLoraSpec(item.path, item.scale)));
    },
    [items, onChange]
  );

  const handleScaleChange = useCallback(
    (index: number, newScale: number) => {
      const newItems = [...items];
      newItems[index] = { ...newItems[index], scale: newScale };
      onChange(newItems.map((item) => formatLoraSpec(item.path, item.scale)));
    },
    [items, onChange]
  );

  const handleRemove = useCallback(
    (index: number) => {
      const newItems = items.filter((_, i) => i !== index);
      onChange(newItems.map((item) => formatLoraSpec(item.path, item.scale)));
    },
    [items, onChange]
  );

  const handleAdd = useCallback(() => {
    if (items.length >= maxCount) return;
    const newItems = [...items, { path: '', scale: 0.8 }];
    onChange(newItems.map((item) => formatLoraSpec(item.path, item.scale)));
  }, [items, maxCount, onChange]);

  return (
    <div className="form-control">
      <div className="flex items-center justify-between mb-2">
        <label className="form-label" title={param.tooltip}>
          {param.label}
        </label>
        <span className="text-xs text-gray-500">
          {items.length}/{maxCount}
        </span>
      </div>

      {/* LoRA items */}
      <div className="space-y-2">
        {items.map((item, index) => (
          <LoRAItem
            key={index}
            index={index}
            path={item.path}
            scale={item.scale}
            scaleMin={scaleMin}
            scaleMax={scaleMax}
            onPathChange={(path) => handlePathChange(index, path)}
            onScaleChange={(scale) => handleScaleChange(index, scale)}
            onRemove={() => handleRemove(index)}
            disabled={disabled}
          />
        ))}
      </div>

      {/* Add button */}
      {items.length < maxCount && (
        <button
          type="button"
          onClick={handleAdd}
          disabled={disabled}
          className={cn(
            'mt-2 w-full py-1.5 px-3 text-sm text-gray-400 border border-dashed border-gray-600',
            'rounded-lg hover:border-gray-500 hover:text-gray-300 transition-colors',
            'flex items-center justify-center gap-1',
            disabled && 'opacity-50 cursor-not-allowed'
          )}
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          Add LoRA
        </button>
      )}

      {/* Help text */}
      {items.length === 0 && (
        <p className="mt-2 text-xs text-gray-500">
          LoRAs modify the model's style or subject. Add paths to .safetensors files.
        </p>
      )}
    </div>
  );
}
