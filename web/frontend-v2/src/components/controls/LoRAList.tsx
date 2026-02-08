/**
 * LoRAList Component
 *
 * Container for multiple LoRA entries. Fetches available LoRAs from server
 * and provides dropdown selection. Manages add/remove and converts between
 * UI state and "path:scale" string format.
 */

import { useCallback, useEffect, useState } from 'react';
import type { ParamSchema } from '@/api/types';
import { fetchAvailableLoras, type LoRAFile } from '@/api/client';
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

  // Fetch available LoRAs from server
  const [availableLoras, setAvailableLoras] = useState<LoRAFile[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchAvailableLoras()
      .then((response) => {
        setAvailableLoras(response.loras);
        setIsLoading(false);
      })
      .catch((err) => {
        setError(err.message || 'Failed to load LoRAs');
        setIsLoading(false);
      });
  }, []);

  // Parse all specs into structured form for editing
  const items = (value || []).map(parseLoraSpec);

  // Convert items to spec strings for the form store.
  // Empty-path entries (":0.80") are kept so placeholder rows persist in the UI.
  // The backend filters these out before processing (see web/routers/flux2.py).
  const toSpecs = useCallback(
    (list: { path: string; scale: number }[]) =>
      list.map((item) => formatLoraSpec(item.path, item.scale)),
    []
  );

  const handlePathChange = useCallback(
    (index: number, newPath: string) => {
      const newItems = [...items];
      newItems[index] = { ...newItems[index], path: newPath };
      onChange(toSpecs(newItems));
    },
    [items, onChange, toSpecs]
  );

  const handleScaleChange = useCallback(
    (index: number, newScale: number) => {
      const newItems = [...items];
      newItems[index] = { ...newItems[index], scale: newScale };
      onChange(toSpecs(newItems));
    },
    [items, onChange, toSpecs]
  );

  const handleRemove = useCallback(
    (index: number) => {
      const newItems = items.filter((_, i) => i !== index);
      onChange(toSpecs(newItems));
    },
    [items, onChange, toSpecs]
  );

  const handleAdd = useCallback(() => {
    if (items.length >= maxCount) return;
    // Add empty-path entry -- rendered as a dropdown row for the user to pick a LoRA.
    // The ":0.80" spec is filtered by the backend before processing.
    const newItems = [...items, { path: '', scale: 0.8 }];
    onChange(toSpecs(newItems));
  }, [items, maxCount, onChange, toSpecs]);

  // Get LoRAs that aren't already selected
  const availableForSelection = availableLoras.filter(
    (lora) => !items.some((item) => item.path === lora.path)
  );

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

      {/* Loading/error state */}
      {isLoading && (
        <p className="text-xs text-gray-500 mb-2">Loading available LoRAs...</p>
      )}
      {error && (
        <p className="text-xs text-red-400 mb-2">{error}</p>
      )}

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
            availableLoras={availableLoras}
            onPathChange={(path) => handlePathChange(index, path)}
            onScaleChange={(scale) => handleScaleChange(index, scale)}
            onRemove={() => handleRemove(index)}
            disabled={disabled}
          />
        ))}
      </div>

      {/* Add button */}
      {items.length < maxCount && availableForSelection.length > 0 && (
        <button
          type="button"
          onClick={handleAdd}
          disabled={disabled || isLoading}
          className={cn(
            'mt-2 w-full py-1.5 px-3 text-sm text-gray-400 border border-dashed border-gray-600',
            'rounded-lg hover:border-gray-500 hover:text-gray-300 transition-colors',
            'flex items-center justify-center gap-1',
            (disabled || isLoading) && 'opacity-50 cursor-not-allowed'
          )}
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          Add LoRA
        </button>
      )}

      {/* Help text */}
      {items.length === 0 && !isLoading && (
        <p className="mt-2 text-xs text-gray-500">
          {availableLoras.length > 0
            ? `${availableLoras.length} LoRA${availableLoras.length > 1 ? 's' : ''} available. Click "Add LoRA" to apply style/subject modifications.`
            : 'No LoRAs found in configured directories. Add .safetensors files to loras/ folder.'}
        </p>
      )}
    </div>
  );
}
