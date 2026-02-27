/**
 * LoRAItem Component
 *
 * Individual LoRA entry with dropdown selection and scale control.
 * Supports both slider and manual number input for mobile usability.
 * Outputs "path:scale" format string.
 */

import { useId, useState, useRef, useEffect, useCallback } from 'react';
import type { LoRAFile } from '@/api/types';
import { cn } from '@/utils';

const STRENGTH_PRESETS = [0.25, 0.50, 0.75, 1.00];

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
  const [isEditing, setIsEditing] = useState(false);
  const [inputValue, setInputValue] = useState(scale.toFixed(2));
  const inputRef = useRef<HTMLInputElement>(null);
  const longPressTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const longPressIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Sync input value when scale prop changes (but not while editing)
  useEffect(() => {
    if (!isEditing) {
      setInputValue(scale.toFixed(2));
    }
  }, [scale, isEditing]);

  // Focus input when entering edit mode
  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  const handleInputBlur = () => {
    commitInputValue();
    setIsEditing(false);
  };

  const handleInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      commitInputValue();
      setIsEditing(false);
    } else if (e.key === 'Escape') {
      setInputValue(scale.toFixed(2));
      setIsEditing(false);
    }
  };

  const commitInputValue = () => {
    const parsed = parseFloat(inputValue);
    if (!isNaN(parsed)) {
      const clamped = Math.min(scaleMax, Math.max(scaleMin, parsed));
      const rounded = Math.round(clamped * 100) / 100;
      onScaleChange(rounded);
      setInputValue(rounded.toFixed(2));
    } else {
      setInputValue(scale.toFixed(2));
    }
  };

  // Stepper: increment/decrement by 0.05, clamped
  const step = useCallback((delta: number) => {
    const next = Math.round((scale + delta) * 100) / 100;
    const clamped = Math.min(scaleMax, Math.max(scaleMin, next));
    onScaleChange(clamped);
  }, [scale, scaleMin, scaleMax, onScaleChange]);

  // Long-press: initial 0.05 step, then accelerate to 0.10 after 500ms hold
  const startLongPress = useCallback((delta: number) => {
    step(delta);
    longPressTimerRef.current = setTimeout(() => {
      longPressIntervalRef.current = setInterval(() => {
        // Use functional approach to get latest scale via the callback
        step(delta * 2);
      }, 100);
    }, 500);
  }, [step]);

  const stopLongPress = useCallback(() => {
    if (longPressTimerRef.current) {
      clearTimeout(longPressTimerRef.current);
      longPressTimerRef.current = null;
    }
    if (longPressIntervalRef.current) {
      clearInterval(longPressIntervalRef.current);
      longPressIntervalRef.current = null;
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => stopLongPress, [stopLongPress]);

  // Calculate fill percentage for slider visual
  const fillPercent = ((scale - scaleMin) / (scaleMax - scaleMin)) * 100;

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
    <div className="p-3 bg-gray-800/50 rounded-lg border border-gray-700 space-y-3">
      {/* Top row: Index, LoRA selector, Remove button */}
      <div className="flex items-center gap-2">
        {/* Index badge */}
        <span className="text-xs text-gray-500 w-5 text-center shrink-0">{index + 1}</span>

        {/* LoRA selector dropdown */}
        <div className="flex-1 min-w-0">
          <select
            id={selectId}
            value={path}
            onChange={(e) => onPathChange(e.target.value)}
            disabled={disabled}
            className={cn(
              'w-full px-3 py-2 text-sm bg-gray-900 border border-gray-600 rounded-lg',
              'text-gray-200 cursor-pointer',
              'focus:outline-hidden focus:ring-1 focus:ring-blue-500 focus:border-blue-500',
              disabled && 'opacity-50 cursor-not-allowed',
              !path && 'text-gray-500'
            )}
          >
            <option value="">Select a LoRA...</option>
            {directories.length === 1 ? (
              // Single directory - flat list
              availableLoras.map((lora) => (
                <option key={lora.path} value={lora.path}>
                  {lora.name} ({lora.sizeMb}MB)
                </option>
              ))
            ) : (
              // Multiple directories - grouped
              directories.map((dir) => (
                <optgroup key={dir} label={dir}>
                  {lorasByDirectory[dir].map((lora) => (
                    <option key={lora.path} value={lora.path}>
                      {lora.name} ({lora.sizeMb}MB)
                    </option>
                  ))}
                </optgroup>
              ))
            )}
          </select>
        </div>

        {/* Remove button - 44px minimum touch target for mobile */}
        <button
          type="button"
          onClick={onRemove}
          disabled={disabled}
          className={cn(
            'p-2.5 text-gray-400 hover:text-red-400 hover:bg-red-500/10 active:bg-red-500/20 rounded-lg transition-colors shrink-0',
            'min-w-touch min-h-touch flex items-center justify-center',
            disabled && 'opacity-50 cursor-not-allowed'
          )}
          title="Remove LoRA"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Strength controls (visible when a LoRA is selected) */}
      {path && (
        <div className="space-y-2 pl-5">
          {/* Row 1: Stepper [-] value [+] */}
          <div className="flex items-center gap-2">
            <label htmlFor={scaleId} className="text-xs text-gray-400 shrink-0 w-14">
              Strength
            </label>

            {/* Decrement button -- 44px touch target */}
            <button
              type="button"
              disabled={disabled || scale <= scaleMin}
              onMouseDown={() => startLongPress(-0.05)}
              onMouseUp={stopLongPress}
              onMouseLeave={stopLongPress}
              onTouchStart={() => startLongPress(-0.05)}
              onTouchEnd={stopLongPress}
              className={cn(
                'min-w-touch min-h-touch flex items-center justify-center',
                'bg-gray-700 hover:bg-gray-600 active:bg-gray-500 rounded-lg transition-colors',
                'text-gray-200 text-lg font-medium select-none',
                (disabled || scale <= scaleMin) && 'opacity-30 cursor-not-allowed'
              )}
            >
              -
            </button>

            {/* Editable value display */}
            {isEditing ? (
              <input
                ref={inputRef}
                type="number"
                value={inputValue}
                onChange={handleInputChange}
                onBlur={handleInputBlur}
                onKeyDown={handleInputKeyDown}
                min={scaleMin}
                max={scaleMax}
                step={0.05}
                disabled={disabled}
                className={cn(
                  'w-18 px-3 py-2 text-sm text-center',
                  'bg-gray-800 border border-blue-500 rounded-lg',
                  'text-gray-200 focus:outline-hidden',
                  'tabular-nums min-h-[40px]'
                )}
              />
            ) : (
              <button
                type="button"
                onClick={() => !disabled && setIsEditing(true)}
                disabled={disabled}
                className={cn(
                  'w-18 px-3 py-2 text-sm text-center tabular-nums',
                  'bg-gray-800 border border-gray-600 rounded-lg',
                  'text-gray-300 hover:border-gray-500 hover:bg-gray-700 active:bg-gray-600 transition-colors',
                  'min-h-[40px] flex items-center justify-center',
                  disabled && 'cursor-not-allowed opacity-50'
                )}
                title="Tap to edit value"
              >
                {scale.toFixed(2)}
              </button>
            )}

            {/* Increment button -- 44px touch target */}
            <button
              type="button"
              disabled={disabled || scale >= scaleMax}
              onMouseDown={() => startLongPress(0.05)}
              onMouseUp={stopLongPress}
              onMouseLeave={stopLongPress}
              onTouchStart={() => startLongPress(0.05)}
              onTouchEnd={stopLongPress}
              className={cn(
                'min-w-touch min-h-touch flex items-center justify-center',
                'bg-gray-700 hover:bg-gray-600 active:bg-gray-500 rounded-lg transition-colors',
                'text-gray-200 text-lg font-medium select-none',
                (disabled || scale >= scaleMax) && 'opacity-30 cursor-not-allowed'
              )}
            >
              +
            </button>
          </div>

          {/* Row 2: Quick-set preset pills */}
          <div className="flex items-center gap-1.5">
            <span className="text-xs text-gray-500 w-14 shrink-0">Presets</span>
            {STRENGTH_PRESETS.filter((p) => p >= scaleMin && p <= scaleMax).map((preset) => {
              const isActive = Math.abs(scale - preset) < 0.001;
              return (
                <button
                  key={preset}
                  type="button"
                  onClick={() => !disabled && onScaleChange(preset)}
                  disabled={disabled}
                  className={cn(
                    'px-3 py-1.5 text-xs font-medium rounded-lg transition-colors tabular-nums',
                    'min-h-[32px]',
                    isActive
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700/50 text-gray-400 hover:bg-gray-700 hover:text-gray-200 active:bg-gray-600',
                    disabled && 'opacity-50 cursor-not-allowed'
                  )}
                >
                  {preset.toFixed(2)}
                </button>
              );
            })}
          </div>

          {/* Row 3: Slider (secondary control, visual feedback) */}
          <div className="hidden md:flex items-center gap-2">
            <span className="text-xs text-gray-500 w-14 shrink-0" />
            <input
              id={scaleId}
              type="range"
              value={scale}
              onChange={(e) => onScaleChange(parseFloat(e.target.value))}
              min={scaleMin}
              max={scaleMax}
              step={0.05}
              disabled={disabled}
              className={cn(
                'flex-1 h-2 cursor-pointer rounded-full appearance-none min-w-[100px]',
                disabled && 'opacity-50 cursor-not-allowed'
              )}
              style={{
                background: `linear-gradient(to right, var(--pipeline-color, #3b82f6) 0%, var(--pipeline-color, #3b82f6) ${fillPercent}%, #374151 ${fillPercent}%, #374151 100%)`,
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
