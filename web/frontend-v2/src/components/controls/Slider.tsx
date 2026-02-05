/**
 * Slider Control
 *
 * Range slider with value display, visual fill, and manual input.
 * On mobile, tapping the value opens an input field for precise entry.
 */

import { useId, useState, useRef, useEffect } from 'react';
import { cn } from '@/utils';

interface SliderProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step?: number;
  tooltip?: string;
  disabled?: boolean;
  showValue?: boolean;
  formatValue?: (value: number) => string;
  className?: string;
}

export function Slider({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  tooltip,
  disabled = false,
  showValue = true,
  formatValue = (v) => v.toString(),
  className,
}: SliderProps) {
  const id = useId();
  const [isEditing, setIsEditing] = useState(false);
  const [inputValue, setInputValue] = useState(value.toString());
  const inputRef = useRef<HTMLInputElement>(null);

  // Sync input value when value prop changes (but not while editing)
  useEffect(() => {
    if (!isEditing) {
      setInputValue(value.toString());
    }
  }, [value, isEditing]);

  // Focus input when entering edit mode
  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(parseFloat(e.target.value));
  };

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
      setInputValue(value.toString());
      setIsEditing(false);
    }
  };

  const commitInputValue = () => {
    const parsed = parseFloat(inputValue);
    if (!isNaN(parsed)) {
      // Clamp to valid range
      const clamped = Math.min(max, Math.max(min, parsed));
      // Round to step precision
      const precision = step < 1 ? Math.ceil(-Math.log10(step)) : 0;
      const rounded = Math.round(clamped / step) * step;
      const finalValue = Number(rounded.toFixed(precision));
      onChange(finalValue);
      setInputValue(finalValue.toString());
    } else {
      // Reset to current value if invalid
      setInputValue(value.toString());
    }
  };

  // Calculate fill percentage for visual feedback
  const fillPercent = ((value - min) / (max - min)) * 100;

  return (
    <div className={cn('form-control', className)}>
      <div className="flex items-center justify-between">
        <label htmlFor={id} className="form-label" title={tooltip}>
          {label}
        </label>
        {showValue && (
          <>
            {isEditing ? (
              <input
                ref={inputRef}
                type="number"
                value={inputValue}
                onChange={handleInputChange}
                onBlur={handleInputBlur}
                onKeyDown={handleInputKeyDown}
                min={min}
                max={max}
                step={step}
                disabled={disabled}
                className={cn(
                  'w-20 px-3 py-2 text-sm text-right',
                  'bg-gray-800 border border-blue-500 rounded-lg',
                  'text-gray-200 focus:outline-none',
                  'tabular-nums min-h-[36px]'
                )}
              />
            ) : (
              <button
                type="button"
                onClick={() => !disabled && setIsEditing(true)}
                disabled={disabled}
                className={cn(
                  'slider-value px-3 py-1.5 -mr-2 rounded-lg',
                  'hover:bg-gray-700 active:bg-gray-600 transition-colors',
                  'cursor-pointer min-w-[3.5rem] min-h-[36px] text-right',
                  'flex items-center justify-end',
                  disabled && 'cursor-not-allowed opacity-50'
                )}
                title="Tap to edit value"
              >
                {formatValue(value)}
              </button>
            )}
          </>
        )}
      </div>
      <div className="mt-2">
        <input
          type="range"
          id={id}
          value={value}
          onChange={handleSliderChange}
          min={min}
          max={max}
          step={step}
          disabled={disabled}
          className="slider-track w-full h-2"
          style={{
            background: `linear-gradient(to right, var(--pipeline-color, #3b82f6) 0%, var(--pipeline-color, #3b82f6) ${fillPercent}%, #374151 ${fillPercent}%, #374151 100%)`,
          }}
        />
      </div>
    </div>
  );
}
