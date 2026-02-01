/**
 * Slider Control
 *
 * Range slider with value display and visual fill.
 */

import { useId } from 'react';
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

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(parseFloat(e.target.value));
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
          <span className="slider-value">{formatValue(value)}</span>
        )}
      </div>
      <div className="mt-2">
        <input
          type="range"
          id={id}
          value={value}
          onChange={handleChange}
          min={min}
          max={max}
          step={step}
          disabled={disabled}
          className="slider-track w-full"
          style={{
            background: `linear-gradient(to right, var(--pipeline-color, #3b82f6) 0%, var(--pipeline-color, #3b82f6) ${fillPercent}%, #374151 ${fillPercent}%, #374151 100%)`,
          }}
        />
      </div>
    </div>
  );
}
