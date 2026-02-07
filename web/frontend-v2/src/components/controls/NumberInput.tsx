/**
 * NumberInput Control
 *
 * Numeric input with optional min/max/step constraints.
 * Auto-snaps to the nearest valid step on blur (e.g., typing "1000"
 * for a step=16 field auto-corrects to 992 or 1008).
 */

import { useId } from 'react';
import { cn, snapToStep } from '@/utils';

interface NumberInputProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  tooltip?: string;
  disabled?: boolean;
  error?: string;
  className?: string;
}

export function NumberInput({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  tooltip,
  disabled = false,
  error,
  className,
}: NumberInputProps) {
  const id = useId();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = parseFloat(e.target.value);
    if (!isNaN(newValue)) {
      onChange(newValue);
    }
  };

  const handleBlur = () => {
    if (step && step > 0) {
      const snapped = snapToStep(value, step, min, max);
      if (snapped !== value) {
        onChange(snapped);
      }
    }
  };

  return (
    <div className={cn('form-control', className)}>
      <label htmlFor={id} className="form-label" title={tooltip}>
        {label}
      </label>
      <input
        type="number"
        id={id}
        value={value}
        onChange={handleChange}
        onBlur={handleBlur}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        className={cn(
          'form-input',
          error && 'border-red-500 focus:border-red-500 focus:ring-red-500'
        )}
      />
      {error && <p className="text-sm text-red-400 mt-1">{error}</p>}
    </div>
  );
}
