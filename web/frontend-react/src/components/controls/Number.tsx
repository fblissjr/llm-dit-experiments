/**
 * Number Input
 *
 * For numeric values with optional min/max/step constraints.
 */

import { useId } from 'react';

interface NumberInputProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  tooltip?: string;
  disabled?: boolean;
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
  className = '',
}: NumberInputProps) {
  const id = useId();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = parseFloat(e.target.value);
    if (!isNaN(newValue)) {
      // Clamp to min/max if provided
      let clamped = newValue;
      if (min !== undefined) clamped = Math.max(min, clamped);
      if (max !== undefined) clamped = Math.min(max, clamped);
      onChange(clamped);
    }
  };

  return (
    <div className={`form-control ${className}`}>
      <label
        htmlFor={id}
        className="form-label"
        title={tooltip}
      >
        {label}
      </label>
      <input
        type="number"
        id={id}
        value={value}
        onChange={handleChange}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        className="form-input"
      />
    </div>
  );
}
