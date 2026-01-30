/**
 * Shift Slider
 *
 * For flow matching models (Z-Image, FLUX.2).
 * Controls noise schedule timing.
 */

import { Slider } from './Slider';

interface ShiftSliderProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  tooltip?: string;
  disabled?: boolean;
}

export function ShiftSlider({
  value,
  onChange,
  min = 0,
  max = 15,
  step = 0.1,
  tooltip = 'Flow matching shift. Higher = more denoising in early steps.',
  disabled = false,
}: ShiftSliderProps) {
  return (
    <Slider
      label="Shift"
      value={value}
      onChange={onChange}
      min={min}
      max={max}
      step={step}
      tooltip={tooltip}
      disabled={disabled}
      formatValue={(v) => v.toFixed(1)}
    />
  );
}
