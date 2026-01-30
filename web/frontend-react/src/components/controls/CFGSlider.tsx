/**
 * CFG/Guidance Scale Slider
 *
 * Pre-configured slider for classifier-free guidance.
 * Range and defaults vary per pipeline.
 */

import { Slider } from './Slider';

interface CFGSliderProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  tooltip?: string;
  disabled?: boolean;
}

export function CFGSlider({
  value,
  onChange,
  min = 0,
  max = 30,
  step = 0.5,
  tooltip = 'Classifier-free guidance scale. Higher = more prompt adherence.',
  disabled = false,
}: CFGSliderProps) {
  return (
    <Slider
      label="CFG Scale"
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
