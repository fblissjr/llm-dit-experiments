/**
 * Steps Slider
 *
 * Pre-configured slider for inference steps.
 * Defaults differ per pipeline (distilled vs base models).
 */

import { Slider } from './Slider';

interface StepsSliderProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  tooltip?: string;
  disabled?: boolean;
}

export function StepsSlider({
  value,
  onChange,
  min = 1,
  max = 50,
  step = 1,
  tooltip = 'Number of denoising steps. More steps = higher quality but slower.',
  disabled = false,
}: StepsSliderProps) {
  return (
    <Slider
      label="Steps"
      value={value}
      onChange={onChange}
      min={min}
      max={max}
      step={step}
      tooltip={tooltip}
      disabled={disabled}
      formatValue={(v) => Math.round(v).toString()}
    />
  );
}
