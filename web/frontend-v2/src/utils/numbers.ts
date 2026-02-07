/**
 * Numeric utilities for form controls.
 *
 * Shared snap-to-step logic used by Slider and NumberInput to ensure
 * consistent rounding behavior across all numeric inputs.
 */

/**
 * Snap a value to the nearest step, respecting min/max bounds.
 *
 * Uses fixed-point precision to avoid floating-point drift
 * (e.g., 0.1 + 0.2 !== 0.3 in IEEE 754).
 */
export function snapToStep(
  value: number,
  step: number,
  min?: number,
  max?: number
): number {
  const precision = step < 1 ? Math.ceil(-Math.log10(step)) : 0;
  const rounded = Math.round(value / step) * step;
  let snapped = Number(rounded.toFixed(precision));
  if (min !== undefined && snapped < min) snapped = min;
  if (max !== undefined && snapped > max) snapped = max;
  return snapped;
}
