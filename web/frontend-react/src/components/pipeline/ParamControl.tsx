/**
 * Parameter Control
 *
 * Renders the appropriate control component based on ParamSchema type.
 * This is the core of the schema-driven form system.
 */

import type { ParamSchema, FormValues } from '@/types';
import {
  Slider,
  PromptTextarea,
  NumberInput,
  Select,
  Checkbox,
  ImageUpload,
} from '../controls';

interface ParamControlProps {
  param: ParamSchema;
  value: unknown;
  onChange: (value: unknown) => void;
  formValues: FormValues;
  disabled?: boolean;
}

export function ParamControl({
  param,
  value,
  onChange,
  formValues,
  disabled = false,
}: ParamControlProps) {
  // Check conditional visibility
  if (param.conditional) {
    const isVisible = Object.entries(param.conditional).every(
      ([key, expectedValue]) => formValues[key] === expectedValue
    );
    if (!isVisible) return null;
  }

  switch (param.type) {
    case 'textarea':
      return (
        <PromptTextarea
          label={param.label}
          value={(value as string) ?? ''}
          onChange={(v) => onChange(v)}
          placeholder={param.placeholder}
          rows={param.rows ?? 4}
          tooltip={param.tooltip}
          required={param.required}
          disabled={disabled}
        />
      );

    case 'slider':
      return (
        <Slider
          label={param.label}
          value={(value as number) ?? param.default ?? param.min ?? 0}
          onChange={(v) => onChange(v)}
          min={param.min ?? 0}
          max={param.max ?? 100}
          step={param.step ?? 1}
          tooltip={param.tooltip}
          disabled={disabled}
          formatValue={(v) => {
            // Format based on step precision
            const step = param.step ?? 1;
            if (step < 1) {
              const decimals = Math.abs(Math.floor(Math.log10(step)));
              return v.toFixed(decimals);
            }
            return Math.round(v).toString();
          }}
        />
      );

    case 'number':
      return (
        <NumberInput
          label={param.label}
          value={(value as number) ?? param.default ?? param.min ?? 0}
          onChange={(v) => onChange(v)}
          min={param.min}
          max={param.max}
          step={param.step ?? 1}
          tooltip={param.tooltip}
          disabled={disabled}
        />
      );

    case 'select':
      return (
        <Select
          label={param.label}
          value={(value as string) ?? param.default ?? param.options?.[0] ?? ''}
          onChange={(v) => onChange(v)}
          options={param.options ?? []}
          tooltip={param.tooltip}
          disabled={disabled}
        />
      );

    case 'checkbox':
      return (
        <Checkbox
          label={param.label}
          checked={(value as boolean) ?? param.default ?? false}
          onChange={(v) => onChange(v)}
          tooltip={param.tooltip}
          disabled={disabled}
        />
      );

    case 'image':
      return (
        <ImageUpload
          label={param.label}
          value={(value as string | string[] | null) ?? null}
          onChange={(v) => onChange(v)}
          tooltip={param.tooltip}
          disabled={disabled}
        />
      );

    case 'color':
      // Simple color picker
      return (
        <div className="form-control">
          <label className="form-label" title={param.tooltip}>
            {param.label}
          </label>
          <input
            type="color"
            value={(value as string) ?? param.default ?? '#000000'}
            onChange={(e) => onChange(e.target.value)}
            disabled={disabled}
            className="w-full h-10 rounded-lg cursor-pointer"
          />
        </div>
      );

    default:
      console.warn(`Unknown param type: ${param.type}`);
      return null;
  }
}
