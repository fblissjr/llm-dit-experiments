/**
 * ParamControl Component
 *
 * Type-dispatching component that renders the appropriate control
 * based on the parameter's type from the schema.
 */

import type { ParamSchema, FormValues, ValidationError } from '@/api/types';
import {
  Textarea,
  Slider,
  NumberInput,
  Select,
  Checkbox,
  ImageUpload,
  ColorPicker,
  LoRAList,
} from '@/components/controls';

interface ParamControlProps {
  param: ParamSchema;
  value: unknown;
  onChange: (value: unknown) => void;
  formValues: FormValues;
  errors: ValidationError[];
  disabled?: boolean;
}

export function ParamControl({
  param,
  value,
  onChange,
  formValues,
  errors,
  disabled = false,
}: ParamControlProps) {
  // Check conditional visibility
  if (param.conditional) {
    const isVisible = Object.entries(param.conditional).every(
      ([key, expectedValue]) => formValues[key] === expectedValue
    );
    if (!isVisible) return null;
  }

  // Find error for this param
  const error = errors.find((e) => e.paramId === param.id)?.message;

  // Dispatch based on param type
  switch (param.type) {
    case 'textarea':
      return (
        <Textarea
          label={param.label}
          value={(value as string) ?? ''}
          onChange={onChange}
          placeholder={param.placeholder}
          rows={param.rows}
          tooltip={param.tooltip}
          required={param.required}
          disabled={disabled}
          error={error}
        />
      );

    case 'slider':
      return (
        <Slider
          label={param.label}
          value={(value as number) ?? param.default ?? param.min ?? 0}
          onChange={onChange}
          min={param.min ?? 0}
          max={param.max ?? 100}
          step={param.step}
          tooltip={param.tooltip}
          disabled={disabled}
        />
      );

    case 'number':
      return (
        <NumberInput
          label={param.label}
          value={(value as number) ?? param.default ?? 0}
          onChange={onChange}
          min={param.min}
          max={param.max}
          step={param.step}
          tooltip={param.tooltip}
          disabled={disabled}
          error={error}
        />
      );

    case 'select':
      return (
        <Select
          label={param.label}
          value={(value as string) ?? param.default ?? ''}
          onChange={onChange}
          options={param.options ?? []}
          optionsEndpoint={param.options_endpoint}
          tooltip={param.tooltip}
          disabled={disabled}
          error={error}
        />
      );

    case 'checkbox':
      return (
        <Checkbox
          label={param.label}
          checked={(value as boolean) ?? param.default ?? false}
          onChange={onChange}
          tooltip={param.tooltip}
          disabled={disabled}
        />
      );

    case 'image':
      return (
        <ImageUpload
          label={param.label}
          value={(value as string) ?? null}
          onChange={onChange}
          tooltip={param.tooltip}
          maxCount={param.max_count}
          disabled={disabled}
        />
      );

    case 'color':
      return (
        <ColorPicker
          label={param.label}
          value={(value as string) ?? '#ffffff'}
          onChange={onChange}
          tooltip={param.tooltip}
          disabled={disabled}
        />
      );

    case 'lora_list':
      return (
        <LoRAList
          param={param}
          value={(value as string[]) ?? []}
          onChange={onChange}
          disabled={disabled}
        />
      );

    default:
      // Fallback for unknown types
      console.warn(`Unknown param type: ${param.type}`);
      return (
        <div className="form-control">
          <label className="form-label">{param.label}</label>
          <input
            type="text"
            value={String(value ?? '')}
            onChange={(e) => onChange(e.target.value)}
            className="form-input"
            disabled={disabled}
          />
        </div>
      );
  }
}
