/**
 * ParamControl Component
 *
 * Type-dispatching component that renders the appropriate control
 * based on the parameter's type from the schema.
 *
 * Wrapped in React.memo with a custom comparator to prevent re-renders
 * when unrelated form values change. This is the single biggest rendering
 * optimization -- without it, all 20+ controls re-render on every
 * single value change.
 */

import { memo } from 'react';
import type { ParamSchema, FormValues, ValidationError } from '@/api/types';
import { logger } from '@/utils/logger';
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
  /** Pipeline ID for context-aware controls (e.g., LoRA filtering). */
  pipelineId?: string;
  /** For textarea controls: called when the inline action button is clicked. */
  onAction?: () => void;
  /** Accessible tooltip/title for the action button. */
  actionLabel?: string;
  /** When true, the action button shows a loading spinner. */
  actionLoading?: boolean;
}

function arePropsEqual(
  prev: Readonly<ParamControlProps>,
  next: Readonly<ParamControlProps>
): boolean {
  // Use param.id (stable string) instead of param reference.
  // PipelineForm creates { ...param, tooltip } for fixed FLUX.2 params,
  // producing a new reference even when content is identical.
  if (prev.param.id !== next.param.id) return false;

  // Value comparison (by value, not reference)
  if (prev.value !== next.value) return false;

  // onChange by reference (stable if PipelineForm memoizes per-param callbacks)
  if (prev.onChange !== next.onChange) return false;

  // disabled by value
  if (prev.disabled !== next.disabled) return false;

  // tooltip may change when isFixed toggles (the spread in PipelineForm)
  if (prev.param.tooltip !== next.param.tooltip) return false;

  // Action button props (only relevant for textarea, but compare for all)
  if (prev.onAction !== next.onAction) return false;
  if (prev.actionLoading !== next.actionLoading) return false;
  if (prev.actionLabel !== next.actionLabel) return false;

  // For formValues: only compare keys that matter for this param's conditional
  if (prev.param.conditional) {
    for (const key of Object.keys(prev.param.conditional)) {
      if (prev.formValues[key] !== next.formValues[key]) return false;
    }
  }

  // For errors: only compare the error for this specific param
  const prevError = prev.errors.find((e) => e.paramId === prev.param.id);
  const nextError = next.errors.find((e) => e.paramId === next.param.id);
  if (prevError?.message !== nextError?.message) return false;

  return true;
}

export const ParamControl = memo(function ParamControl({
  param,
  value,
  onChange,
  formValues,
  errors,
  disabled = false,
  pipelineId,
  onAction,
  actionLabel,
  actionLoading,
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
          onAction={onAction}
          actionLabel={actionLabel}
          actionLoading={actionLoading}
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
          pipelineId={pipelineId}
        />
      );

    default:
      // Fallback for unknown types
      logger('ParamControl').warn(`Unknown param type: ${param.type}`);
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
}, arePropsEqual);
