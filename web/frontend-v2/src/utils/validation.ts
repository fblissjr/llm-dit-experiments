/**
 * Schema-based form validation utilities.
 *
 * Single source of truth for parameter validation. Used by both
 * formStore (store-level validation) and any component-level checks.
 */

import type { ParamSchema, ValidationError, FormValues } from '@/api/types';

/**
 * Validate a single parameter value against its schema.
 *
 * Merges all validation features:
 * - Conditional visibility check (skips hidden params)
 * - Required field check
 * - Number range validation with step alignment
 * - Select validation (skips dynamic options_endpoint selects)
 * - Textarea type check
 */
export function validateParam(
  param: ParamSchema,
  value: unknown,
  formValues?: FormValues
): ValidationError | null {
  // Check conditional visibility first (if formValues provided)
  if (param.conditional && formValues) {
    const isVisible = Object.entries(param.conditional).every(
      ([key, expectedValue]) => formValues[key] === expectedValue
    );
    if (!isVisible) return null;
  }

  // Required check
  if (param.required) {
    if (value === undefined || value === null || value === '') {
      return {
        paramId: param.id,
        message: `${param.label} is required`,
      };
    }
  }

  // Skip further validation if no value
  if (value === undefined || value === null || value === '') return null;

  // Type-specific validation
  switch (param.type) {
    case 'slider':
    case 'number': {
      const num = Number(value);
      if (isNaN(num)) {
        return {
          paramId: param.id,
          message: `${param.label} must be a number`,
        };
      }
      if (param.min !== undefined && num < param.min) {
        return {
          paramId: param.id,
          message: `${param.label} must be at least ${param.min}`,
        };
      }
      if (param.max !== undefined && num > param.max) {
        return {
          paramId: param.id,
          message: `${param.label} must be at most ${param.max}`,
        };
      }
      // Step alignment (e.g., width/height multiples of 64, frames 8n+1)
      if (param.step && param.step > 1) {
        const offset = param.min ?? 0;
        const rounded = Math.round((num - offset) / param.step) * param.step + offset;
        if (rounded !== num) {
          return {
            paramId: param.id,
            message: `${param.label} should be a multiple of ${param.step} (nearest: ${rounded})`,
          };
        }
      }
      break;
    }

    case 'select': {
      // Skip validation for dynamic options loaded from API endpoint
      if (param.options && !param.options_endpoint) {
        if (!param.options.includes(String(value))) {
          return {
            paramId: param.id,
            message: `${param.label} has an invalid value`,
          };
        }
      }
      break;
    }

    case 'textarea': {
      if (typeof value !== 'string') {
        return {
          paramId: param.id,
          message: `${param.label} must be text`,
        };
      }
      break;
    }
  }

  return null;
}

/**
 * Validate all parameters for a pipeline
 */
export function validateForm(
  params: ParamSchema[],
  formValues: FormValues
): ValidationError[] {
  const errors: ValidationError[] = [];

  for (const param of params) {
    const error = validateParam(param, formValues[param.id], formValues);
    if (error) {
      errors.push(error);
    }
  }

  return errors;
}

/**
 * Check if a parameter should be visible based on conditionals
 */
export function isParamVisible(
  param: ParamSchema,
  formValues: FormValues
): boolean {
  if (!param.conditional) return true;

  return Object.entries(param.conditional).every(
    ([key, expectedValue]) => formValues[key] === expectedValue
  );
}
