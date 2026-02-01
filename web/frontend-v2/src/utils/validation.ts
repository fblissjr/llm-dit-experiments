/**
 * Schema-based form validation utilities
 */

import type { ParamSchema, ValidationError, FormValues } from '@/api/types';

/**
 * Validate a single parameter value against its schema
 */
export function validateParam(
  param: ParamSchema,
  value: unknown,
  formValues: FormValues
): ValidationError | null {
  // Check conditional visibility first
  if (param.conditional) {
    const isVisible = Object.entries(param.conditional).every(
      ([key, expectedValue]) => formValues[key] === expectedValue
    );
    if (!isVisible) return null; // Skip validation for hidden fields
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
      break;
    }

    case 'select': {
      if (param.options && !param.options.includes(String(value))) {
        return {
          paramId: param.id,
          message: `${param.label} has an invalid value`,
        };
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
