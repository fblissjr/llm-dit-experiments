/**
 * Form Store - Form Values and Validation
 *
 * Manages:
 * - Form values per pipeline (user modifications only)
 * - Validation errors per pipeline
 * - Computed resolved values (schema < server < user)
 */

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import type { FormValues, ValidationError, ParamSchema } from '@/api/types';
import { useAppStore } from './appStore';

interface FormState {
  // User-modified values per pipeline
  values: Record<string, FormValues>;

  // Validation errors per pipeline
  errors: Record<string, ValidationError[]>;

  // Actions
  setValue: (pipelineId: string, paramId: string, value: unknown) => void;
  setValues: (pipelineId: string, values: FormValues) => void;
  resetPipeline: (pipelineId: string) => void;
  applyPreset: (pipelineId: string, params: FormValues) => void;
  validate: (pipelineId: string) => ValidationError[];
  clearErrors: (pipelineId: string) => void;

  // Computed
  getResolvedValues: (pipelineId: string) => FormValues;
  getValue: (pipelineId: string, paramId: string) => unknown;
  hasErrors: (pipelineId: string) => boolean;
}

/**
 * Validate a single parameter against its schema
 */
function validateParam(
  param: ParamSchema,
  value: unknown
): ValidationError | null {
  // Required check
  if (param.required && (value === undefined || value === null || value === '')) {
    return { paramId: param.id, message: `${param.label} is required` };
  }

  // Skip further validation if no value
  if (value === undefined || value === null) return null;

  // Number range validation
  if (param.type === 'slider' || param.type === 'number') {
    const num = Number(value);
    if (isNaN(num)) {
      return { paramId: param.id, message: `${param.label} must be a number` };
    }
    if (param.min !== undefined && num < param.min) {
      return { paramId: param.id, message: `${param.label} must be at least ${param.min}` };
    }
    if (param.max !== undefined && num > param.max) {
      return { paramId: param.id, message: `${param.label} must be at most ${param.max}` };
    }
  }

  // Select validation - skip for dynamic options (options_endpoint)
  // Dynamic options are loaded from API and not in param.options
  if (param.type === 'select' && param.options && !param.options_endpoint) {
    if (!param.options.includes(String(value))) {
      return { paramId: param.id, message: `${param.label} has an invalid value` };
    }
  }

  return null;
}

export const useFormStore = create<FormState>()(
  immer((set, get) => ({
    values: {},
    errors: {},

    /**
     * Set a single form value
     */
    setValue: (pipelineId, paramId, value) => {
      set((state) => {
        if (!state.values[pipelineId]) {
          state.values[pipelineId] = {};
        }
        state.values[pipelineId][paramId] = value;

        // Clear error for this param if it exists
        if (state.errors[pipelineId]) {
          state.errors[pipelineId] = state.errors[pipelineId].filter(
            (e) => e.paramId !== paramId
          );
        }
      });
    },

    /**
     * Set multiple form values at once
     */
    setValues: (pipelineId, values) => {
      set((state) => {
        state.values[pipelineId] = {
          ...(state.values[pipelineId] ?? {}),
          ...values,
        };
      });
    },

    /**
     * Reset a pipeline's form to defaults
     */
    resetPipeline: (pipelineId) => {
      set((state) => {
        state.values[pipelineId] = {};
        state.errors[pipelineId] = [];
      });
    },

    /**
     * Apply a preset's parameters to the form
     */
    applyPreset: (pipelineId, params) => {
      set((state) => {
        state.values[pipelineId] = {
          ...(state.values[pipelineId] ?? {}),
          ...params,
        };
      });
    },

    /**
     * Validate all fields for a pipeline
     */
    validate: (pipelineId) => {
      const appStore = useAppStore.getState();
      const pipeline = appStore.pipelines[pipelineId];
      if (!pipeline) return [];

      const resolvedValues = get().getResolvedValues(pipelineId);
      const errors: ValidationError[] = [];

      for (const param of pipeline.params) {
        // Skip conditionally hidden params
        if (param.conditional) {
          const isVisible = Object.entries(param.conditional).every(
            ([key, expectedValue]) => resolvedValues[key] === expectedValue
          );
          if (!isVisible) continue;
        }

        const error = validateParam(param, resolvedValues[param.id]);
        if (error) {
          errors.push(error);
        }
      }

      set((state) => {
        state.errors[pipelineId] = errors;
      });

      return errors;
    },

    clearErrors: (pipelineId) => {
      set((state) => {
        state.errors[pipelineId] = [];
      });
    },

    /**
     * Get resolved values: schema defaults < server defaults < user values
     */
    getResolvedValues: (pipelineId) => {
      const appStore = useAppStore.getState();
      const pipeline = appStore.pipelines[pipelineId];
      const serverDefaults = appStore.serverDefaults[pipelineId] ?? {};
      const userValues = get().values[pipelineId] ?? {};

      if (!pipeline) return { ...serverDefaults, ...userValues };

      // Start with schema defaults
      const result: FormValues = {};
      for (const param of pipeline.params) {
        if (param.default !== undefined) {
          result[param.id] = param.default;
        }
      }

      // Layer server defaults
      Object.assign(result, serverDefaults);

      // Layer user values
      Object.assign(result, userValues);

      return result;
    },

    /**
     * Get a single resolved value
     */
    getValue: (pipelineId, paramId) => {
      return get().getResolvedValues(pipelineId)[paramId];
    },

    hasErrors: (pipelineId) => {
      return (get().errors[pipelineId]?.length ?? 0) > 0;
    },
  }))
);
