/**
 * Form Store - Form Values, Validation, and Dependent Defaults
 *
 * Manages:
 * - Form values per pipeline (user modifications only)
 * - Validation errors per pipeline
 * - Computed resolved values (schema < dependent_defaults < server < user)
 * - User modification tracking for dependent defaults
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

  // Tracks which params were explicitly set by the user (not auto-applied)
  // Used by dependent defaults to avoid overriding user intent
  userModified: Record<string, Set<string>>;

  // Actions
  setValue: (pipelineId: string, paramId: string, value: unknown) => void;
  setValues: (pipelineId: string, values: FormValues) => void;
  resetPipeline: (pipelineId: string) => void;
  applyPreset: (pipelineId: string, params: FormValues) => void;
  applyDependentDefaults: (pipelineId: string, triggerParamId: string, triggerValue: unknown) => void;
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
    // Step alignment warning (e.g., width/height must be multiples of 16/32)
    if (param.step && param.step > 1) {
      const rounded = Math.round(num / param.step) * param.step;
      if (rounded !== num) {
        return {
          paramId: param.id,
          message: `${param.label} should be a multiple of ${param.step} (nearest: ${rounded})`,
        };
      }
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
    userModified: {},

    /**
     * Set a single form value and mark it as user-modified
     */
    setValue: (pipelineId, paramId, value) => {
      set((state) => {
        if (!state.values[pipelineId]) {
          state.values[pipelineId] = {};
        }
        state.values[pipelineId][paramId] = value;

        // Track that the user explicitly modified this param
        if (!state.userModified[pipelineId]) {
          state.userModified[pipelineId] = new Set();
        }
        state.userModified[pipelineId].add(paramId);

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
        state.userModified[pipelineId] = new Set();
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
     * Apply dependent defaults when a trigger param changes.
     *
     * Finds all params with dependent_defaults keyed on triggerParamId,
     * then updates their values -- but only if the user hasn't manually
     * modified them.
     */
    applyDependentDefaults: (pipelineId, triggerParamId, triggerValue) => {
      const appStore = useAppStore.getState();
      const pipeline = appStore.pipelines[pipelineId];
      if (!pipeline) return;

      const triggerStr = String(triggerValue);

      set((state) => {
        if (!state.values[pipelineId]) {
          state.values[pipelineId] = {};
        }

        const modified = state.userModified[pipelineId] ?? new Set();

        for (const param of pipeline.params) {
          if (!param.dependent_defaults) continue;

          const mapping = param.dependent_defaults[triggerParamId];
          if (!mapping) continue;

          const newDefault = mapping[triggerStr];
          if (newDefault === undefined) continue;

          // Only apply if user hasn't manually modified this param
          if (!modified.has(param.id)) {
            state.values[pipelineId][param.id] = newDefault;
          }
        }
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
     * Get resolved values: schema defaults < dependent defaults < server defaults < user values
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

      // Layer dependent defaults (between schema and server)
      // Uses current resolved trigger values to compute dependent defaults
      for (const param of pipeline.params) {
        if (!param.dependent_defaults) continue;

        for (const [triggerParamId, mapping] of Object.entries(param.dependent_defaults)) {
          // Look up trigger value from user values, then server defaults, then schema defaults
          const triggerValue = String(
            userValues[triggerParamId] ??
            serverDefaults[triggerParamId] ??
            result[triggerParamId] ??
            ''
          );
          const depDefault = mapping[triggerValue];
          if (depDefault !== undefined) {
            result[param.id] = depDefault;
          }
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
