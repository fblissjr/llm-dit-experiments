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
import { persist, createJSONStorage } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import { idbStorage } from '@/utils/idbStorage';
import type { FormValues, ValidationError, PipelineSchema } from '@/api/types';
import { validateParam } from '@/utils/validation';
import { useAppStore } from './appStore';

// Module-level reference-equality cache for getResolvedValues.
// Lives outside Immer state to avoid Object.freeze() overhead.
// Invalidates when any input reference changes (Immer produces new refs on mutation).
interface ResolvedCache {
  result: FormValues;
  userRef: FormValues;
  serverRef: FormValues;
  pipelineRef: PipelineSchema | undefined;
}
const _resolvedCache = new Map<string, ResolvedCache>();

interface ActivePresetState {
  name: string;
  params: FormValues;
}

interface FormState {
  // User-modified values per pipeline
  values: Record<string, FormValues>;

  // Validation errors per pipeline
  errors: Record<string, ValidationError[]>;

  // Tracks which params were explicitly set by the user (not auto-applied)
  // Used by dependent defaults to avoid overriding user intent
  userModified: Record<string, Set<string>>;

  // Active preset per pipeline (name + original params for modification detection)
  activePreset: Record<string, ActivePresetState | null>;

  // Actions
  setValue: (pipelineId: string, paramId: string, value: unknown) => void;
  setValues: (pipelineId: string, values: FormValues) => void;
  resetPipeline: (pipelineId: string) => void;
  applyPreset: (pipelineId: string, presetName: string, params: FormValues) => void;
  clearPreset: (pipelineId: string) => void;
  restorePreset: (pipelineId: string) => void;
  applyDependentDefaults: (pipelineId: string, triggerParamId: string, triggerValue: unknown) => void;
  validate: (pipelineId: string) => ValidationError[];
  clearErrors: (pipelineId: string) => void;

  // Computed
  getResolvedValues: (pipelineId: string) => FormValues;
  getValue: (pipelineId: string, paramId: string) => unknown;
  hasErrors: (pipelineId: string) => boolean;
  isPresetModified: (pipelineId: string) => boolean;
  getActivePresetName: (pipelineId: string) => string | null;
}

export const useFormStore = create<FormState>()(
  persist(
    immer((set, get) => ({
      values: {},
      errors: {},
      userModified: {},
      activePreset: {},

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
        state.activePreset[pipelineId] = null;
      });
    },

    /**
     * Apply a preset's parameters to the form.
     * Records the preset as active and clears userModified for preset-touched
     * params so dependent_defaults can override them later if needed.
     */
    applyPreset: (pipelineId, presetName, params) => {
      set((state) => {
        state.values[pipelineId] = {
          ...(state.values[pipelineId] ?? {}),
          ...params,
        };

        // Record active preset for modification detection
        state.activePreset[pipelineId] = { name: presetName, params: { ...params } };

        // Clear userModified for all params the preset touches.
        // This allows dependent_defaults to override preset values when
        // the trigger changes (e.g., switching model after applying preset).
        if (!state.userModified[pipelineId]) {
          state.userModified[pipelineId] = new Set();
        }
        for (const key of Object.keys(params)) {
          state.userModified[pipelineId].delete(key);
        }
      });
    },

    /**
     * Clear the active preset without changing form values
     */
    clearPreset: (pipelineId) => {
      set((state) => {
        state.activePreset[pipelineId] = null;
      });
    },

    /**
     * Restore the active preset's original values
     */
    restorePreset: (pipelineId) => {
      const preset = get().activePreset[pipelineId];
      if (!preset) return;

      set((state) => {
        state.values[pipelineId] = {
          ...(state.values[pipelineId] ?? {}),
          ...preset.params,
        };

        // Clear userModified for restored params
        if (!state.userModified[pipelineId]) {
          state.userModified[pipelineId] = new Set();
        }
        for (const key of Object.keys(preset.params)) {
          state.userModified[pipelineId].delete(key);
        }
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
        const error = validateParam(param, resolvedValues[param.id], resolvedValues);
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
     * Get resolved values: schema defaults < dependent defaults < server defaults < user values.
     *
     * Uses a module-level reference-equality cache. Immer produces new
     * references on any mutation, so a triple-reference check (pipeline,
     * serverDefaults, userValues) is sufficient for cache invalidation.
     */
    getResolvedValues: (pipelineId) => {
      const appStore = useAppStore.getState();
      const pipeline = appStore.pipelines[pipelineId];
      const serverDefaults = appStore.serverDefaults[pipelineId] ?? {};
      const userValues = get().values[pipelineId] ?? {};

      // Check cache: return immediately if all input references match
      const cached = _resolvedCache.get(pipelineId);
      if (
        cached &&
        cached.userRef === userValues &&
        cached.serverRef === serverDefaults &&
        cached.pipelineRef === pipeline
      ) {
        return cached.result;
      }

      if (!pipeline) {
        const result = { ...serverDefaults, ...userValues };
        _resolvedCache.set(pipelineId, {
          result, userRef: userValues, serverRef: serverDefaults, pipelineRef: undefined,
        });
        return result;
      }

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

      // Clamp number/slider values to schema min/max.
      // Handles stale persisted values from previous schema versions where
      // ranges may have been different.
      for (const param of pipeline.params) {
        if ((param.type === 'slider' || param.type === 'number') && result[param.id] !== undefined) {
          let val = Number(result[param.id]);
          if (!isNaN(val)) {
            if (param.min !== undefined && val < param.min) val = param.min;
            if (param.max !== undefined && val > param.max) val = param.max;
            if (val !== Number(result[param.id])) {
              result[param.id] = val;
            }
          }
        }
      }

      _resolvedCache.set(pipelineId, {
        result, userRef: userValues, serverRef: serverDefaults, pipelineRef: pipeline,
      });

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

    /**
     * Check if the active preset's values have been modified
     */
    isPresetModified: (pipelineId) => {
      const preset = get().activePreset[pipelineId];
      if (!preset) return false;
      const resolved = get().getResolvedValues(pipelineId);
      return Object.entries(preset.params).some(
        ([key, val]) => resolved[key] !== val
      );
    },

    /**
     * Get the active preset name for a pipeline
     */
    getActivePresetName: (pipelineId) => {
      return get().activePreset[pipelineId]?.name ?? null;
    },
  })),
    {
      name: 'llm-dit-form',
      storage: createJSONStorage(() => idbStorage),
      partialize: (state) => ({
        values: state.values,
        activePreset: state.activePreset,
      }),
    }
  )
);
