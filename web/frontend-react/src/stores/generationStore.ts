/**
 * Generation Store
 *
 * Manages form values, generation progress, and current result.
 *
 * IMPORTANT: This store uses Immer middleware. When updating form values,
 * Immer handles immutability automatically. For controlled inputs like sliders
 * to work properly, avoid calling functions that create new objects on every
 * render - prefer selectors that return stable references.
 */

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import type {
  FormValues,
  GenerationStatus,
  GenerationProgress,
  GenerationResult,
  GenerationError,
  TimeEstimate,
  PipelineSchema,
} from '@/types';
import { useHistoryStore } from './historyStore';
import { useUIStore } from './uiStore';
import { usePipelineStore } from './pipelineStore';

interface GenerationState {
  // Form values keyed by pipeline ID (stores user-modified values only)
  formValues: Record<string, FormValues>;

  // Tracks which pipelines have been initialized
  initializedPipelines: Record<string, boolean>;

  // Current generation state
  status: GenerationStatus;
  progress: GenerationProgress | null;
  currentResult: GenerationResult | null;
  error: GenerationError | null;

  // Abort controller for cancellation
  abortController: AbortController | null;

  // Actions
  initializeFormValues: (pipelineId: string, schema: PipelineSchema) => void;
  setFormValue: (pipelineId: string, paramId: string, value: unknown) => void;
  setFormValues: (pipelineId: string, values: FormValues) => void;
  resetFormValues: (pipelineId: string, schema: PipelineSchema) => void;
  restoreFromHistory: (pipelineId: string, params: FormValues) => void;

  generate: (pipelineId: string, endpoint: string, isStreaming: boolean) => Promise<void>;
  cancelGeneration: () => void;
  dismissError: () => void;

  getFormValues: (pipelineId: string, schema: PipelineSchema) => FormValues;
  getTimeEstimate: (pipelineId: string) => TimeEstimate;
  isInitialized: (pipelineId: string) => boolean;
}

export const useGenerationStore = create<GenerationState>()(
  immer((set, get) => ({
    // Initial state
    formValues: {},
    initializedPipelines: {},
    status: 'idle',
    progress: null,
    currentResult: null,
    error: null,
    abortController: null,

    // Actions

    /**
     * Initialize form values for a pipeline. Called once per pipeline lifecycle.
     * This builds defaults from schema + server, persists _variant immediately,
     * and merges with any existing user values.
     */
    initializeFormValues: (pipelineId, schema) => {
      // Skip if already initialized
      if (get().initializedPipelines[pipelineId]) {
        return;
      }

      // Build defaults from schema
      const defaults: FormValues = {};
      for (const param of schema.params) {
        if (param.default !== undefined) {
          defaults[param.id] = param.default;
        }
      }

      // Inject server-side defaults for Z-Image (variant-aware values)
      if (pipelineId === 'zimage') {
        const serverDefaults = usePipelineStore.getState().serverDefaults;
        if (serverDefaults.zimage_variant) {
          defaults['_variant'] = serverDefaults.zimage_variant;
        }
        // Apply variant-aware defaults from server (overrides schema defaults)
        if (serverDefaults.steps !== undefined) {
          defaults['steps'] = serverDefaults.steps;
        }
        if (serverDefaults.guidance_scale !== undefined) {
          defaults['guidance_scale'] = serverDefaults.guidance_scale;
        }
        if (serverDefaults.shift !== undefined) {
          defaults['shift'] = serverDefaults.shift;
        }
        if (serverDefaults.d_noise !== undefined) {
          defaults['d_noise'] = serverDefaults.d_noise;
        }
        if (serverDefaults.dynamic_shift !== undefined) {
          defaults['dynamic_shift'] = serverDefaults.dynamic_shift;
        }
      }

      // Get any existing stored values (preserves user input across re-initialization)
      const stored = get().formValues[pipelineId] ?? {};

      set((state) => {
        // Merge: defaults first, then stored values take precedence
        state.formValues[pipelineId] = { ...defaults, ...stored };
        // Mark as initialized
        state.initializedPipelines[pipelineId] = true;
      });
    },

    setFormValue: (pipelineId, paramId, value) => {
      set((state) => {
        if (!state.formValues[pipelineId]) {
          state.formValues[pipelineId] = {};
        }
        state.formValues[pipelineId][paramId] = value;
      });
    },

    setFormValues: (pipelineId, values) => {
      set((state) => {
        state.formValues[pipelineId] = { ...state.formValues[pipelineId], ...values };
      });
    },

    resetFormValues: (pipelineId, schema) => {
      const defaults: FormValues = {};
      schema.params.forEach((param) => {
        if (param.default !== undefined) {
          defaults[param.id] = param.default;
        }
      });

      // Inject server-side defaults for Z-Image (variant-aware values)
      if (pipelineId === 'zimage') {
        const serverDefaults = usePipelineStore.getState().serverDefaults;
        if (serverDefaults.zimage_variant) {
          defaults['_variant'] = serverDefaults.zimage_variant;
        }
        // Apply variant-aware defaults from server (overrides schema defaults)
        if (serverDefaults.steps !== undefined) {
          defaults['steps'] = serverDefaults.steps;
        }
        if (serverDefaults.guidance_scale !== undefined) {
          defaults['guidance_scale'] = serverDefaults.guidance_scale;
        }
        if (serverDefaults.shift !== undefined) {
          defaults['shift'] = serverDefaults.shift;
        }
        if (serverDefaults.d_noise !== undefined) {
          defaults['d_noise'] = serverDefaults.d_noise;
        }
        if (serverDefaults.dynamic_shift !== undefined) {
          defaults['dynamic_shift'] = serverDefaults.dynamic_shift;
        }
      }

      set((state) => {
        state.formValues[pipelineId] = defaults;
      });
    },

    restoreFromHistory: (pipelineId, params) => {
      set((state) => {
        state.formValues[pipelineId] = { ...params };
      });
    },

    generate: async (pipelineId, endpoint, isStreaming) => {
      const abortController = new AbortController();
      const startTime = Date.now();

      set((state) => {
        state.status = 'generating';
        state.progress = null;
        state.currentResult = null;
        state.error = null;
        state.abortController = abortController;
      });

      // Get merged form values (defaults + stored) - need pipeline schema
      const pipelineState = usePipelineStore.getState();
      const pipeline = pipelineState.pipelines?.[pipelineId];
      const values = pipeline
        ? get().getFormValues(pipelineId, pipeline)
        : get().formValues[pipelineId] ?? {};

      try {
        if (isStreaming) {
          // SSE streaming for LTX-2
          await handleStreamingGeneration(
            endpoint,
            values,
            abortController.signal,
            (progress) => {
              set((state) => {
                state.progress = progress;
              });
            },
            (result) => {
              set((state) => {
                state.status = 'completed';
                state.currentResult = result;
                state.abortController = null;
              });
              // Add to history
              useHistoryStore.getState().addItem(result);
            }
          );
        } else {
          // Standard POST request
          const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(values),
            signal: abortController.signal,
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error ?? 'Generation failed');
          }

          const data = await response.json();
          const result: GenerationResult = {
            id: data.id ?? `gen-${Date.now()}`,
            pipelineId,
            outputType: data.output_type ?? 'image',
            urls: data.urls ?? [data.url],
            thumbnailUrl: data.thumbnail_url ?? data.urls?.[0] ?? data.url,
            params: values,
            seed: data.seed ?? -1,
            durationMs: Date.now() - startTime,
            timestamp: Date.now(),
          };

          set((state) => {
            state.status = 'completed';
            state.currentResult = result;
            state.abortController = null;
          });

          // Add to history
          useHistoryStore.getState().addItem(result);
        }
      } catch (error) {
        if (error instanceof Error && error.name === 'AbortError') {
          set((state) => {
            state.status = 'cancelled';
            state.abortController = null;
          });
          return;
        }

        const errorMessage = error instanceof Error ? error.message : 'Generation failed';

        set((state) => {
          state.status = 'error';
          state.error = {
            message: errorMessage,
            recoverable: true,
          };
          state.abortController = null;
        });

        // Show error notification (5 second duration)
        useUIStore.getState().addNotification({
          type: 'error',
          message: errorMessage,
          duration: 5000,
        });
      }
    },

    cancelGeneration: () => {
      const { abortController } = get();
      if (abortController) {
        abortController.abort();
      }
    },

    dismissError: () => {
      set((state) => {
        state.status = 'idle';
        state.error = null;
      });
    },

    /**
     * Get merged form values for a pipeline.
     * PURE FUNCTION: no side effects, no conditional resets.
     * Initialization logic moved to initializeFormValues().
     */
    getFormValues: (pipelineId, schema) => {
      // Build defaults from schema first
      const defaults: FormValues = {};
      schema.params.forEach((param) => {
        if (param.default !== undefined) {
          defaults[param.id] = param.default;
        }
      });

      // Inject server-side defaults for Z-Image (variant-aware values)
      // These are used as fallbacks if not yet initialized
      if (pipelineId === 'zimage') {
        const serverDefaults = usePipelineStore.getState().serverDefaults;
        if (serverDefaults.zimage_variant) {
          defaults['_variant'] = serverDefaults.zimage_variant;
        }
        if (serverDefaults.steps !== undefined) {
          defaults['steps'] = serverDefaults.steps;
        }
        if (serverDefaults.guidance_scale !== undefined) {
          defaults['guidance_scale'] = serverDefaults.guidance_scale;
        }
        if (serverDefaults.shift !== undefined) {
          defaults['shift'] = serverDefaults.shift;
        }
        if (serverDefaults.d_noise !== undefined) {
          defaults['d_noise'] = serverDefaults.d_noise;
        }
        if (serverDefaults.dynamic_shift !== undefined) {
          defaults['dynamic_shift'] = serverDefaults.dynamic_shift;
        }
      }

      // Get stored values and merge: stored takes precedence over defaults
      const stored = get().formValues[pipelineId] ?? {};
      return { ...defaults, ...stored };
    },

    /**
     * Check if a pipeline has been initialized.
     */
    isInitialized: (pipelineId) => {
      return get().initializedPipelines[pipelineId] === true;
    },

    getTimeEstimate: (pipelineId: string) => {
      // Get history items for this pipeline
      const historyItems = useHistoryStore.getState().getItemsByPipeline(pipelineId);

      if (historyItems.length === 0) {
        // No history - return low confidence default
        return {
          estimatedSeconds: 30,
          basedOn: 'default' as const,
          confidence: 'low' as const,
        };
      }

      // Get the current form values to compare similar configurations
      const currentValues = get().formValues[pipelineId] ?? {};
      const currentSteps = Number(currentValues.steps ?? currentValues.num_inference_steps ?? 20);

      // Weight recent items more heavily and prefer similar step counts
      const recentItems = historyItems.slice(0, 10); // Last 10 items
      let totalWeight = 0;
      let weightedDuration = 0;

      for (const item of recentItems) {
        const itemSteps = Number(
          item.params.steps ?? item.params.num_inference_steps ?? 20
        );
        const durationMs = item.result.durationMs;

        if (durationMs > 0) {
          // Base weight: 1.0 for most recent, decaying for older
          const recencyIndex = recentItems.indexOf(item);
          const recencyWeight = 1.0 - recencyIndex * 0.1;

          // Steps similarity bonus (more weight if steps are similar)
          const stepsDiff = Math.abs(currentSteps - itemSteps);
          const stepsWeight = stepsDiff <= 5 ? 1.5 : stepsDiff <= 10 ? 1.2 : 1.0;

          const weight = recencyWeight * stepsWeight;
          weightedDuration += durationMs * weight;
          totalWeight += weight;
        }
      }

      if (totalWeight === 0) {
        return {
          estimatedSeconds: 30,
          basedOn: 'default' as const,
          confidence: 'low' as const,
        };
      }

      const estimatedMs = weightedDuration / totalWeight;
      const estimatedSeconds = Math.round(estimatedMs / 1000);

      // Determine confidence based on sample size and recency
      let confidence: 'low' | 'medium' | 'high' = 'low';
      if (recentItems.length >= 5) {
        confidence = 'high';
      } else if (recentItems.length >= 2) {
        confidence = 'medium';
      }

      return {
        estimatedSeconds,
        basedOn: 'history' as const,
        confidence,
      };
    },
  }))
);

/**
 * Selector: Get raw form values for a pipeline (without merging defaults).
 * Returns the stored values object directly, providing a stable reference
 * that only changes when values are actually modified.
 *
 * Use this in components to get form values, then merge with defaults
 * at the point of use (e.g., when reading a specific value).
 */
// Stable empty object to avoid creating new references on every selector call
const EMPTY_FORM_VALUES: FormValues = {};

export function selectFormValues(
  state: GenerationState,
  pipelineId: string
): FormValues {
  return state.formValues[pipelineId] ?? EMPTY_FORM_VALUES;
}

/**
 * Helper: Get a single form value with fallback to default.
 * Use this when reading individual values to avoid creating new objects.
 */
export function getFormValue(
  formValues: FormValues,
  param: { id: string; default?: unknown },
  serverDefault?: unknown
): unknown {
  if (formValues[param.id] !== undefined) {
    return formValues[param.id];
  }
  if (serverDefault !== undefined) {
    return serverDefault;
  }
  return param.default;
}

/**
 * Handle SSE streaming generation (for LTX-2)
 */
async function handleStreamingGeneration(
  endpoint: string,
  values: FormValues,
  signal: AbortSignal,
  onProgress: (progress: GenerationProgress) => void,
  onComplete: (result: GenerationResult) => void
): Promise<void> {
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(values),
    signal,
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error ?? 'Streaming generation failed');
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('No response body');
  }

  const decoder = new TextDecoder();
  let buffer = '';
  const startTime = Date.now();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6);
        if (data === '[DONE]') continue;

        try {
          const event = JSON.parse(data);

          if (event.type === 'progress') {
            onProgress({
              step: event.step,
              totalSteps: event.total_steps,
              percent: (event.step / event.total_steps) * 100,
              elapsedMs: Date.now() - startTime,
              estimatedRemainingMs: event.estimated_remaining_ms,
              message: event.message,
            });
          } else if (event.type === 'complete') {
            onComplete({
              id: event.id ?? `gen-${Date.now()}`,
              pipelineId: event.pipeline_id,
              outputType: event.output_type ?? 'image',
              urls: event.urls ?? [event.url],
              thumbnailUrl: event.thumbnail_url,
              params: values,
              seed: event.seed ?? -1,
              durationMs: Date.now() - startTime,
              timestamp: Date.now(),
            });
          } else if (event.type === 'error') {
            throw new Error(event.message ?? 'Generation failed');
          }
        } catch {
          // Ignore parse errors
        }
      }
    }
  }
}
