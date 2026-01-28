/**
 * Generation Store
 *
 * Manages form values, generation progress, and current result.
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
  // Form values keyed by pipeline ID
  formValues: Record<string, FormValues>;

  // Current generation state
  status: GenerationStatus;
  progress: GenerationProgress | null;
  currentResult: GenerationResult | null;
  error: GenerationError | null;

  // Abort controller for cancellation
  abortController: AbortController | null;

  // Actions
  setFormValue: (pipelineId: string, paramId: string, value: unknown) => void;
  setFormValues: (pipelineId: string, values: FormValues) => void;
  resetFormValues: (pipelineId: string, schema: PipelineSchema) => void;
  restoreFromHistory: (pipelineId: string, params: FormValues) => void;

  generate: (pipelineId: string, endpoint: string, isStreaming: boolean) => Promise<void>;
  cancelGeneration: () => void;
  dismissError: () => void;

  getFormValues: (pipelineId: string, schema: PipelineSchema) => FormValues;
  getTimeEstimate: (pipelineId: string) => TimeEstimate;
}

export const useGenerationStore = create<GenerationState>()(
  immer((set, get) => ({
    // Initial state
    formValues: {},
    status: 'idle',
    progress: null,
    currentResult: null,
    error: null,
    abortController: null,

    // Actions
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

      const values = get().formValues[pipelineId] ?? {};

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

    getFormValues: (pipelineId, schema) => {
      // Build defaults from schema first
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
      }

      // Get stored values
      const stored = get().formValues[pipelineId] ?? {};

      // For Z-Image: Reset variant-sensitive fields when variant differs or stored has no variant
      // This ensures variant-aware defaults (steps, guidance_scale, shift) are correct
      // while preserving user-entered values like prompt
      if (pipelineId === 'zimage' && defaults['_variant']) {
        if (!stored['_variant'] || stored['_variant'] !== defaults['_variant']) {
          // Variant changed or never set - reset only variant-sensitive fields
          const variantSensitiveFields = ['steps', 'guidance_scale', 'shift', '_variant'];
          const result = { ...defaults, ...stored };
          for (const field of variantSensitiveFields) {
            if (defaults[field] !== undefined) {
              result[field] = defaults[field];
            }
          }
          return result;
        }
      }

      // Normal merge: stored values take precedence
      return { ...defaults, ...stored };
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
              outputType: event.output_type ?? 'video',
              urls: event.urls ?? [event.url],
              thumbnailUrl: event.thumbnail_url,
              params: values,
              seed: event.seed ?? -1,
              durationMs: Date.now() - startTime,
              timestamp: Date.now(),
            });
          }
        } catch {
          // Ignore parse errors
        }
      }
    }
  }
}
