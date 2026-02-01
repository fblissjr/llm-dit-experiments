/**
 * Session Store - Generation State and History
 *
 * Manages:
 * - Current generation status
 * - Progress during generation
 * - Generation results
 * - History with localStorage persistence
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import type {
  GenerationResult,
  HistoryItem,
  FormValues,
} from '@/api/types';
import { generate, generateStream } from '@/api/client';
import { generateUUID } from '@/utils';
import { useAppStore } from './appStore';
import { useFormStore } from './formStore';

type GenerationStatus = 'idle' | 'generating' | 'completed' | 'error';

interface SessionState {
  // Generation state
  status: GenerationStatus;
  progress: { step: number; total: number; message?: string } | null;
  result: GenerationResult | null;
  error: string | null;

  // History
  history: HistoryItem[];

  // Actions
  startGeneration: (pipelineId: string) => Promise<void>;
  cancelGeneration: () => void;
  clearResult: () => void;
  clearHistory: () => void;
  removeHistoryItem: (id: string) => void;
  loadHistoryParams: (item: HistoryItem) => void;

  // Internal
  addToHistory: (
    result: GenerationResult,
    pipeline: { name: string; color: string },
    params: FormValues
  ) => void;
}

const MAX_HISTORY_ITEMS = 50;

/**
 * Extract short prompt for history card
 */
function truncatePrompt(prompt: string, maxLength = 50): string {
  if (!prompt) return '';
  if (prompt.length <= maxLength) return prompt;
  return prompt.substring(0, maxLength - 3) + '...';
}

/**
 * Extract key params for history card display
 */
function extractKeyParams(params: FormValues): string {
  const parts: string[] = [];

  const steps = params.steps ?? params.num_inference_steps ?? params.num_steps;
  if (steps !== undefined) {
    parts.push(`${steps} steps`);
  }

  const cfg = params.guidance_scale ?? params.guidance;
  if (cfg !== undefined) {
    parts.push(`CFG ${cfg}`);
  }

  const width = params.width;
  const height = params.height;
  if (width && height) {
    parts.push(`${width}x${height}`);
  }

  return parts.join(' / ') || 'Default settings';
}

// Abort controller for cancellation
let abortController: AbortController | null = null;

export const useSessionStore = create<SessionState>()(
  persist(
    immer((set, get) => ({
      status: 'idle',
      progress: null,
      result: null,
      error: null,
      history: [],

      /**
       * Start a generation for the given pipeline
       */
      startGeneration: async (pipelineId) => {
        const appStore = useAppStore.getState();
        const formStore = useFormStore.getState();

        const pipeline = appStore.pipelines[pipelineId];
        if (!pipeline) {
          set((state) => {
            state.error = 'Pipeline not found';
            state.status = 'error';
          });
          return;
        }

        // Validate form
        const errors = formStore.validate(pipelineId);
        if (errors.length > 0) {
          set((state) => {
            state.error = errors[0].message;
            state.status = 'error';
          });
          return;
        }

        // Get resolved parameters
        const params = formStore.getResolvedValues(pipelineId);

        // Reset state
        set((state) => {
          state.status = 'generating';
          state.progress = null;
          state.result = null;
          state.error = null;
        });

        // Create abort controller
        abortController = new AbortController();

        const startTime = Date.now();

        try {
          if (pipeline.supports_streaming) {
            // Use SSE streaming for video/progress pipelines
            for await (const event of generateStream(pipeline.endpoint, params)) {
              // Check for cancellation
              if (abortController?.signal.aborted) {
                set((state) => {
                  state.status = 'idle';
                  state.progress = null;
                });
                return;
              }

              if (event.type === 'progress') {
                set((state) => {
                  state.progress = {
                    step: event.step,
                    total: event.total,
                    message: event.message,
                  };
                });
              } else if (event.type === 'result') {
                // Server may return urls (array), url (string), or output_path (string)
                const eventData = event.data as unknown as {
                  urls?: string[];
                  url?: string;
                  output_path?: string;
                  seed?: number;
                };
                const urls = eventData.urls
                  ?? (eventData.url ? [eventData.url] : null)
                  ?? (eventData.output_path ? [eventData.output_path] : []);

                const result: GenerationResult = {
                  id: generateUUID(),
                  pipelineId,
                  outputType: pipeline.output_type,
                  urls,
                  seed: (eventData.seed ?? params.seed ?? -1) as number,
                  params,
                  durationMs: Date.now() - startTime,
                  timestamp: Date.now(),
                };

                set((state) => {
                  state.status = 'completed';
                  state.result = result;
                  state.progress = null;
                });

                // Add to history
                get().addToHistory(result, pipeline, params);
              } else if (event.type === 'error') {
                set((state) => {
                  state.status = 'error';
                  state.error = event.error;
                  state.progress = null;
                });
              }
            }
          } else {
            // Standard POST request
            const response = await generate(pipeline.endpoint, params);

            const result: GenerationResult = {
              id: generateUUID(),
              pipelineId,
              outputType: pipeline.output_type,
              urls: response.urls ?? [],
              seed: (response.seed ?? params.seed ?? -1) as number,
              params,
              durationMs: Date.now() - startTime,
              timestamp: Date.now(),
            };

            set((state) => {
              state.status = 'completed';
              state.result = result;
            });

            // Add to history
            get().addToHistory(result, pipeline, params);
          }
        } catch (error) {
          if (abortController?.signal.aborted) {
            set((state) => {
              state.status = 'idle';
              state.progress = null;
            });
          } else {
            set((state) => {
              state.status = 'error';
              state.error = error instanceof Error ? error.message : 'Generation failed';
              state.progress = null;
            });
          }
        } finally {
          abortController = null;
        }
      },

      /**
       * Cancel an in-progress generation
       */
      cancelGeneration: () => {
        abortController?.abort();
        set((state) => {
          state.status = 'idle';
          state.progress = null;
        });
      },

      clearResult: () => {
        set((state) => {
          state.result = null;
          state.status = 'idle';
        });
      },

      clearHistory: () => {
        set((state) => {
          state.history = [];
        });
      },

      removeHistoryItem: (id) => {
        set((state) => {
          state.history = state.history.filter((item) => item.id !== id);
        });
      },

      /**
       * Load parameters from a history item into the form and display the result
       */
      loadHistoryParams: (item) => {
        const formStore = useFormStore.getState();
        const appStore = useAppStore.getState();

        // Switch to the correct tab and pipeline
        const pipeline = appStore.pipelines[item.pipelineId];
        if (pipeline) {
          appStore.setActiveTab(pipeline.category as 'image' | 'video');
          appStore.selectPipeline(item.pipelineId);
        }

        // Apply the parameters
        formStore.setValues(item.pipelineId, item.params);

        // Display the historical result if it has valid URLs
        if (item.result && item.result.urls.length > 0 && item.result.urls[0]) {
          set((state) => {
            state.result = item.result;
            state.status = 'completed';
            state.error = null;
            state.progress = null;
          });
        }
      },

      // Internal: Add result to history
      addToHistory: (
        result: GenerationResult,
        pipeline: { name: string; color: string },
        params: FormValues
      ) => {
        // Don't store base64 data URLs in history - they're too large for localStorage
        // Only persist file paths (like /outputs/...) which can be fetched later
        const url = result.urls[0] ?? '';
        const isBase64 = url.startsWith('data:');
        const thumbnailUrl = isBase64 ? '' : url;

        // Create a lightweight result without base64 URLs for storage
        const storableResult: GenerationResult = {
          ...result,
          urls: result.urls.map((u) => (u.startsWith('data:') ? '' : u)),
        };

        const historyItem: HistoryItem = {
          id: result.id,
          pipelineId: result.pipelineId,
          pipelineName: pipeline.name,
          pipelineColor: pipeline.color,
          thumbnailUrl,
          prompt: (params.prompt as string) ?? '',
          shortPrompt: truncatePrompt((params.prompt as string) ?? ''),
          keyParams: extractKeyParams(params),
          timestamp: result.timestamp,
          params,
          result: storableResult,
        };

        set((state) => {
          // Add to front, limit size
          state.history = [historyItem, ...state.history].slice(0, MAX_HISTORY_ITEMS);
        });
      },
    })),
    {
      name: 'llm-dit-history',
      partialize: (state) => ({
        history: state.history,
      }),
    }
  )
);

