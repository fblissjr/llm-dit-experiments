/**
 * Pipeline Store
 *
 * Manages pipeline schemas fetched from the API and current pipeline selection.
 */

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import type { PipelineSchema } from '@/types';

interface PipelineState {
  // Data
  pipelines: Record<string, PipelineSchema>;
  selectedPipelineId: string | null;

  // Loading state
  isLoading: boolean;
  error: string | null;

  // Derived
  selectedPipeline: PipelineSchema | null;
  pipelinesByCategory: Record<string, PipelineSchema[]>;

  // Actions
  fetchPipelines: () => Promise<void>;
  selectPipeline: (id: string) => void;
  getPipeline: (id: string) => PipelineSchema | undefined;
}

export const usePipelineStore = create<PipelineState>()(
  immer((set, get) => ({
    // Initial state
    pipelines: {},
    selectedPipelineId: null,
    isLoading: false,
    error: null,

    // Computed getters
    get selectedPipeline() {
      const { pipelines, selectedPipelineId } = get();
      return selectedPipelineId ? pipelines[selectedPipelineId] ?? null : null;
    },

    get pipelinesByCategory() {
      const { pipelines } = get();
      const byCategory: Record<string, PipelineSchema[]> = {};

      Object.values(pipelines).forEach((pipeline) => {
        if (!byCategory[pipeline.category]) {
          byCategory[pipeline.category] = [];
        }
        byCategory[pipeline.category].push(pipeline);
      });

      return byCategory;
    },

    // Actions
    fetchPipelines: async () => {
      set((state) => {
        state.isLoading = true;
        state.error = null;
      });

      try {
        const response = await fetch('/api/pipelines');
        if (!response.ok) {
          throw new Error(`Failed to fetch pipelines: ${response.statusText}`);
        }

        const data = await response.json();
        const pipelines: Record<string, PipelineSchema> = {};

        // API returns array of pipeline schemas
        if (Array.isArray(data)) {
          data.forEach((pipeline: PipelineSchema) => {
            pipelines[pipeline.id] = pipeline;
          });
        } else if (data.pipelines) {
          // Or might be wrapped in { pipelines: [...] }
          data.pipelines.forEach((pipeline: PipelineSchema) => {
            pipelines[pipeline.id] = pipeline;
          });
        }

        set((state) => {
          state.pipelines = pipelines;
          state.isLoading = false;

          // Auto-select first pipeline if none selected
          if (!state.selectedPipelineId && Object.keys(pipelines).length > 0) {
            // Prefer Z-Image as default
            state.selectedPipelineId = pipelines['zimage']
              ? 'zimage'
              : Object.keys(pipelines)[0];
          }
        });
      } catch (error) {
        set((state) => {
          state.isLoading = false;
          state.error = error instanceof Error ? error.message : 'Failed to fetch pipelines';
        });
      }
    },

    selectPipeline: (id: string) => {
      set((state) => {
        if (state.pipelines[id]) {
          state.selectedPipelineId = id;
        }
      });
    },

    getPipeline: (id: string) => {
      return get().pipelines[id];
    },
  }))
);
