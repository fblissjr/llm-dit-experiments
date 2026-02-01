/**
 * Pipeline Store
 *
 * Manages pipeline schemas fetched from the API and current pipeline selection.
 * Also manages generation presets for each pipeline.
 */

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import type { PipelineSchema, GenerationPreset } from '@/types';

interface PipelineState {
  // Data
  pipelines: Record<string, PipelineSchema>;
  selectedPipelineId: string | null;
  serverDefaults: Record<string, unknown>;  // Defaults from server (includes zimage_variant)

  // Presets: keyed by pipeline ID
  presets: Record<string, GenerationPreset[]>;
  defaultPresets: Record<string, string>;  // Pipeline ID -> default preset name
  presetsLoading: Record<string, boolean>;

  // Loading state
  isLoading: boolean;
  error: string | null;

  // Derived
  selectedPipeline: PipelineSchema | null;
  pipelinesByCategory: Record<string, PipelineSchema[]>;

  // Actions
  fetchPipelines: () => Promise<void>;
  fetchPresets: (pipelineId: string) => Promise<void>;
  selectPipeline: (id: string) => void;
  getPipeline: (id: string) => PipelineSchema | undefined;
  getPreset: (pipelineId: string, presetName: string) => GenerationPreset | undefined;
  getPresetsForPipeline: (pipelineId: string) => GenerationPreset[];
}

export const usePipelineStore = create<PipelineState>()(
  immer((set, get) => ({
    // Initial state
    pipelines: {},
    selectedPipelineId: null,
    serverDefaults: {},
    presets: {},
    defaultPresets: {},
    presetsLoading: {},
    isLoading: false,
    error: null,

    // Note: selectedPipeline and pipelinesByCategory are now exported selectors
    // (see below) instead of ES5 getters, which don't work with getState()
    selectedPipeline: null,
    pipelinesByCategory: {},

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
        let pipelines: Record<string, PipelineSchema> = {};

        // API returns { pipelines: {id: schema, ...}, defaults: {...}, loaded_pipeline: string|null }
        if (data.pipelines && typeof data.pipelines === 'object') {
          // Server returns pipelines as a dict: { pipeline_id: schema }
          if (Array.isArray(data.pipelines)) {
            // Array format (legacy): [{id: "zimage", ...}, ...]
            data.pipelines.forEach((pipeline: PipelineSchema) => {
              pipelines[pipeline.id] = pipeline;
            });
          } else {
            // Dict format (current): { "zimage": {...}, "ltx2": {...} }
            pipelines = data.pipelines;
          }
        } else if (Array.isArray(data)) {
          // Direct array format
          data.forEach((pipeline: PipelineSchema) => {
            pipelines[pipeline.id] = pipeline;
          });
        }

        set((state) => {
          state.pipelines = pipelines;
          state.serverDefaults = data.defaults || {};
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

    /**
     * Fetch presets for a specific pipeline.
     * Results are cached - subsequent calls return cached data.
     */
    fetchPresets: async (pipelineId: string) => {
      // Skip if already loading or loaded
      const state = get();
      if (state.presetsLoading[pipelineId] || state.presets[pipelineId]) {
        return;
      }

      set((state) => {
        state.presetsLoading[pipelineId] = true;
      });

      try {
        const response = await fetch(`/api/presets/${pipelineId}`);
        if (!response.ok) {
          throw new Error(`Failed to fetch presets: ${response.statusText}`);
        }

        const data = await response.json();
        // API returns { presets: [...], default_preset: string }
        const presets: GenerationPreset[] = data.presets || [];
        const defaultPreset: string = data.default_preset || '';

        set((state) => {
          state.presets[pipelineId] = presets;
          state.defaultPresets[pipelineId] = defaultPreset;
          state.presetsLoading[pipelineId] = false;
        });
      } catch (error) {
        console.error(`Failed to fetch presets for ${pipelineId}:`, error);
        set((state) => {
          state.presets[pipelineId] = [];
          state.presetsLoading[pipelineId] = false;
        });
      }
    },

    /**
     * Get a specific preset by name for a pipeline.
     */
    getPreset: (pipelineId: string, presetName: string) => {
      const presets = get().presets[pipelineId];
      if (!presets) return undefined;
      return presets.find((p) => p.name === presetName);
    },

    /**
     * Get all presets for a pipeline.
     */
    getPresetsForPipeline: (pipelineId: string) => {
      return get().presets[pipelineId] || [];
    },
  }))
);

/**
 * Selector: Get the currently selected pipeline.
 * Use this instead of the old getter pattern which doesn't work with getState().
 *
 * Usage in component: const pipeline = usePipelineStore(selectSelectedPipeline);
 * Usage outside:      const pipeline = selectSelectedPipeline(usePipelineStore.getState());
 */
export const selectSelectedPipeline = (state: PipelineState): PipelineSchema | null => {
  return state.selectedPipelineId ? state.pipelines[state.selectedPipelineId] ?? null : null;
};

/**
 * Selector: Get pipelines grouped by category.
 * Use this instead of the old getter pattern which doesn't work with getState().
 *
 * Usage in component: const byCategory = usePipelineStore(selectPipelinesByCategory);
 * Usage outside:      const byCategory = selectPipelinesByCategory(usePipelineStore.getState());
 */
export const selectPipelinesByCategory = (state: PipelineState): Record<string, PipelineSchema[]> => {
  const byCategory: Record<string, PipelineSchema[]> = {};

  Object.values(state.pipelines).forEach((pipeline) => {
    if (!byCategory[pipeline.category]) {
      byCategory[pipeline.category] = [];
    }
    byCategory[pipeline.category].push(pipeline);
  });

  return byCategory;
};
