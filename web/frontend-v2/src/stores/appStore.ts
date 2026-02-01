/**
 * App Store - Configuration, Navigation, and UI State
 *
 * Manages:
 * - Pipeline schemas from /api/pipelines
 * - Server defaults per pipeline
 * - Presets per pipeline
 * - Tab/pipeline navigation
 * - VRAM status
 * - UI state (mobile, history panel)
 */

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import type {
  PipelineSchema,
  GenerationPreset,
  VRAMStatus,
  FormValues,
} from '@/api/types';
import {
  fetchPipelines,
  fetchPresets,
  fetchVRAMStatus,
  fetchPipelineDefaults,
} from '@/api/client';

interface AppState {
  // Pipeline data
  pipelines: Record<string, PipelineSchema>;
  serverDefaults: Record<string, FormValues>;
  presets: Record<string, GenerationPreset[]>;

  // Navigation
  activeTab: 'image' | 'video';
  selectedPipelineId: string | null;

  // UI state
  isMobile: boolean;
  isHistoryOpen: boolean;

  // VRAM
  vram: VRAMStatus | null;

  // Loading state
  isLoading: boolean;
  error: string | null;

  // Actions
  initialize: () => Promise<void>;
  setActiveTab: (tab: 'image' | 'video') => void;
  selectPipeline: (pipelineId: string) => void;
  setIsMobile: (isMobile: boolean) => void;
  toggleHistory: () => void;
  refreshVRAM: () => Promise<void>;
  loadPresets: (pipelineId: string) => Promise<void>;

  // Computed
  getPipelinesForTab: (tab: 'image' | 'video') => PipelineSchema[];
  getSelectedPipeline: () => PipelineSchema | null;
  getPipelineColor: (pipelineId: string) => string;
}

// Map pipeline color names to CSS values
const colorMap: Record<string, string> = {
  blue: '#3b82f6',
  purple: '#a855f7',
  orange: '#f97316',
  teal: '#14b8a6',
  green: '#22c55e',
  pink: '#ec4899',
};

export const useAppStore = create<AppState>()(
  immer((set, get) => ({
    // Initial state
    pipelines: {},
    serverDefaults: {},
    presets: {},
    activeTab: 'image',
    selectedPipelineId: null,
    isMobile: false,
    isHistoryOpen: false,
    vram: null,
    isLoading: true,
    error: null,

    /**
     * Initialize app - fetch pipelines, defaults, and VRAM
     */
    initialize: async () => {
      set((state) => {
        state.isLoading = true;
        state.error = null;
      });

      try {
        // Fetch pipeline schemas
        const response = await fetchPipelines();
        const pipelines = response.pipelines;

        // Fetch defaults for each pipeline
        const defaultsEntries = await Promise.all(
          Object.keys(pipelines).map(async (id) => {
            try {
              const defaults = await fetchPipelineDefaults(id);
              return [id, defaults] as const;
            } catch {
              // Fall back to schema defaults
              const schema = pipelines[id];
              const schemaDefaults: FormValues = {};
              for (const param of schema.params) {
                if (param.default !== undefined) {
                  schemaDefaults[param.id] = param.default;
                }
              }
              return [id, schemaDefaults] as const;
            }
          })
        );
        const serverDefaults = Object.fromEntries(defaultsEntries);

        // Determine first pipeline to select
        const imagePipelines = Object.values(pipelines).filter(
          (p) => p.category === 'image'
        );
        const firstPipeline = imagePipelines[0]?.id ?? Object.keys(pipelines)[0];

        set((state) => {
          state.pipelines = pipelines;
          state.serverDefaults = serverDefaults;
          state.selectedPipelineId = firstPipeline ?? null;
          state.isLoading = false;
        });

        // Fetch VRAM status (non-blocking)
        get().refreshVRAM();

        // Load presets for first pipeline
        if (firstPipeline) {
          get().loadPresets(firstPipeline);
        }
      } catch (error) {
        set((state) => {
          state.isLoading = false;
          state.error = error instanceof Error ? error.message : 'Failed to load pipelines';
        });
      }
    },

    /**
     * Switch between Image and Video tabs
     */
    setActiveTab: (tab) => {
      set((state) => {
        state.activeTab = tab;

        // Select first pipeline in new tab
        const pipelinesInTab = get().getPipelinesForTab(tab);
        if (pipelinesInTab.length > 0) {
          state.selectedPipelineId = pipelinesInTab[0].id;
        }
      });

      // Load presets for new pipeline
      const pipelineId = get().selectedPipelineId;
      if (pipelineId) {
        get().loadPresets(pipelineId);
      }
    },

    /**
     * Select a specific pipeline within current tab
     */
    selectPipeline: (pipelineId) => {
      set((state) => {
        state.selectedPipelineId = pipelineId;
      });
      get().loadPresets(pipelineId);
    },

    setIsMobile: (isMobile) => {
      set((state) => {
        state.isMobile = isMobile;
      });
    },

    toggleHistory: () => {
      set((state) => {
        state.isHistoryOpen = !state.isHistoryOpen;
      });
    },

    /**
     * Refresh VRAM status from server
     */
    refreshVRAM: async () => {
      try {
        const vram = await fetchVRAMStatus();
        set((state) => {
          state.vram = vram;
        });
      } catch {
        // Silently fail - VRAM status is non-critical
      }
    },

    /**
     * Load presets for a pipeline and apply default preset if configured
     */
    loadPresets: async (pipelineId) => {
      // Skip if already loaded
      if (get().presets[pipelineId]) return;

      try {
        const { presets, defaultPreset } = await fetchPresets(pipelineId);
        set((state) => {
          state.presets[pipelineId] = presets;
        });

        // Apply default preset if one is configured
        // Use setTimeout to break circular dependency (formStore imports appStore)
        if (defaultPreset && presets.length > 0) {
          const preset = presets.find((p) => p.name === defaultPreset);
          console.log('[appStore] Default preset:', defaultPreset, 'found:', !!preset, 'params:', preset?.params);
          if (preset && preset.params) {
            setTimeout(async () => {
              const { useFormStore } = await import('./formStore');
              const formStore = useFormStore.getState();

              console.log('[appStore] Applying default preset:', defaultPreset);
              // Apply preset params (includes negative_prompt, steps, guidance_scale, etc.)
              formStore.applyPreset(pipelineId, preset.params);
              formStore.setValue(pipelineId, 'preset', defaultPreset);
            }, 0);
          }
        } else {
          console.log('[appStore] No default preset or no presets. defaultPreset:', defaultPreset, 'presets.length:', presets.length);
        }
      } catch {
        // Silently fail - presets are optional
        set((state) => {
          state.presets[pipelineId] = [];
        });
      }
    },

    // Computed
    getPipelinesForTab: (tab) => {
      const category = tab === 'image' ? 'image' : 'video';
      return Object.values(get().pipelines).filter(
        (p) => p.category === category
      );
    },

    getSelectedPipeline: () => {
      const { pipelines, selectedPipelineId } = get();
      return selectedPipelineId ? pipelines[selectedPipelineId] ?? null : null;
    },

    getPipelineColor: (pipelineId) => {
      const pipeline = get().pipelines[pipelineId];
      if (!pipeline) return colorMap.blue;
      return colorMap[pipeline.color] ?? colorMap.blue;
    },
  }))
);
