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
import { persist, createJSONStorage } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import { idbStorage } from '@/utils/idbStorage';
import type {
  PipelineSchema,
  GenerationPreset,
  VRAMStatus,
  FormValues,
  ModelStatusResponse,
  GenerationContext,
} from '@/api/types';
import { PIPELINE_COLOR_MAP } from '@/constants/colors';
import { useFormStore } from './formStore';
import {
  fetchPipelines,
  fetchPresets,
  fetchVRAMStatus,
  fetchPipelineDefaults,
  fetchModelStatus,
  loadModel,
  unloadModel,
  fetchGenerationContext,
  restartServer as restartServerApi,
  clearCache as clearCacheApi,
} from '@/api/client';
import { logger } from '@/utils/logger';

const log = logger('Model');

interface AppState {
  // Pipeline data
  pipelines: Record<string, PipelineSchema>;
  serverDefaults: Record<string, FormValues>;
  presets: Record<string, GenerationPreset[]>;

  // Model state
  modelStatus: Record<string, ModelStatusResponse>;

  // Navigation
  activeTab: 'image' | 'video';
  selectedPipelineId: string | null;

  // UI state
  isMobile: boolean;
  isHistoryOpen: boolean;
  isLeftNavOpen: boolean;
  isSettingsOpen: boolean;

  // Generation context (composite status)
  generationContext: GenerationContext | null;

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
  toggleLeftNav: () => void;
  toggleSettings: () => void;
  refreshVRAM: () => Promise<void>;
  refreshContext: () => Promise<void>;
  restartServer: () => Promise<void>;
  clearCache: () => Promise<{ freedGb: number }>;
  loadPresets: (pipelineId: string) => Promise<void>;

  // Model management
  refreshModelStatus: (pipelineId: string) => Promise<void>;
  refreshAllModelStatus: () => Promise<void>;
  loadPipelineModel: (pipelineId: string) => Promise<void>;
  unloadPipelineModel: (pipelineId: string) => Promise<void>;

  // Computed
  getPipelinesForTab: (tab: 'image' | 'video') => PipelineSchema[];
  getSelectedPipeline: () => PipelineSchema | null;
  getPipelineColor: (pipelineId: string) => string;
}

const colorMap = PIPELINE_COLOR_MAP;

export const useAppStore = create<AppState>()(
  persist(
    immer((set, get) => ({
    // Initial state
    pipelines: {},
    serverDefaults: {},
    presets: {},
    modelStatus: {},
    activeTab: 'image' as const,
    selectedPipelineId: null,
    isMobile: false,
    isHistoryOpen: false,
    isLeftNavOpen: true,
    isSettingsOpen: false,
    generationContext: null,
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

        // Pipeline selection priority:
        // 1. Server-loaded pipeline (authoritative -- server knows what's in VRAM)
        // 2. Persisted selection from previous session (if pipeline ID still valid)
        // 3. Fall back to first image pipeline
        const loadedId = response.loaded_pipeline;
        const persistedId = get().selectedPipelineId;
        let firstPipeline: string | undefined;
        if (loadedId && pipelines[loadedId]) {
          firstPipeline = loadedId;
        } else if (persistedId && pipelines[persistedId]) {
          firstPipeline = persistedId;
        } else {
          const imagePipelines = Object.values(pipelines).filter(
            (p) => p.category === 'image'
          );
          firstPipeline = imagePipelines[0]?.id ?? Object.keys(pipelines)[0];
        }

        // Set the active tab to match the selected pipeline's category
        const selectedCategory = firstPipeline
          ? pipelines[firstPipeline]?.category ?? 'image'
          : 'image';

        set((state) => {
          state.pipelines = pipelines;
          state.serverDefaults = serverDefaults;
          state.selectedPipelineId = firstPipeline ?? null;
          state.activeTab = selectedCategory as 'image' | 'video';
          state.isLoading = false;
        });

        // Fetch VRAM + model status (non-blocking, run in parallel)
        get().refreshVRAM();
        get().refreshAllModelStatus();

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

    toggleLeftNav: () => {
      set((state) => {
        state.isLeftNavOpen = !state.isLeftNavOpen;
      });
    },

    toggleSettings: () => {
      set((state) => {
        state.isSettingsOpen = !state.isSettingsOpen;
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
     * Refresh generation context (composite status for status bar)
     */
    refreshContext: async () => {
      try {
        const ctx = await fetchGenerationContext();
        set((state) => {
          state.generationContext = ctx;
        });
      } catch {
        // Silently fail - context is non-critical
      }
    },

    /**
     * Restart the server (with health polling for recovery)
     */
    restartServer: async () => {
      try {
        await restartServerApi('user_request');
      } catch {
        // Expected -- server goes down during restart
      }
    },

    /**
     * Clear CUDA cache and refresh context
     */
    clearCache: async () => {
      try {
        const result = await clearCacheApi();
        // Refresh context to pick up new VRAM numbers
        get().refreshContext();
        return { freedGb: result.freedGb ?? 0 };
      } catch {
        return { freedGb: 0 };
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
        if (defaultPreset && presets.length > 0) {
          const preset = presets.find((p) => p.name === defaultPreset);
          if (preset && preset.params) {
            useFormStore.getState().applyPreset(pipelineId, defaultPreset, preset.params);
          }
        }
      } catch {
        // Silently fail - presets are optional
        set((state) => {
          state.presets[pipelineId] = [];
        });
      }
    },

    /**
     * Model Management
     */

    /**
     * Refresh model status for a single pipeline
     */
    refreshModelStatus: async (pipelineId) => {
      try {
        const status = await fetchModelStatus(pipelineId);
        set((state) => {
          state.modelStatus[pipelineId] = status;
        });
      } catch {
        // If API fails, mark as unloaded
        set((state) => {
          state.modelStatus[pipelineId] = { status: 'unloaded' };
        });
      }
    },

    /**
     * Refresh model status for all pipelines
     */
    refreshAllModelStatus: async () => {
      const pipelineIds = Object.keys(get().pipelines);
      await Promise.all(
        pipelineIds.map((id) => get().refreshModelStatus(id))
      );
    },

    /**
     * Load a pipeline model
     */
    loadPipelineModel: async (pipelineId) => {
      log.info(`Loading ${pipelineId}...`);
      // Set to loading state
      set((state) => {
        state.modelStatus[pipelineId] = { status: 'loading' };
      });

      try {
        const result = await loadModel(pipelineId);
        set((state) => {
          state.modelStatus[pipelineId] = result;
        });
        log.info(`${pipelineId} status:`, result.status);

        // Refresh VRAM and context after loading
        get().refreshVRAM();
        get().refreshContext();
      } catch (error) {
        log.error(`${pipelineId} load failed:`, error);
        set((state) => {
          state.modelStatus[pipelineId] = {
            status: 'error',
            error: error instanceof Error ? error.message : 'Failed to load model',
          };
        });
      }
    },

    /**
     * Unload a pipeline model
     */
    unloadPipelineModel: async (pipelineId) => {
      log.info(`Unloading ${pipelineId}...`);
      // Set to loading state (unloading)
      set((state) => {
        state.modelStatus[pipelineId] = { status: 'loading' };
      });

      try {
        const result = await unloadModel(pipelineId);
        set((state) => {
          state.modelStatus[pipelineId] = result;
        });
        log.info(`${pipelineId} status:`, result.status);

        // Refresh VRAM and context after unloading
        get().refreshVRAM();
        get().refreshContext();
      } catch (error) {
        log.error(`${pipelineId} unload failed:`, error);
        set((state) => {
          state.modelStatus[pipelineId] = {
            status: 'error',
            error: error instanceof Error ? error.message : 'Failed to unload model',
          };
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
    })),
    {
      name: 'llm-dit-app',
      storage: createJSONStorage(() => idbStorage),
      partialize: (state) => ({
        selectedPipelineId: state.selectedPipelineId,
        activeTab: state.activeTab,
        isHistoryOpen: state.isHistoryOpen,
        isLeftNavOpen: state.isLeftNavOpen,
      }),
    }
  )
);
