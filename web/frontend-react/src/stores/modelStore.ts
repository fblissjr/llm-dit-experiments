/**
 * Model Store
 *
 * Manages model loading state, VRAM usage, and component status.
 */

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import type {
  VRAMStatus,
  PipelineModelStatus,
  VRAMEstimate,
  LoadStatus,
} from '@/types';

interface ModelState {
  // VRAM status
  vram: VRAMStatus | null;

  // Per-pipeline model status
  modelStatus: Record<string, PipelineModelStatus>;

  // Loading states
  isLoadingVRAM: boolean;
  isLoadingModel: boolean;
  loadingPipelineId: string | null;

  // Error state
  error: string | null;

  // Actions
  fetchVRAMStatus: () => Promise<void>;
  fetchModelStatus: (pipelineId: string) => Promise<void>;
  loadModel: (pipelineId: string) => Promise<void>;
  unloadModel: (pipelineId: string) => Promise<void>;
  unloadAll: () => Promise<void>;
  estimateVRAM: (pipelineId: string) => Promise<VRAMEstimate | null>;

  // Derived
  getModelStatus: (pipelineId: string) => PipelineModelStatus | undefined;
  isModelLoaded: (pipelineId: string) => boolean;
}

export const useModelStore = create<ModelState>()(
  immer((set, get) => ({
    // Initial state
    vram: null,
    modelStatus: {},
    isLoadingVRAM: false,
    isLoadingModel: false,
    loadingPipelineId: null,
    error: null,

    // Actions
    fetchVRAMStatus: async () => {
      set((state) => {
        state.isLoadingVRAM = true;
      });

      try {
        const response = await fetch('/api/vram/status');
        if (!response.ok) {
          throw new Error('Failed to fetch VRAM status');
        }

        const data = await response.json();

        set((state) => {
          state.vram = {
            usedMB: data.used_mb ?? data.usedMB ?? 0,
            totalMB: data.total_mb ?? data.totalMB ?? 24576, // Default 24GB
            freeMB: data.free_mb ?? data.freeMB ?? 24576,
            utilizationPercent: data.utilization_percent ?? data.utilizationPercent ?? 0,
            breakdown: data.breakdown ?? [],
          };
          state.isLoadingVRAM = false;
        });
      } catch (error) {
        set((state) => {
          state.isLoadingVRAM = false;
          // Don't set error for VRAM fetch failures - not critical
        });
      }
    },

    fetchModelStatus: async (pipelineId: string) => {
      try {
        const response = await fetch(`/api/models/${pipelineId}/status`);
        if (!response.ok) {
          return; // Silently fail - model might not be loaded
        }

        const data = await response.json();

        set((state) => {
          state.modelStatus[pipelineId] = {
            pipelineId,
            status: data.status as LoadStatus,
            components: data.components ?? [],
            totalVramMB: data.total_vram_mb ?? 0,
            loadTimeMs: data.load_time_ms,
          };
        });
      } catch {
        // Silently fail
      }
    },

    loadModel: async (pipelineId: string) => {
      set((state) => {
        state.isLoadingModel = true;
        state.loadingPipelineId = pipelineId;
        state.error = null;
      });

      try {
        const response = await fetch(`/api/models/${pipelineId}/load`, {
          method: 'POST',
        });

        if (!response.ok) {
          const data = await response.json();
          throw new Error(data.error ?? 'Failed to load model');
        }

        // Refresh status after load
        await get().fetchModelStatus(pipelineId);
        await get().fetchVRAMStatus();

        set((state) => {
          state.isLoadingModel = false;
          state.loadingPipelineId = null;
        });
      } catch (error) {
        set((state) => {
          state.isLoadingModel = false;
          state.loadingPipelineId = null;
          state.error = error instanceof Error ? error.message : 'Failed to load model';
        });
      }
    },

    unloadModel: async (pipelineId: string) => {
      try {
        const response = await fetch(`/api/models/${pipelineId}/unload`, {
          method: 'POST',
        });

        if (!response.ok) {
          throw new Error('Failed to unload model');
        }

        set((state) => {
          delete state.modelStatus[pipelineId];
        });

        await get().fetchVRAMStatus();
      } catch (error) {
        set((state) => {
          state.error = error instanceof Error ? error.message : 'Failed to unload model';
        });
      }
    },

    unloadAll: async () => {
      try {
        const response = await fetch('/api/models/unload-all', {
          method: 'POST',
        });

        if (!response.ok) {
          throw new Error('Failed to unload models');
        }

        set((state) => {
          state.modelStatus = {};
        });

        await get().fetchVRAMStatus();
      } catch (error) {
        set((state) => {
          state.error = error instanceof Error ? error.message : 'Failed to unload models';
        });
      }
    },

    estimateVRAM: async (pipelineId: string) => {
      try {
        const response = await fetch(`/api/models/${pipelineId}/estimate`);
        if (!response.ok) {
          return null;
        }

        const data = await response.json();
        const vram = get().vram;

        return {
          requiredMB: data.required_mb ?? data.requiredMB ?? 0,
          currentFreeMB: vram?.freeMB ?? 0,
          wouldFit: (data.required_mb ?? 0) <= (vram?.freeMB ?? 0),
          suggestions: data.suggestions,
        };
      } catch {
        return null;
      }
    },

    // Derived
    getModelStatus: (pipelineId: string) => {
      return get().modelStatus[pipelineId];
    },

    isModelLoaded: (pipelineId: string) => {
      return get().modelStatus[pipelineId]?.status === 'loaded';
    },
  }))
);
