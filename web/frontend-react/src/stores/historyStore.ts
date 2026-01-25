/**
 * History Store
 *
 * Manages generation history with localStorage persistence.
 * Supports comparison mode and cross-pipeline workflows.
 */

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { persist } from 'zustand/middleware';
import type {
  GenerationResult,
  HistoryItem,
  ParameterDiff,
} from '@/types';
import {
  formatRelativeTime,
  extractKeyParams,
  truncatePrompt,
} from '@/types/generation';
import { usePipelineStore } from './pipelineStore';

const MAX_HISTORY_ITEMS = 100;
const STORAGE_KEY = 'llm-dit-history';

interface HistoryState {
  // History items (most recent first)
  items: HistoryItem[];

  // Comparison mode
  selectedForCompare: string[];  // IDs of items selected for comparison
  isCompareMode: boolean;

  // Actions
  addItem: (result: GenerationResult) => void;
  removeItem: (id: string) => void;
  clearHistory: () => void;

  // Comparison
  toggleCompareMode: () => void;
  selectForCompare: (id: string) => void;
  deselectForCompare: (id: string) => void;
  clearCompareSelection: () => void;
  getComparisonDiff: () => ParameterDiff[];

  // Query
  getItem: (id: string) => HistoryItem | undefined;
  getItemsByPipeline: (pipelineId: string) => HistoryItem[];

  // Cross-pipeline workflow
  useAsInput: (id: string, targetPipelineId: string) => string | null;
}

export const useHistoryStore = create<HistoryState>()(
  persist(
    immer((set, get) => ({
      // Initial state
      items: [],
      selectedForCompare: [],
      isCompareMode: false,

      // Actions
      addItem: (result: GenerationResult) => {
        const pipelineStore = usePipelineStore.getState();
        const pipeline = pipelineStore.getPipeline(result.pipelineId);

        const historyItem: HistoryItem = {
          id: result.id,
          pipelineId: result.pipelineId,
          pipelineName: pipeline?.name ?? result.pipelineId,
          pipelineColor: pipeline?.color ?? 'blue',
          thumbnailUrl: result.thumbnailUrl ?? result.urls[0],
          prompt: (result.params.prompt as string) ?? '',
          shortPrompt: truncatePrompt((result.params.prompt as string) ?? ''),
          keyParams: extractKeyParams(result.params),
          timestamp: result.timestamp,
          relativeTime: formatRelativeTime(result.timestamp),
          params: result.params,
          result,
        };

        set((state) => {
          // Add to front
          state.items.unshift(historyItem);

          // Trim to max size
          if (state.items.length > MAX_HISTORY_ITEMS) {
            state.items = state.items.slice(0, MAX_HISTORY_ITEMS);
          }
        });
      },

      removeItem: (id: string) => {
        set((state) => {
          state.items = state.items.filter((item) => item.id !== id);
          state.selectedForCompare = state.selectedForCompare.filter((sid) => sid !== id);
        });
      },

      clearHistory: () => {
        set((state) => {
          state.items = [];
          state.selectedForCompare = [];
          state.isCompareMode = false;
        });
      },

      // Comparison
      toggleCompareMode: () => {
        set((state) => {
          state.isCompareMode = !state.isCompareMode;
          if (!state.isCompareMode) {
            state.selectedForCompare = [];
          }
        });
      },

      selectForCompare: (id: string) => {
        set((state) => {
          if (state.selectedForCompare.length < 2 && !state.selectedForCompare.includes(id)) {
            state.selectedForCompare.push(id);
          }
        });
      },

      deselectForCompare: (id: string) => {
        set((state) => {
          state.selectedForCompare = state.selectedForCompare.filter((sid) => sid !== id);
        });
      },

      clearCompareSelection: () => {
        set((state) => {
          state.selectedForCompare = [];
        });
      },

      getComparisonDiff: () => {
        const { items, selectedForCompare } = get();
        if (selectedForCompare.length !== 2) return [];

        const itemA = items.find((i) => i.id === selectedForCompare[0]);
        const itemB = items.find((i) => i.id === selectedForCompare[1]);

        if (!itemA || !itemB) return [];

        const diffs: ParameterDiff[] = [];
        const allKeys = new Set([
          ...Object.keys(itemA.params),
          ...Object.keys(itemB.params),
        ]);

        allKeys.forEach((key) => {
          const valueA = itemA.params[key];
          const valueB = itemB.params[key];

          if (JSON.stringify(valueA) !== JSON.stringify(valueB)) {
            diffs.push({
              key,
              label: key.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
              valueA,
              valueB,
            });
          }
        });

        return diffs;
      },

      // Query
      getItem: (id: string) => {
        return get().items.find((item) => item.id === id);
      },

      getItemsByPipeline: (pipelineId: string) => {
        return get().items.filter((item) => item.pipelineId === pipelineId);
      },

      // Cross-pipeline workflow
      useAsInput: (id: string, targetPipelineId: string) => {
        const item = get().getItem(id);
        if (!item) return null;

        // Return the URL for use as input image in another pipeline
        // The calling code should handle setting this in the target pipeline's form
        const pipelineStore = usePipelineStore.getState();
        const targetPipeline = pipelineStore.getPipeline(targetPipelineId);

        if (!targetPipeline?.supports_img2img && !targetPipeline?.supports_reference_images) {
          return null;
        }

        return item.thumbnailUrl;
      },
    })),
    {
      name: STORAGE_KEY,
      partialize: (state) => ({
        items: state.items,
      }),
      // Update relative times on rehydration
      onRehydrateStorage: () => (state) => {
        if (state) {
          state.items = state.items.map((item) => ({
            ...item,
            relativeTime: formatRelativeTime(item.timestamp),
          }));
        }
      },
    }
  )
);

// Update relative times periodically
if (typeof window !== 'undefined') {
  setInterval(() => {
    const state = useHistoryStore.getState();
    useHistoryStore.setState({
      items: state.items.map((item) => ({
        ...item,
        relativeTime: formatRelativeTime(item.timestamp),
      })),
    });
  }, 60000); // Every minute
}
