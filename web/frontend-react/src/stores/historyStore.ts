/**
 * History Store
 *
 * Manages generation history with localStorage persistence.
 * Supports comparison mode and cross-pipeline workflows.
 */

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { persist, createJSONStorage } from 'zustand/middleware';
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

/**
 * Custom storage that handles localStorage quota errors.
 * When quota is exceeded, it trims oldest items and retries.
 */
const quotaSafeStorage = {
  getItem: (name: string): string | null => {
    try {
      return localStorage.getItem(name);
    } catch {
      console.warn('[HistoryStore] Failed to read from localStorage');
      return null;
    }
  },
  setItem: (name: string, value: string): void => {
    const saveWithRetry = (val: string, retries = 3): void => {
      try {
        localStorage.setItem(name, val);
      } catch (error) {
        // Check if it's a quota error
        if (
          error instanceof DOMException &&
          (error.name === 'QuotaExceededError' ||
            error.name === 'NS_ERROR_DOM_QUOTA_REACHED')
        ) {
          if (retries > 0) {
            // Parse the current value, trim oldest items, and retry
            try {
              const parsed = JSON.parse(val);
              if (parsed.state?.items && Array.isArray(parsed.state.items)) {
                // Remove oldest 20% of items
                const trimCount = Math.max(1, Math.floor(parsed.state.items.length * 0.2));
                parsed.state.items = parsed.state.items.slice(0, -trimCount);
                console.warn(
                  `[HistoryStore] localStorage quota exceeded, trimmed ${trimCount} oldest items`
                );
                saveWithRetry(JSON.stringify(parsed), retries - 1);
                return;
              }
            } catch {
              // If parsing fails, just log and give up
            }
          }
          console.error(
            '[HistoryStore] localStorage quota exceeded, could not save history'
          );
        } else {
          console.error('[HistoryStore] Failed to save to localStorage:', error);
        }
      }
    };
    saveWithRetry(value);
  },
  removeItem: (name: string): void => {
    try {
      localStorage.removeItem(name);
    } catch {
      console.warn('[HistoryStore] Failed to remove from localStorage');
    }
  },
};

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
      storage: createJSONStorage(() => quotaSafeStorage),
      partialize: (state) => ({
        items: state.items,
      }) as HistoryState,
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

/**
 * Update all history items' relative times.
 * Call this from a component's useEffect with setInterval.
 * Example:
 *   useEffect(() => {
 *     const interval = setInterval(updateHistoryRelativeTimes, 60000);
 *     return () => clearInterval(interval);
 *   }, []);
 */
export function updateHistoryRelativeTimes() {
  const state = useHistoryStore.getState();
  if (state.items.length === 0) return;

  useHistoryStore.setState({
    items: state.items.map((item) => ({
      ...item,
      relativeTime: formatRelativeTime(item.timestamp),
    })),
  });
}
