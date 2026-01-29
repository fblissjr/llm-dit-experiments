/**
 * Server Store
 *
 * Manages server status, health, and restart operations.
 */

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';

type ServerStatus = 'online' | 'offline' | 'restarting' | 'unknown';

interface ServerState {
  // Status
  status: ServerStatus;
  uptime: number | null; // seconds
  configFile: string | null;
  version: string | null;
  pendingChanges: string[];

  // Polling state
  isPolling: boolean;
  pollingIntervalId: ReturnType<typeof setInterval> | null;

  // Error state
  error: string | null;

  // Actions
  fetchStatus: () => Promise<void>;
  restartServer: (reason?: string) => Promise<boolean>;
  startHealthPolling: (onReconnect?: () => void) => void;
  stopHealthPolling: () => void;
  clearError: () => void;
}

// Health check with timeout
async function checkHealth(timeoutMs = 3000): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    const response = await fetch('/health', { signal: controller.signal });
    clearTimeout(timeoutId);

    return response.ok;
  } catch {
    return false;
  }
}

export const useServerStore = create<ServerState>()(
  immer((set, get) => ({
    // Initial state
    status: 'unknown',
    uptime: null,
    configFile: null,
    version: null,
    pendingChanges: [],
    isPolling: false,
    pollingIntervalId: null,
    error: null,

    // Actions
    fetchStatus: async () => {
      try {
        const response = await fetch('/api/server/status');
        if (!response.ok) {
          set((state) => {
            state.status = 'offline';
          });
          return;
        }

        const data = await response.json();

        set((state) => {
          state.status = 'online';
          state.uptime = data.uptime ?? null;
          state.configFile = data.config_file ?? null;
          state.version = data.version ?? null;
          state.pendingChanges = data.pending_changes ?? [];
        });
      } catch {
        set((state) => {
          state.status = 'offline';
        });
      }
    },

    restartServer: async (reason?: string) => {
      set((state) => {
        state.status = 'restarting';
        state.error = null;
      });

      try {
        const response = await fetch('/api/server/restart', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ reason: reason ?? 'Manual restart' }),
        });

        if (!response.ok) {
          const data = await response.json();
          throw new Error(data.error ?? 'Failed to restart server');
        }

        // Start polling for reconnection
        get().startHealthPolling(() => {
          // On reconnect, fetch full status
          get().fetchStatus();
        });

        return true;
      } catch (error) {
        set((state) => {
          state.status = 'offline';
          state.error = error instanceof Error ? error.message : 'Failed to restart server';
        });
        return false;
      }
    },

    startHealthPolling: (onReconnect?: () => void) => {
      const { isPolling, pollingIntervalId } = get();

      // Already polling
      if (isPolling) return;

      // Clear existing interval if any
      if (pollingIntervalId) {
        clearInterval(pollingIntervalId);
      }

      set((state) => {
        state.isPolling = true;
      });

      const intervalId = setInterval(async () => {
        const isHealthy = await checkHealth();

        if (isHealthy) {
          // Server is back
          get().stopHealthPolling();

          set((state) => {
            state.status = 'online';
          });

          onReconnect?.();
        }
      }, 2000); // Poll every 2 seconds

      set((state) => {
        state.pollingIntervalId = intervalId;
      });
    },

    stopHealthPolling: () => {
      const { pollingIntervalId } = get();

      if (pollingIntervalId) {
        clearInterval(pollingIntervalId);
      }

      set((state) => {
        state.isPolling = false;
        state.pollingIntervalId = null;
      });
    },

    clearError: () => {
      set((state) => {
        state.error = null;
      });
    },
  }))
);
