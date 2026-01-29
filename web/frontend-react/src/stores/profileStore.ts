/**
 * Profile Store
 *
 * Manages configuration profiles (e.g., "default", "low-vram", "high-quality").
 * Profiles control server-side settings; switching requires server restart.
 */

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';

interface ProfileState {
  // Available profiles
  profiles: string[];
  currentProfile: string | null;

  // Loading states
  isLoading: boolean;
  isRestarting: boolean;

  // Error state
  error: string | null;

  // Actions
  fetchProfiles: () => Promise<void>;
  switchProfile: (profileName: string) => Promise<boolean>;
  clearError: () => void;
}

export const useProfileStore = create<ProfileState>()(
  immer((set, get) => ({
    // Initial state
    profiles: [],
    currentProfile: null,
    isLoading: false,
    isRestarting: false,
    error: null,

    // Actions
    fetchProfiles: async () => {
      set((state) => {
        state.isLoading = true;
        state.error = null;
      });

      try {
        const response = await fetch('/api/config/profiles');
        if (!response.ok) {
          throw new Error('Failed to fetch profiles');
        }

        const data = await response.json();

        set((state) => {
          state.profiles = data.profiles ?? [];
          state.currentProfile = data.current ?? null;
          state.isLoading = false;
        });
      } catch (error) {
        set((state) => {
          state.isLoading = false;
          state.error = error instanceof Error ? error.message : 'Failed to fetch profiles';
        });
      }
    },

    switchProfile: async (profileName: string) => {
      const { currentProfile } = get();

      // Don't switch to current profile
      if (profileName === currentProfile) {
        return true;
      }

      set((state) => {
        state.isRestarting = true;
        state.error = null;
      });

      try {
        const response = await fetch('/api/server/restart', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            reason: `Switch profile to ${profileName}`,
            new_profile: profileName,
          }),
        });

        if (!response.ok) {
          const data = await response.json();
          throw new Error(data.error ?? 'Failed to switch profile');
        }

        // After restart, the server will use the new profile
        // The UI will need to poll health and refresh
        set((state) => {
          state.currentProfile = profileName;
          state.isRestarting = false;
        });

        return true;
      } catch (error) {
        set((state) => {
          state.isRestarting = false;
          state.error = error instanceof Error ? error.message : 'Failed to switch profile';
        });
        return false;
      }
    },

    clearError: () => {
      set((state) => {
        state.error = null;
      });
    },
  }))
);
