/**
 * UI Store
 *
 * Manages UI state: panels, responsive breakpoints, notifications.
 */

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';

type NotificationType = 'info' | 'success' | 'warning' | 'error';

interface Notification {
  id: string;
  type: NotificationType;
  message: string;
  duration?: number;  // Auto-dismiss after ms, 0 for persistent
}

interface UIState {
  // Responsive state
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;

  // Panel states
  isHistoryPanelOpen: boolean;
  isModelPanelOpen: boolean;
  isCommandPaletteOpen: boolean;

  // Collapsed sections (keyed by section ID)
  collapsedSections: Record<string, boolean>;

  // Notifications
  notifications: Notification[];

  // Actions
  setResponsiveState: (width: number) => void;
  toggleHistoryPanel: () => void;
  toggleModelPanel: () => void;
  toggleCommandPalette: () => void;
  toggleSection: (sectionId: string) => void;
  setSectionCollapsed: (sectionId: string, collapsed: boolean) => void;

  // Notifications
  addNotification: (notification: Omit<Notification, 'id'>) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
}

// Breakpoints matching Tailwind
const MOBILE_BREAKPOINT = 768;
const TABLET_BREAKPOINT = 1024;

export const useUIStore = create<UIState>()(
  immer((set) => ({
    // Initial state
    isMobile: typeof window !== 'undefined' ? window.innerWidth < MOBILE_BREAKPOINT : false,
    isTablet:
      typeof window !== 'undefined'
        ? window.innerWidth >= MOBILE_BREAKPOINT && window.innerWidth < TABLET_BREAKPOINT
        : false,
    isDesktop: typeof window !== 'undefined' ? window.innerWidth >= TABLET_BREAKPOINT : true,

    isHistoryPanelOpen: false,
    isModelPanelOpen: false,
    isCommandPaletteOpen: false,

    collapsedSections: {},

    notifications: [],

    // Actions
    setResponsiveState: (width: number) => {
      set((state) => {
        state.isMobile = width < MOBILE_BREAKPOINT;
        state.isTablet = width >= MOBILE_BREAKPOINT && width < TABLET_BREAKPOINT;
        state.isDesktop = width >= TABLET_BREAKPOINT;

        // Auto-close panels on mobile
        if (state.isMobile) {
          state.isHistoryPanelOpen = false;
        }
      });
    },

    toggleHistoryPanel: () => {
      set((state) => {
        state.isHistoryPanelOpen = !state.isHistoryPanelOpen;
      });
    },

    toggleModelPanel: () => {
      set((state) => {
        state.isModelPanelOpen = !state.isModelPanelOpen;
      });
    },

    toggleCommandPalette: () => {
      set((state) => {
        state.isCommandPaletteOpen = !state.isCommandPaletteOpen;
      });
    },

    toggleSection: (sectionId: string) => {
      set((state) => {
        state.collapsedSections[sectionId] = !state.collapsedSections[sectionId];
      });
    },

    setSectionCollapsed: (sectionId: string, collapsed: boolean) => {
      set((state) => {
        state.collapsedSections[sectionId] = collapsed;
      });
    },

    // Notifications
    addNotification: (notification) => {
      const id = `notif-${Date.now()}-${Math.random().toString(36).slice(2)}`;

      set((state) => {
        state.notifications.push({ ...notification, id });
      });

      // Auto-dismiss after duration
      const duration = notification.duration ?? 5000;
      if (duration > 0) {
        setTimeout(() => {
          set((state) => {
            state.notifications = state.notifications.filter((n) => n.id !== id);
          });
        }, duration);
      }
    },

    removeNotification: (id: string) => {
      set((state) => {
        state.notifications = state.notifications.filter((n) => n.id !== id);
      });
    },

    clearNotifications: () => {
      set((state) => {
        state.notifications = [];
      });
    },
  }))
);

/**
 * Initialize responsive state from window.
 * Call this once from App.tsx on mount.
 */
export function initResponsiveState() {
  if (typeof window !== 'undefined') {
    useUIStore.getState().setResponsiveState(window.innerWidth);
  }
}

/**
 * Handle resize listener setup (for use in useEffect).
 * Returns cleanup function.
 *
 * Usage in App.tsx:
 *   useEffect(() => {
 *     initResponsiveState();
 *     return setupResizeListener();
 *   }, []);
 */
export function setupResizeListener(): () => void {
  if (typeof window === 'undefined') return () => {};

  const handleResize = () => {
    useUIStore.getState().setResponsiveState(window.innerWidth);
  };

  window.addEventListener('resize', handleResize);

  // Return cleanup function
  return () => window.removeEventListener('resize', handleResize);
}
