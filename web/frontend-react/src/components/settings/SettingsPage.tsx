/**
 * SettingsPage Component
 *
 * Full-page settings view with tab navigation for:
 * - Models: Model management (load/unload/VRAM)
 * - Profiles: Config profile switching
 * - Defaults: Generation parameter defaults
 * - Server: Server status and restart controls
 */

import { useEffect } from 'react';
import { useUIStore, type SettingsTab } from '@/stores/uiStore';
import { useProfileStore } from '@/stores/profileStore';
import { useServerStore } from '@/stores/serverStore';
import { ModelsTab } from './ModelsTab';
import { ProfilesTab } from './ProfilesTab';
import { DefaultsTab } from './DefaultsTab';
import { ServerTab } from './ServerTab';

const TABS: { id: SettingsTab; label: string; icon: string }[] = [
  { id: 'models', label: 'Models', icon: '🧠' },
  { id: 'profiles', label: 'Profiles', icon: '📋' },
  { id: 'defaults', label: 'Defaults', icon: '⚙️' },
  { id: 'server', label: 'Server', icon: '🖥️' },
];

export function SettingsPage() {
  const { settingsTab, setSettingsTab, setView } = useUIStore();
  const { fetchProfiles } = useProfileStore();
  const { fetchStatus } = useServerStore();

  // Fetch data when settings page opens
  useEffect(() => {
    fetchProfiles();
    fetchStatus();
  }, [fetchProfiles, fetchStatus]);

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-4 py-3 sticky top-0 z-40">
        <div className="flex items-center justify-between max-w-5xl mx-auto">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setView('studio')}
              className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
            >
              <span className="text-lg">←</span>
              <span>Studio</span>
            </button>
            <div className="h-6 w-px bg-gray-700" />
            <h1 className="text-lg font-semibold text-gray-100">Settings</h1>
          </div>

          {/* Keyboard hint */}
          <div className="text-xs text-gray-500">
            Press <kbd className="px-1.5 py-0.5 bg-gray-700 rounded text-gray-400">,</kbd> to toggle
          </div>
        </div>
      </header>

      {/* Tab navigation */}
      <nav className="bg-gray-800/50 border-b border-gray-700">
        <div className="max-w-5xl mx-auto px-4">
          <div className="flex gap-1">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setSettingsTab(tab.id)}
                className={`
                  px-4 py-3 text-sm font-medium transition-colors
                  border-b-2 -mb-px flex items-center gap-2
                  ${
                    settingsTab === tab.id
                      ? 'border-blue-500 text-blue-400'
                      : 'border-transparent text-gray-400 hover:text-gray-200 hover:bg-gray-700/30'
                  }
                `}
              >
                <span>{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Tab content */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-5xl mx-auto p-6">
          {settingsTab === 'models' && <ModelsTab />}
          {settingsTab === 'profiles' && <ProfilesTab />}
          {settingsTab === 'defaults' && <DefaultsTab />}
          {settingsTab === 'server' && <ServerTab />}
        </div>
      </main>
    </div>
  );
}
