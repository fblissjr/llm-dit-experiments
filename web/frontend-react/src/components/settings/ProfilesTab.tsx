/**
 * ProfilesTab Component
 *
 * Config profile management:
 * - List available profiles from server
 * - Highlight current profile
 * - Switch profile (triggers server restart)
 */

import { useState, useCallback } from 'react';
import { useProfileStore } from '@/stores/profileStore';
import { useServerStore } from '@/stores/serverStore';

export function ProfilesTab() {
  const { profiles, currentProfile, isLoading, isRestarting, error, switchProfile, clearError } = useProfileStore();
  const { status: serverStatus } = useServerStore();

  const [confirmProfile, setConfirmProfile] = useState<string | null>(null);

  const handleSwitchProfile = useCallback(async (profileName: string) => {
    if (profileName === currentProfile) return;
    setConfirmProfile(profileName);
  }, [currentProfile]);

  const confirmSwitch = useCallback(async () => {
    if (!confirmProfile) return;

    const success = await switchProfile(confirmProfile);
    if (success) {
      setConfirmProfile(null);
    }
  }, [confirmProfile, switchProfile]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="flex items-center gap-3 text-gray-400">
          <span className="w-5 h-5 border-2 border-gray-400 border-t-transparent rounded-full animate-spin" />
          Loading profiles...
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Description */}
      <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
        <h3 className="text-sm font-medium text-gray-200 mb-2">About Config Profiles</h3>
        <p className="text-sm text-gray-400">
          Profiles define server-side settings like model quantization, VRAM limits, and default parameters.
          Switching profiles requires a server restart, which will take a few seconds.
        </p>
      </div>

      {/* Error display */}
      {error && (
        <div className="bg-red-900/20 border border-red-600/30 rounded-lg p-4 flex items-start gap-3">
          <span className="text-red-500">⚠️</span>
          <div className="flex-1">
            <p className="text-sm text-red-300">{error}</p>
            <button
              onClick={clearError}
              className="text-xs text-red-400 hover:text-red-300 mt-1"
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      {/* Server restarting indicator */}
      {(isRestarting || serverStatus === 'restarting') && (
        <div className="bg-yellow-900/20 border border-yellow-600/30 rounded-lg p-4 flex items-center gap-3">
          <span className="w-5 h-5 border-2 border-yellow-400 border-t-transparent rounded-full animate-spin" />
          <span className="text-yellow-300">Server is restarting... Please wait.</span>
        </div>
      )}

      {/* Profile list */}
      <div className="space-y-3">
        <h2 className="text-lg font-semibold text-gray-100">Available Profiles</h2>

        {profiles.length === 0 ? (
          <div className="bg-gray-800 rounded-lg p-6 text-center text-gray-400">
            No profiles found. Check your config.toml for profile definitions.
          </div>
        ) : (
          <div className="grid gap-3">
            {profiles.map((profile) => (
              <ProfileCard
                key={profile}
                name={profile}
                isCurrent={profile === currentProfile}
                isRestarting={isRestarting}
                onSelect={() => handleSwitchProfile(profile)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Confirmation dialog */}
      {confirmProfile && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-800 rounded-lg p-6 max-w-md w-full border border-gray-700 shadow-xl">
            <h3 className="text-lg font-semibold text-gray-100 mb-2">
              Switch to "{confirmProfile}" profile?
            </h3>
            <p className="text-sm text-gray-400 mb-4">
              This will restart the server. All loaded models will be unloaded,
              and generation will be interrupted if in progress.
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setConfirmProfile(null)}
                className="px-4 py-2 text-sm font-medium rounded-lg
                  bg-gray-700 text-gray-300 hover:bg-gray-600 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={confirmSwitch}
                disabled={isRestarting}
                className="px-4 py-2 text-sm font-medium rounded-lg
                  bg-blue-600 text-white hover:bg-blue-500 transition-colors
                  disabled:bg-gray-700 disabled:text-gray-500"
              >
                {isRestarting ? 'Restarting...' : 'Switch Profile'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Profile card component
interface ProfileCardProps {
  name: string;
  isCurrent: boolean;
  isRestarting: boolean;
  onSelect: () => void;
}

function ProfileCard({ name, isCurrent, isRestarting, onSelect }: ProfileCardProps) {
  // Map common profile names to descriptions and icons
  const profileMeta: Record<string, { icon: string; description: string }> = {
    default: { icon: '⚖️', description: 'Balanced settings for general use' },
    'low-vram': { icon: '💾', description: 'Optimized for GPUs with limited VRAM' },
    'low_vram': { icon: '💾', description: 'Optimized for GPUs with limited VRAM' },
    'high-quality': { icon: '✨', description: 'Maximum quality, higher VRAM usage' },
    'high_quality': { icon: '✨', description: 'Maximum quality, higher VRAM usage' },
    turbo: { icon: '⚡', description: 'Faster generation with fewer steps' },
    dev: { icon: '🔧', description: 'Development and testing configuration' },
  };

  const meta = profileMeta[name.toLowerCase()] ?? {
    icon: '📋',
    description: `Configuration profile: ${name}`,
  };

  return (
    <div
      className={`
        bg-gray-800 rounded-lg p-4 border transition-colors
        ${isCurrent ? 'border-blue-500/50 ring-1 ring-blue-500/20' : 'border-gray-700'}
      `}
    >
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <span className="text-2xl">{meta.icon}</span>
          <div>
            <div className="flex items-center gap-2">
              <h4 className="font-medium text-gray-100">{name}</h4>
              {isCurrent && (
                <span className="px-2 py-0.5 text-xs font-medium rounded bg-blue-500/20 text-blue-400">
                  Active
                </span>
              )}
            </div>
            <p className="text-sm text-gray-400">{meta.description}</p>
          </div>
        </div>

        {!isCurrent && (
          <button
            onClick={onSelect}
            disabled={isRestarting}
            className="px-3 py-1.5 text-sm font-medium rounded-lg
              bg-gray-700 text-gray-300 hover:bg-gray-600 transition-colors
              disabled:bg-gray-700/50 disabled:text-gray-500 disabled:cursor-not-allowed"
          >
            Load Profile
          </button>
        )}
      </div>
    </div>
  );
}
