/**
 * ServerTab Component
 *
 * Server status and controls:
 * - Uptime, version, config file path
 * - Health indicator
 * - Restart button with confirmation
 * - Pending changes list
 */

import { useState, useCallback, useEffect } from 'react';
import { useServerStore } from '@/stores/serverStore';
import { useProfileStore } from '@/stores/profileStore';

export function ServerTab() {
  const {
    status,
    uptime,
    configFile,
    version,
    pendingChanges,
    isPolling,
    error,
    fetchStatus,
    restartServer,
    clearError,
  } = useServerStore();

  const { currentProfile } = useProfileStore();

  const [showRestartConfirm, setShowRestartConfirm] = useState(false);
  const [isRestarting, setIsRestarting] = useState(false);

  // Refresh status periodically
  useEffect(() => {
    const interval = setInterval(fetchStatus, 10000); // Every 10s
    return () => clearInterval(interval);
  }, [fetchStatus]);

  const handleRestart = useCallback(async () => {
    setIsRestarting(true);
    const success = await restartServer('Manual restart from settings');
    if (!success) {
      setIsRestarting(false);
    }
    setShowRestartConfirm(false);
  }, [restartServer]);

  // Reset restarting state when status changes to online
  useEffect(() => {
    if (status === 'online' && isRestarting) {
      setIsRestarting(false);
    }
  }, [status, isRestarting]);

  const formatUptime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    }
    return `${secs}s`;
  };

  return (
    <div className="space-y-6">
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

      {/* Restarting indicator */}
      {(status === 'restarting' || isPolling) && (
        <div className="bg-yellow-900/20 border border-yellow-600/30 rounded-lg p-4 flex items-center gap-3">
          <span className="w-5 h-5 border-2 border-yellow-400 border-t-transparent rounded-full animate-spin" />
          <span className="text-yellow-300">
            Server is restarting... Waiting for reconnection.
          </span>
        </div>
      )}

      {/* Status Overview */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h2 className="text-lg font-semibold text-gray-100 mb-4">Server Status</h2>

        <div className="grid gap-4 md:grid-cols-2">
          {/* Health */}
          <StatusItem
            label="Health"
            value={
              <span className="flex items-center gap-2">
                <span
                  className={`w-3 h-3 rounded-full ${
                    status === 'online' ? 'bg-green-500' :
                    status === 'restarting' ? 'bg-yellow-500 animate-pulse' :
                    'bg-red-500'
                  }`}
                />
                {status === 'online' ? 'Online' :
                 status === 'restarting' ? 'Restarting' :
                 status === 'offline' ? 'Offline' :
                 'Unknown'}
              </span>
            }
          />

          {/* Uptime */}
          <StatusItem
            label="Uptime"
            value={uptime !== null ? formatUptime(uptime) : '—'}
          />

          {/* Current profile */}
          <StatusItem
            label="Profile"
            value={currentProfile ?? '—'}
          />

          {/* Version */}
          <StatusItem
            label="Version"
            value={version ?? '—'}
          />
        </div>

        {/* Config file path */}
        {configFile && (
          <div className="mt-4 pt-4 border-t border-gray-700">
            <span className="text-sm text-gray-400">Config file: </span>
            <code className="text-sm text-gray-300 bg-gray-700/50 px-2 py-0.5 rounded">
              {configFile}
            </code>
          </div>
        )}
      </div>

      {/* Pending Changes */}
      {pendingChanges.length > 0 && (
        <div className="bg-yellow-900/20 border border-yellow-600/30 rounded-lg p-4">
          <h3 className="text-sm font-medium text-yellow-300 mb-2">
            Pending Changes (Require Restart)
          </h3>
          <ul className="text-sm text-yellow-400/80 space-y-1">
            {pendingChanges.map((change, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <span>•</span>
                <span>{change}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Actions */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h2 className="text-lg font-semibold text-gray-100 mb-4">Actions</h2>

        <div className="space-y-4">
          {/* Restart button */}
          <div className="flex items-center justify-between">
            <div>
              <h4 className="text-sm font-medium text-gray-200">Restart Server</h4>
              <p className="text-xs text-gray-400">
                Restart the backend server. All models will be unloaded.
              </p>
            </div>
            <button
              onClick={() => setShowRestartConfirm(true)}
              disabled={status === 'restarting' || isRestarting || isPolling}
              className="px-4 py-2 text-sm font-medium rounded-lg
                bg-orange-600/20 text-orange-400 hover:bg-orange-600/30
                border border-orange-600/30 transition-colors
                disabled:bg-gray-700/50 disabled:text-gray-500 disabled:border-gray-700 disabled:cursor-not-allowed"
            >
              {isRestarting ? 'Restarting...' : 'Restart'}
            </button>
          </div>

          {/* Refresh status */}
          <div className="flex items-center justify-between">
            <div>
              <h4 className="text-sm font-medium text-gray-200">Refresh Status</h4>
              <p className="text-xs text-gray-400">
                Manually refresh server status information.
              </p>
            </div>
            <button
              onClick={() => fetchStatus()}
              className="px-4 py-2 text-sm font-medium rounded-lg
                bg-gray-700 text-gray-300 hover:bg-gray-600 transition-colors"
            >
              Refresh
            </button>
          </div>
        </div>
      </div>

      {/* Restart confirmation dialog */}
      {showRestartConfirm && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-800 rounded-lg p-6 max-w-md w-full border border-gray-700 shadow-xl">
            <h3 className="text-lg font-semibold text-gray-100 mb-2">
              Restart Server?
            </h3>
            <p className="text-sm text-gray-400 mb-4">
              This will restart the backend server. All loaded models will be unloaded
              and any in-progress generation will be interrupted.
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setShowRestartConfirm(false)}
                className="px-4 py-2 text-sm font-medium rounded-lg
                  bg-gray-700 text-gray-300 hover:bg-gray-600 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleRestart}
                disabled={isRestarting}
                className="px-4 py-2 text-sm font-medium rounded-lg
                  bg-orange-600 text-white hover:bg-orange-500 transition-colors
                  disabled:bg-gray-700 disabled:text-gray-500"
              >
                {isRestarting ? 'Restarting...' : 'Restart Server'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Status item helper
function StatusItem({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <span className="text-sm text-gray-400">{label}</span>
      <div className="text-gray-100 font-medium">{value}</div>
    </div>
  );
}
