/**
 * SettingsMenu Component
 *
 * Server management dropdown with restart, cache clear, and system info.
 * Desktop: positioned dropdown below the gear icon in LeftNav.
 * Mobile: rendered inside the StatusBar expanded view.
 */

import { useState, useEffect, useRef } from 'react';
import { cn, formatUptime } from '@/utils';
import { useAppStore } from '@/stores';
import { ConfirmDialog } from '@/components/common/ConfirmDialog';
import { RestartWarning } from '@/components/common/RestartWarning';

interface SettingsMenuProps {
  /** Whether the menu is rendered as a dropdown (desktop) or inline (mobile) */
  variant?: 'dropdown' | 'inline';
  onClose?: () => void;
}

export function SettingsMenu({ variant = 'dropdown', onClose }: SettingsMenuProps) {
  const ctx = useAppStore((s) => s.generationContext);
  const restartServer = useAppStore((s) => s.restartServer);
  const clearCache = useAppStore((s) => s.clearCache);

  const [showRestartConfirm, setShowRestartConfirm] = useState(false);
  const [cacheFeedback, setCacheFeedback] = useState<string | null>(null);
  const [isRestarting, setIsRestarting] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click (only for dropdown variant)
  useEffect(() => {
    if (variant !== 'dropdown' || !onClose) return;

    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        onClose();
      }
    };

    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };

    // Delay adding to avoid the click that opened the menu
    const timer = setTimeout(() => {
      document.addEventListener('mousedown', handleClickOutside);
      document.addEventListener('keydown', handleEscape);
    }, 50);

    return () => {
      clearTimeout(timer);
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [variant, onClose]);

  const handleRestart = async () => {
    setShowRestartConfirm(false);
    setIsRestarting(true);
    await restartServer();
    // Server is restarting -- the UI will show a health-polling state
  };

  const handleClearCache = async () => {
    setCacheFeedback(null);
    const result = await clearCache();
    setCacheFeedback(`Freed ${result.freedGb.toFixed(2)} GB`);
    setTimeout(() => setCacheFeedback(null), 3000);
  };

  const content = (
    <div className="space-y-4">
      {/* Server Actions */}
      <section>
        <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2">
          Server Actions
        </h4>
        <div className="space-y-2">
          {/* Restart Server */}
          <button
            onClick={() => setShowRestartConfirm(true)}
            disabled={isRestarting}
            className={cn(
              'w-full px-3 py-2.5 text-sm text-left rounded-lg transition-colors',
              'flex items-center gap-3',
              isRestarting
                ? 'bg-gray-700/50 text-gray-500 cursor-not-allowed'
                : 'bg-red-500/10 text-red-400 hover:bg-red-500/20'
            )}
          >
            <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            {isRestarting ? 'Restarting...' : 'Restart Server'}
          </button>

          {/* Clear CUDA Cache */}
          <button
            onClick={handleClearCache}
            className="w-full px-3 py-2.5 text-sm text-left rounded-lg transition-colors flex items-center gap-3 bg-gray-700/50 text-gray-300 hover:bg-gray-700"
          >
            <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
            <span className="flex-1">Clear CUDA Cache</span>
            {cacheFeedback && (
              <span className="text-xs text-green-400">{cacheFeedback}</span>
            )}
          </button>
        </div>
      </section>

      {/* System Info */}
      {ctx && (
        <section>
          <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2">
            System Info
          </h4>
          <div className="space-y-1.5 text-sm">
            {ctx.uptimeSeconds != null && (
              <div className="flex justify-between text-gray-400">
                <span>Uptime</span>
                <span className="text-gray-300">{formatUptime(ctx.uptimeSeconds)}</span>
              </div>
            )}
            <div className="flex justify-between text-gray-400">
              <span>Profile</span>
              <span className="text-gray-300">{ctx.profile}</span>
            </div>
            {ctx.vramUsedGb != null && ctx.vramTotalGb != null && (
              <div className="flex justify-between text-gray-400">
                <span>VRAM</span>
                <span className="text-gray-300 tabular-nums">
                  {ctx.vramUsedGb.toFixed(1)} / {ctx.vramTotalGb.toFixed(1)} GB
                  {ctx.vramPercent != null && ` (${ctx.vramPercent.toFixed(0)}%)`}
                </span>
              </div>
            )}
          </div>
        </section>
      )}

      {/* Pending Changes */}
      {ctx && (
        <RestartWarning fields={ctx.pendingRestartFields} className="p-3" />
      )}
    </div>
  );

  return (
    <>
      {variant === 'dropdown' ? (
        <div
          ref={menuRef}
          className="absolute left-0 top-full mt-1 w-72 bg-gray-800 border border-gray-700 rounded-xl shadow-xl z-30 p-4"
        >
          {content}
        </div>
      ) : (
        <div className="p-4">{content}</div>
      )}

      <ConfirmDialog
        isOpen={showRestartConfirm}
        title="Restart Server"
        message="The server will restart. All in-progress generations will be lost and models will need to reload. This typically takes 10-30 seconds."
        confirmLabel="Restart"
        confirmVariant="danger"
        onConfirm={handleRestart}
        onCancel={() => setShowRestartConfirm(false)}
      />
    </>
  );
}
