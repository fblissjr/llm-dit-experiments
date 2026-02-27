/**
 * RestartWarning Component
 *
 * Yellow warning banner shown when config changes require a server restart.
 * Used in both StatusBar (expanded details) and SettingsMenu.
 */

interface RestartWarningProps {
  fields: string[];
  className?: string;
}

export function RestartWarning({ fields, className }: RestartWarningProps) {
  if (fields.length === 0) return null;

  return (
    <div className={`flex items-start gap-2 p-2 bg-yellow-500/10 border border-yellow-500/20 rounded-lg ${className ?? ''}`}>
      <svg className="w-4 h-4 text-yellow-400 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
      <div>
        <p className="text-xs font-medium text-yellow-300">Restart required</p>
        <p className="text-xs text-yellow-400/70 mt-0.5">
          Changed: {fields.join(', ')}
        </p>
      </div>
    </div>
  );
}
