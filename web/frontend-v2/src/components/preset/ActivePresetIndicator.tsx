/**
 * ActivePresetIndicator - Thin status bar showing preset state.
 *
 * Returns null when no preset is active. Two visible states:
 * - Active (clean): checkmark + "Using 'name' preset" + [Clear]
 * - Active (modified): warning + "'name' preset modified" + [Restore] [Clear]
 */

interface ActivePresetIndicatorProps {
  presetName: string | null;
  isModified: boolean;
  pipelineColor: string;
  onRestore: () => void;
  onClear: () => void;
}

export function ActivePresetIndicator({
  presetName,
  isModified,
  pipelineColor,
  onRestore,
  onClear,
}: ActivePresetIndicatorProps) {
  if (!presetName) {
    return null;
  }

  if (isModified) {
    return (
      <div className="flex items-center gap-2 text-xs py-1 flex-wrap">
        <span
          className="w-4 h-4 inline-flex items-center justify-center rounded-full text-white text-[10px] flex-shrink-0"
          style={{ backgroundColor: '#d97706' }}
        >
          <svg width="8" height="8" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path d="M8 3v6M8 12v1" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
          </svg>
        </span>
        <span className="text-gray-300">
          &apos;{presetName}&apos; preset modified
        </span>
        <button
          type="button"
          onClick={onRestore}
          className="text-gray-400 hover:text-white underline underline-offset-2 transition-colors"
        >
          Restore
        </button>
        <button
          type="button"
          onClick={onClear}
          className="text-gray-500 hover:text-gray-300 transition-colors"
          aria-label="Clear preset"
        >
          <svg width="10" height="10" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
        </button>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2 text-xs py-1">
      <span
        className="w-4 h-4 inline-flex items-center justify-center rounded-full text-white text-[10px] flex-shrink-0"
        style={{ backgroundColor: pipelineColor }}
      >
        <svg width="8" height="8" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <path d="M3 8.5l3.5 3.5L13 4" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </span>
      <span className="text-gray-300">
        Using &apos;{presetName}&apos; preset
      </span>
      <button
        type="button"
        onClick={onClear}
        className="text-gray-500 hover:text-gray-300 transition-colors ml-auto"
        aria-label="Clear preset"
      >
        <svg width="10" height="10" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        </svg>
      </button>
    </div>
  );
}
