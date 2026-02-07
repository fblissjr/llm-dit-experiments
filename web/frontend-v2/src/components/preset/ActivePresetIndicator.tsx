/**
 * ActivePresetIndicator - Thin status bar showing preset state.
 *
 * Three states:
 * - No preset: "No preset selected" (dimmed)
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
    return (
      <div className="text-xs text-gray-500 py-1">
        No preset selected
      </div>
    );
  }

  if (isModified) {
    return (
      <div className="flex items-center gap-2 text-xs py-1 flex-wrap">
        <span
          className="w-4 h-4 inline-flex items-center justify-center rounded-full text-white text-[10px] flex-shrink-0"
          style={{ backgroundColor: '#d97706' }}
        >
          !
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
          &#10005;
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
        &#10003;
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
        &#10005;
      </button>
    </div>
  );
}
