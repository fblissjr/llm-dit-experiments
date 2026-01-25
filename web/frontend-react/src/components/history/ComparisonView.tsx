/**
 * Comparison View
 *
 * Side-by-side comparison of two history items with parameter diff.
 */

import { useHistoryStore } from '@/stores/historyStore';
import { useGenerationStore } from '@/stores/generationStore';
import { usePipelineStore } from '@/stores/pipelineStore';

export function ComparisonView() {
  const {
    items,
    selectedForCompare,
    getComparisonDiff,
    clearCompareSelection,
    toggleCompareMode,
  } = useHistoryStore();
  const { restoreFromHistory, generate } = useGenerationStore();
  const { pipelines, selectPipeline } = usePipelineStore();

  const itemA = items.find((i) => i.id === selectedForCompare[0]);
  const itemB = items.find((i) => i.id === selectedForCompare[1]);
  const diffs = getComparisonDiff();

  if (!itemA || !itemB) {
    return null;
  }

  const handleRegenerate = (item: typeof itemA) => {
    selectPipeline(item.pipelineId);
    restoreFromHistory(item.pipelineId, item.params);
    const pipeline = pipelines[item.pipelineId];
    if (pipeline) {
      generate(item.pipelineId, pipeline.endpoint, pipeline.supports_streaming);
    }
    toggleCompareMode();
  };

  const handleClose = () => {
    clearCompareSelection();
    toggleCompareMode();
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-300">
          Comparing 2 generations
        </h3>
        <button
          onClick={handleClose}
          className="text-xs text-gray-400 hover:text-gray-200 transition-colors"
        >
          × Close
        </button>
      </div>

      {/* Side-by-side images */}
      <div className="grid grid-cols-2 gap-2">
        <ComparisonItem
          item={itemA}
          label="A"
          onRegenerate={() => handleRegenerate(itemA)}
        />
        <ComparisonItem
          item={itemB}
          label="B"
          onRegenerate={() => handleRegenerate(itemB)}
        />
      </div>

      {/* Parameter differences */}
      {diffs.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wide">
            Parameter Differences
          </h4>
          <div className="bg-gray-800/50 rounded-lg divide-y divide-gray-700">
            {diffs.map((diff) => (
              <div
                key={diff.key}
                className="flex items-center justify-between px-3 py-2 text-sm"
              >
                <span className="text-gray-400">{diff.label}</span>
                <div className="flex items-center gap-2 font-mono text-xs">
                  <span className="text-blue-400">{formatValue(diff.valueA)}</span>
                  <span className="text-gray-600">→</span>
                  <span className="text-purple-400">{formatValue(diff.valueB)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {diffs.length === 0 && (
        <p className="text-sm text-gray-500 text-center py-4">
          No parameter differences found
        </p>
      )}
    </div>
  );
}

interface ComparisonItemProps {
  item: {
    id: string;
    thumbnailUrl: string;
    shortPrompt: string;
    keyParams: string;
    pipelineName: string;
  };
  label: string;
  onRegenerate: () => void;
}

function ComparisonItem({ item, label, onRegenerate }: ComparisonItemProps) {
  return (
    <div className="space-y-2">
      {/* Label */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-gray-400">
          {label}: {item.pipelineName}
        </span>
      </div>

      {/* Image */}
      <div className="aspect-square rounded-lg overflow-hidden border border-gray-700">
        <img
          src={item.thumbnailUrl}
          alt={item.shortPrompt}
          className="w-full h-full object-cover"
        />
      </div>

      {/* Params */}
      <p className="text-xs text-gray-500 truncate">{item.keyParams}</p>

      {/* Regenerate button */}
      <button
        onClick={onRegenerate}
        className="w-full btn-ghost text-xs py-1.5"
      >
        Regenerate {label}
      </button>
    </div>
  );
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return 'none';
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'number') return value.toString();
  if (typeof value === 'string') {
    if (value.length > 20) return value.substring(0, 17) + '...';
    return value;
  }
  return JSON.stringify(value);
}
