/**
 * History Panel
 *
 * Lists generation history with filtering and comparison support.
 */

import { useHistoryStore } from '@/stores/historyStore';
import { useGenerationStore } from '@/stores/generationStore';
import { usePipelineStore } from '@/stores/pipelineStore';
import { HistoryCard } from './HistoryCard';
import { ComparisonView } from './ComparisonView';

export function HistoryPanel() {
  const {
    items,
    isCompareMode,
    selectedForCompare,
    toggleCompareMode,
    selectForCompare,
    deselectForCompare,
    removeItem,
    clearHistory,
  } = useHistoryStore();
  const { restoreFromHistory } = useGenerationStore();
  const { selectedPipelineId, selectPipeline } = usePipelineStore();

  const handleCardClick = (item: typeof items[0]) => {
    if (isCompareMode) {
      // Toggle selection for comparison
      if (selectedForCompare.includes(item.id)) {
        deselectForCompare(item.id);
      } else {
        selectForCompare(item.id);
      }
    } else {
      // Restore params
      selectPipeline(item.pipelineId);
      restoreFromHistory(item.pipelineId, item.params);
    }
  };

  // Show comparison view if two items selected
  if (isCompareMode && selectedForCompare.length === 2) {
    return <ComparisonView />;
  }

  if (items.length === 0) {
    return (
      <div className="text-center py-8">
        <div className="text-4xl mb-3 opacity-50">📜</div>
        <p className="text-gray-500 text-sm">No generations yet</p>
        <p className="text-gray-600 text-xs mt-1">
          Your history will appear here
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header with actions */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-gray-500">
          {items.length} generation{items.length !== 1 ? 's' : ''}
        </span>
        <div className="flex items-center gap-2">
          <button
            onClick={toggleCompareMode}
            className={`
              text-xs px-2 py-1 rounded transition-colors
              ${isCompareMode
                ? 'bg-blue-500/20 text-blue-400'
                : 'text-gray-400 hover:text-gray-200 hover:bg-gray-700'
              }
            `}
          >
            {isCompareMode ? 'Exit Compare' : 'Compare'}
          </button>
          <button
            onClick={clearHistory}
            className="text-xs text-gray-500 hover:text-red-400 transition-colors"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Compare mode instructions */}
      {isCompareMode && (
        <div className="text-xs text-gray-400 bg-gray-800/50 rounded-lg p-2">
          Select 2 items to compare ({selectedForCompare.length}/2 selected)
        </div>
      )}

      {/* History grid */}
      <div className="grid grid-cols-2 gap-2">
        {items.map((item) => (
          <HistoryCard
            key={item.id}
            item={item}
            isSelected={selectedForCompare.includes(item.id)}
            isCompareMode={isCompareMode}
            onClick={() => handleCardClick(item)}
            onDelete={() => removeItem(item.id)}
          />
        ))}
      </div>
    </div>
  );
}
