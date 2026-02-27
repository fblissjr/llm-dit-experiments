/**
 * HistoryList Component
 *
 * Scrollable list of history items with empty state.
 */

import { useState } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useSessionStore } from '@/stores';
import { HistoryCard } from './HistoryCard';
import { ConfirmDialog } from '@/components/common/ConfirmDialog';

export function HistoryList() {
  // Use useShallow for array selector to prevent infinite re-renders
  const history = useSessionStore(useShallow((s) => s.history));
  const clearHistory = useSessionStore((s) => s.clearHistory);
  const [showClearConfirm, setShowClearConfirm] = useState(false);

  if (history.length === 0) {
    return (
      <div className="p-8 text-center text-gray-500">
        <svg
          className="w-12 h-12 mx-auto mb-4 opacity-50"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
        <p>No generation history yet</p>
        <p className="text-sm mt-1">Your generations will appear here</p>
      </div>
    );
  }

  return (
    <div className="p-3 space-y-2">
      {/* Clear all button */}
      <div className="flex justify-end mb-2">
        <button
          onClick={() => setShowClearConfirm(true)}
          className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
        >
          Clear all
        </button>
      </div>

      {/* History items */}
      {history.map((item) => (
        <HistoryCard key={item.id} item={item} />
      ))}

      <ConfirmDialog
        isOpen={showClearConfirm}
        title="Clear History"
        message="This will permanently delete all generation history."
        confirmLabel="Clear"
        confirmVariant="danger"
        onConfirm={() => { clearHistory(); setShowClearConfirm(false); }}
        onCancel={() => setShowClearConfirm(false)}
      />
    </div>
  );
}
