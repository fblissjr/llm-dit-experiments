/**
 * Sidebar Component
 *
 * Desktop history panel shown on the right side.
 */

import { useShallow } from 'zustand/react/shallow';
import { cn } from '@/utils';
import { useAppStore, useSessionStore } from '@/stores';
import { HistoryList } from '@/components/history/HistoryList';

interface SidebarProps {
  className?: string;
}

export function Sidebar({ className }: SidebarProps) {
  const isHistoryOpen = useAppStore((s) => s.isHistoryOpen);
  const toggleHistory = useAppStore((s) => s.toggleHistory);
  // Use useShallow for array selector - history array changes frequently
  const history = useSessionStore(useShallow((s) => s.history));

  if (!isHistoryOpen) {
    return (
      <button
        onClick={toggleHistory}
        className={cn(
          'fixed right-4 top-20 p-2 bg-gray-800 rounded-lg border border-gray-700',
          'hover:bg-gray-700 transition-colors',
          className
        )}
        title="Open history"
      >
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
        {history.length > 0 && (
          <span className="absolute -top-1 -right-1 w-4 h-4 bg-blue-500 rounded-full text-xs flex items-center justify-center">
            {history.length}
          </span>
        )}
      </button>
    );
  }

  return (
    <aside
      className={cn(
        'fixed right-0 top-14 bottom-0 w-80 bg-gray-900 border-l border-gray-700',
        'flex flex-col',
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
        <h2 className="font-medium">History</h2>
        <button
          onClick={toggleHistory}
          className="p-1 hover:bg-gray-800 rounded"
          title="Close history"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      </div>

      {/* History list */}
      <div className="flex-1 overflow-y-auto">
        <HistoryList />
      </div>
    </aside>
  );
}
