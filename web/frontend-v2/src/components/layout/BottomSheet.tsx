/**
 * BottomSheet Component
 *
 * Mobile history panel that slides up from the bottom.
 */

import { useState, useRef, useEffect } from 'react';
import { cn } from '@/utils';
import { useAppStore, useSessionStore } from '@/stores';
import { HistoryList } from '@/components/history/HistoryList';

interface BottomSheetProps {
  className?: string;
}

export function BottomSheet({ className }: BottomSheetProps) {
  const isHistoryOpen = useAppStore((s) => s.isHistoryOpen);
  const toggleHistory = useAppStore((s) => s.toggleHistory);
  const history = useSessionStore((s) => s.history);

  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState(0);
  const sheetRef = useRef<HTMLDivElement>(null);
  const startY = useRef(0);

  // Handle touch drag
  const handleTouchStart = (e: React.TouchEvent) => {
    startY.current = e.touches[0].clientY;
    setIsDragging(true);
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    if (!isDragging) return;
    const currentY = e.touches[0].clientY;
    const offset = currentY - startY.current;
    // Only allow dragging down
    if (offset > 0) {
      setDragOffset(offset);
    }
  };

  const handleTouchEnd = () => {
    setIsDragging(false);
    // If dragged more than 100px, close
    if (dragOffset > 100) {
      toggleHistory();
    }
    setDragOffset(0);
  };

  // Close on backdrop click
  const handleBackdropClick = () => {
    toggleHistory();
  };

  // Reset drag offset when closing
  useEffect(() => {
    if (!isHistoryOpen) {
      setDragOffset(0);
    }
  }, [isHistoryOpen]);

  if (!isHistoryOpen && history.length === 0) {
    return null;
  }

  return (
    <>
      {/* Backdrop */}
      {isHistoryOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40"
          onClick={handleBackdropClick}
        />
      )}

      {/* Toggle button when closed */}
      {!isHistoryOpen && history.length > 0 && (
        <button
          onClick={toggleHistory}
          className={cn(
            'fixed bottom-4 right-4 p-3 bg-gray-800 rounded-full border border-gray-700',
            'hover:bg-gray-700 transition-colors shadow-lg z-40',
            className
          )}
          title="Open history"
        >
          <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <span className="absolute -top-1 -right-1 w-5 h-5 bg-blue-500 rounded-full text-xs flex items-center justify-center font-medium">
            {history.length}
          </span>
        </button>
      )}

      {/* Sheet */}
      <div
        ref={sheetRef}
        className={cn(
          'fixed bottom-0 left-0 right-0 bg-gray-900 rounded-t-2xl z-50',
          'transition-transform duration-300 ease-out',
          isHistoryOpen ? 'translate-y-0' : 'translate-y-full',
          className
        )}
        style={{
          maxHeight: '70vh',
          transform: isHistoryOpen
            ? `translateY(${dragOffset}px)`
            : 'translateY(100%)',
        }}
      >
        {/* Drag handle */}
        <div
          className="flex justify-center py-3 cursor-grab active:cursor-grabbing"
          onTouchStart={handleTouchStart}
          onTouchMove={handleTouchMove}
          onTouchEnd={handleTouchEnd}
        >
          <div className="w-12 h-1.5 bg-gray-600 rounded-full" />
        </div>

        {/* Header */}
        <div className="flex items-center justify-between px-4 pb-3 border-b border-gray-700">
          <h2 className="font-medium">History</h2>
          <button
            onClick={toggleHistory}
            className="p-1 hover:bg-gray-800 rounded"
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
        <div className="overflow-y-auto" style={{ maxHeight: 'calc(70vh - 80px)' }}>
          <HistoryList />
        </div>
      </div>
    </>
  );
}
