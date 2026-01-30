/**
 * Bottom Sheet
 *
 * Mobile-friendly panel that slides up from bottom.
 * Used for history on mobile devices.
 */

import { useRef, useState, useEffect } from 'react';

interface BottomSheetProps {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  title?: string;
  children: React.ReactNode;
  minHeight?: number;   // Collapsed height in px
  maxHeight?: string;   // Open height (e.g., "80vh")
}

export function BottomSheet({
  isOpen,
  onOpenChange,
  title,
  children,
  minHeight = 48,
  maxHeight = '70vh',
}: BottomSheetProps) {
  const sheetRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [startY, setStartY] = useState(0);
  const [currentTranslate, setCurrentTranslate] = useState(0);

  // Handle touch start
  const handleTouchStart = (e: React.TouchEvent) => {
    setIsDragging(true);
    setStartY(e.touches[0].clientY);
    setCurrentTranslate(0);
  };

  // Handle touch move
  const handleTouchMove = (e: React.TouchEvent) => {
    if (!isDragging) return;
    const diff = e.touches[0].clientY - startY;
    setCurrentTranslate(diff);
  };

  // Handle touch end
  const handleTouchEnd = () => {
    setIsDragging(false);

    // Threshold for open/close
    const threshold = 50;

    if (isOpen) {
      // If dragged down, close
      if (currentTranslate > threshold) {
        onOpenChange(false);
      }
    } else {
      // If dragged up, open
      if (currentTranslate < -threshold) {
        onOpenChange(true);
      }
    }

    setCurrentTranslate(0);
  };

  // Handle click on header to toggle
  const handleHeaderClick = () => {
    onOpenChange(!isOpen);
  };

  // Close on escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onOpenChange(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onOpenChange]);

  const translateY = isOpen
    ? Math.max(0, currentTranslate) // Can only drag down when open
    : Math.min(0, currentTranslate); // Can only drag up when closed

  return (
    <div
      ref={sheetRef}
      className={`
        fixed inset-x-0 bottom-0 z-30
        bg-gray-800 border-t border-gray-700 rounded-t-2xl shadow-2xl
        transform transition-transform duration-300 ease-out
        ${isDragging ? 'transition-none' : ''}
      `}
      style={{
        height: isOpen ? maxHeight : minHeight,
        transform: `translateY(${translateY}px)`,
      }}
    >
      {/* Drag handle area */}
      <div
        className="touch-none"
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
        onClick={handleHeaderClick}
      >
        {/* Visual drag handle */}
        <div className="flex justify-center py-3">
          <div className="w-10 h-1 bg-gray-600 rounded-full" />
        </div>

        {/* Header */}
        {title && (
          <div className="px-4 pb-2 flex items-center justify-between">
            <h3 className="text-sm font-medium text-gray-300">{title}</h3>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onOpenChange(!isOpen);
              }}
              className="p-1 text-gray-400 hover:text-gray-200 transition-colors"
            >
              <svg
                className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
              </svg>
            </button>
          </div>
        )}
      </div>

      {/* Content */}
      <div
        className={`
          overflow-y-auto px-4 pb-4
          ${isOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'}
          transition-opacity duration-200
        `}
        style={{
          height: isOpen ? `calc(${maxHeight} - ${minHeight}px)` : 0,
        }}
      >
        {children}
      </div>
    </div>
  );
}
