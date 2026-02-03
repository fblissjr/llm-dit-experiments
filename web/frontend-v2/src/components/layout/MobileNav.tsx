/**
 * MobileNav Component
 *
 * Bottom navigation for mobile devices.
 * Shows category tabs and opens a sheet for model management.
 */

import { useState } from 'react';
import { cn } from '@/utils';
import { useAppStore } from '@/stores';
import { ModelManager } from '@/components/models/ModelManager';

export function MobileNav() {
  const activeTab = useAppStore((s) => s.activeTab);
  const setActiveTab = useAppStore((s) => s.setActiveTab);
  const [isModelSheetOpen, setIsModelSheetOpen] = useState(false);

  return (
    <>
      {/* Bottom navigation bar */}
      <nav className="fixed bottom-0 left-0 right-0 bg-gray-900 border-t border-gray-700 z-30 md:hidden">
        <div className="flex items-center justify-around h-16">
          {/* Image tab */}
          <button
            onClick={() => setActiveTab('image')}
            className={cn(
              'flex-1 h-full flex flex-col items-center justify-center gap-1 transition-colors',
              activeTab === 'image' ? 'text-blue-400' : 'text-gray-400'
            )}
          >
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
              />
            </svg>
            <span className="text-xs font-medium">Image</span>
          </button>

          {/* Video tab */}
          <button
            onClick={() => setActiveTab('video')}
            className={cn(
              'flex-1 h-full flex flex-col items-center justify-center gap-1 transition-colors',
              activeTab === 'video' ? 'text-purple-400' : 'text-gray-400'
            )}
          >
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
              />
            </svg>
            <span className="text-xs font-medium">Video</span>
          </button>

          {/* Models button */}
          <button
            onClick={() => setIsModelSheetOpen(true)}
            className="flex-1 h-full flex flex-col items-center justify-center gap-1 text-gray-400 transition-colors active:text-gray-200"
          >
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01"
              />
            </svg>
            <span className="text-xs font-medium">Models</span>
          </button>
        </div>
      </nav>

      {/* Models bottom sheet */}
      {isModelSheetOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 md:hidden"
          onClick={() => setIsModelSheetOpen(false)}
        >
          <div
            className="absolute bottom-0 left-0 right-0 bg-gray-900 rounded-t-2xl border-t border-gray-700 max-h-[80vh] flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Sheet header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
              <h2 className="font-semibold">Model Manager</h2>
              <button
                onClick={() => setIsModelSheetOpen(false)}
                className="p-1 hover:bg-gray-800 rounded transition-colors"
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

            {/* Sheet content */}
            <div className="flex-1 overflow-y-auto px-4 py-4">
              <ModelManager />
            </div>
          </div>
        </div>
      )}
    </>
  );
}
