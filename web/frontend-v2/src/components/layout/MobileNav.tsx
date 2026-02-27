/**
 * MobileNav Component
 *
 * Bottom navigation for mobile devices.
 * Shows category tabs, pipeline selector, and model management.
 */

import { useState } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { cn } from '@/utils';
import { useAppStore } from '@/stores';
import { ModelManager } from '@/components/model-manager/ModelManager';

export function MobileNav() {
  const activeTab = useAppStore((s) => s.activeTab);
  const setActiveTab = useAppStore((s) => s.setActiveTab);
  const selectedPipelineId = useAppStore((s) => s.selectedPipelineId);
  const selectPipeline = useAppStore((s) => s.selectPipeline);
  const getPipelineColor = useAppStore((s) => s.getPipelineColor);

  const [isModelSheetOpen, setIsModelSheetOpen] = useState(false);
  const [isPipelineSheetOpen, setIsPipelineSheetOpen] = useState(false);

  // Get pipelines for current tab
  const pipelines = useAppStore(
    useShallow((s) => {
      const category = activeTab === 'image' ? 'image' : 'video';
      return Object.values(s.pipelines).filter((p) => p.category === category);
    })
  );

  // Get current pipeline info
  const currentPipeline = useAppStore((s) =>
    selectedPipelineId ? s.pipelines[selectedPipelineId] : null
  );
  const currentColor = selectedPipelineId ? getPipelineColor(selectedPipelineId) : '#6b7280';

  const handlePipelineSelect = (pipelineId: string) => {
    selectPipeline(pipelineId);
    setIsPipelineSheetOpen(false);
  };

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

          {/* Pipeline selector button */}
          <button
            onClick={() => setIsPipelineSheetOpen(true)}
            className="flex-1 h-full flex flex-col items-center justify-center gap-1 transition-colors"
            style={{ color: currentColor }}
          >
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
              />
            </svg>
            <span className="text-xs font-medium truncate max-w-[60px]">
              {currentPipeline?.name || 'Pipeline'}
            </span>
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

      {/* Pipeline selection bottom sheet */}
      {isPipelineSheetOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 md:hidden"
          onClick={() => setIsPipelineSheetOpen(false)}
        >
          <div
            className="absolute bottom-0 left-0 right-0 bg-gray-900 rounded-t-2xl border-t border-gray-700 max-h-[60vh] flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Sheet header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
              <h2 className="font-semibold">
                Select {activeTab === 'image' ? 'Image' : 'Video'} Pipeline
              </h2>
              <button
                onClick={() => setIsPipelineSheetOpen(false)}
                className="p-1 hover:bg-gray-800 rounded-sm transition-colors"
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

            {/* Pipeline list */}
            <div className="flex-1 overflow-y-auto px-4 py-3 space-y-2">
              {pipelines.map((pipeline) => {
                const isSelected = pipeline.id === selectedPipelineId;
                const color = getPipelineColor(pipeline.id);

                return (
                  <button
                    key={pipeline.id}
                    onClick={() => handlePipelineSelect(pipeline.id)}
                    className={cn(
                      'w-full px-4 py-3 rounded-lg text-left transition-colors',
                      'flex items-center gap-3',
                      isSelected
                        ? 'bg-gray-800 border border-gray-600'
                        : 'bg-gray-800/50 border border-transparent active:bg-gray-700'
                    )}
                  >
                    {/* Color indicator */}
                    <div
                      className="w-1.5 h-10 rounded-full shrink-0"
                      style={{ backgroundColor: isSelected ? color : '#6b7280' }}
                    />

                    {/* Pipeline info */}
                    <div className="flex-1 min-w-0">
                      <div
                        className="text-sm font-medium"
                        style={{ color: isSelected ? color : undefined }}
                      >
                        {pipeline.name}
                      </div>
                      <div className="text-xs text-gray-500 line-clamp-2">
                        {pipeline.description}
                      </div>
                    </div>

                    {/* Selected checkmark */}
                    {isSelected && (
                      <svg
                        className="w-5 h-5 shrink-0"
                        style={{ color }}
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path
                          fillRule="evenodd"
                          d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                          clipRule="evenodd"
                        />
                      </svg>
                    )}
                  </button>
                );
              })}

              {pipelines.length === 0 && (
                <p className="text-center text-gray-500 py-4">
                  No {activeTab === 'image' ? 'image' : 'video'} pipelines available
                </p>
              )}
            </div>
          </div>
        </div>
      )}

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
                className="p-1 hover:bg-gray-800 rounded-sm transition-colors"
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
