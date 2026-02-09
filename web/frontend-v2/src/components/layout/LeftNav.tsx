/**
 * LeftNav Component
 *
 * Left navigation sidebar with model manager and pipeline selection.
 * Collapsible on desktop, hidden on mobile (replaced by bottom nav).
 */

import { useState } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { cn } from '@/utils';
import { useAppStore } from '@/stores';
import { ModelManager } from '@/components/model-manager/ModelManager';
import { SettingsMenu } from '@/components/settings/SettingsMenu';

export function LeftNav() {
  const isLeftNavOpen = useAppStore((s) => s.isLeftNavOpen);
  const toggleLeftNav = useAppStore((s) => s.toggleLeftNav);
  const activeTab = useAppStore((s) => s.activeTab);
  const setActiveTab = useAppStore((s) => s.setActiveTab);
  const selectedPipelineId = useAppStore((s) => s.selectedPipelineId);
  const selectPipeline = useAppStore((s) => s.selectPipeline);
  const getPipelineColor = useAppStore((s) => s.getPipelineColor);

  const [isModelsExpanded, setIsModelsExpanded] = useState(true);
  const [isPipelinesExpanded, setIsPipelinesExpanded] = useState(true);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  // Get pipelines for current tab
  const pipelines = useAppStore(
    useShallow((s) => {
      const category = activeTab === 'image' ? 'image' : 'video';
      return Object.values(s.pipelines).filter((p) => p.category === category);
    })
  );

  if (!isLeftNavOpen) {
    return (
      <button
        onClick={toggleLeftNav}
        className={cn(
          'fixed left-4 top-20 p-2 bg-gray-800 rounded-lg border border-gray-700',
          'hover:bg-gray-700 transition-colors z-20',
          'hidden md:block'
        )}
        title="Open navigation"
      >
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M4 6h16M4 12h16M4 18h16"
          />
        </svg>
      </button>
    );
  }

  return (
    <aside
      className={cn(
        'fixed left-0 top-14 bottom-0 w-72 bg-gray-900 border-r border-gray-700',
        'flex flex-col z-20',
        'hidden md:flex'
      )}
    >
      {/* Header with settings gear and close button */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
        <h2 className="font-semibold">Navigation</h2>
        <div className="flex items-center gap-1">
          {/* Settings gear */}
          <div className="relative">
            <button
              onClick={() => setIsSettingsOpen(!isSettingsOpen)}
              className={cn(
                'p-1 rounded transition-colors',
                isSettingsOpen ? 'bg-gray-700 text-gray-200' : 'hover:bg-gray-800 text-gray-400 hover:text-gray-300'
              )}
              title="Settings"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </button>
            {isSettingsOpen && (
              <SettingsMenu
                variant="dropdown"
                onClose={() => setIsSettingsOpen(false)}
              />
            )}
          </div>
          {/* Close nav */}
          <button
            onClick={toggleLeftNav}
            className="p-1 hover:bg-gray-800 rounded transition-colors"
            title="Close navigation"
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
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-6">
        {/* Model Manager Section */}
        <section>
          <button
            onClick={() => setIsModelsExpanded(!isModelsExpanded)}
            className="section-header w-full mb-3"
          >
            <span className="flex items-center gap-2">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01"
                />
              </svg>
              Load Models
            </span>
            <svg
              className={cn(
                'w-4 h-4 transition-transform',
                isModelsExpanded ? 'rotate-180' : ''
              )}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 9l-7 7-7-7"
              />
            </svg>
          </button>

          <div
            className={cn(
              'section-content',
              isModelsExpanded ? 'max-h-[2000px]' : 'max-h-0'
            )}
          >
            <ModelManager />
          </div>
        </section>

        {/* Tab Switcher */}
        <section>
          <h3 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-3">
            Category
          </h3>
          <div className="flex gap-2">
            <button
              onClick={() => setActiveTab('image')}
              className={cn(
                'flex-1 px-3 py-2 text-sm font-medium rounded-lg transition-colors',
                activeTab === 'image'
                  ? 'bg-gray-700 text-white'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
              )}
            >
              Image
            </button>
            <button
              onClick={() => setActiveTab('video')}
              className={cn(
                'flex-1 px-3 py-2 text-sm font-medium rounded-lg transition-colors',
                activeTab === 'video'
                  ? 'bg-gray-700 text-white'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
              )}
            >
              Video
            </button>
          </div>
        </section>

        {/* Pipeline List */}
        <section>
          <button
            onClick={() => setIsPipelinesExpanded(!isPipelinesExpanded)}
            className="section-header w-full mb-3"
          >
            <span className="flex items-center gap-2">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"
                />
              </svg>
              Pipelines
            </span>
            <svg
              className={cn(
                'w-4 h-4 transition-transform',
                isPipelinesExpanded ? 'rotate-180' : ''
              )}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 9l-7 7-7-7"
              />
            </svg>
          </button>

          <div
            className={cn(
              'section-content space-y-1',
              isPipelinesExpanded ? 'max-h-[2000px]' : 'max-h-0'
            )}
          >
            {pipelines.map((pipeline) => {
              const isSelected = pipeline.id === selectedPipelineId;
              const color = getPipelineColor(pipeline.id);

              return (
                <button
                  key={pipeline.id}
                  onClick={() => selectPipeline(pipeline.id)}
                  className={cn(
                    'w-full px-3 py-2.5 rounded-lg text-left transition-colors',
                    'flex items-center gap-3',
                    isSelected
                      ? 'bg-gray-800 border border-gray-700'
                      : 'hover:bg-gray-800/50'
                  )}
                >
                  {/* Color indicator */}
                  <div
                    className="w-1 h-8 rounded-full"
                    style={{ backgroundColor: isSelected ? color : '#4b5563' }}
                  />

                  {/* Pipeline info */}
                  <div className="flex-1 min-w-0">
                    <div
                      className="text-sm font-medium truncate"
                      style={{ color: isSelected ? color : undefined }}
                    >
                      {pipeline.name}
                    </div>
                    <div className="text-xs text-gray-500 truncate">
                      {pipeline.description}
                    </div>
                  </div>

                  {/* Selected indicator */}
                  {isSelected && (
                    <svg
                      className="w-4 h-4 flex-shrink-0"
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
          </div>
        </section>
      </div>
    </aside>
  );
}
