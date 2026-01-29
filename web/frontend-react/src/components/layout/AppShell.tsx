/**
 * AppShell
 *
 * Main layout component. Handles responsive structure:
 * - Desktop: Header + main content + sidebar
 * - Mobile: Header + main content + bottom sheet
 */

import { Header } from './Header';
import { BottomSheet } from './BottomSheet';
import { ModelManagementPanel } from '../model/ModelManagementPanel';
import { ModelSidebar } from '../model/ModelSidebar';
import { Notifications } from '../common/Notifications';
import { useUIStore } from '@/stores/uiStore';

interface AppShellProps {
  main: React.ReactNode;
  sidebar?: React.ReactNode;
}

export function AppShell({ main, sidebar }: AppShellProps) {
  const { isMobile, isDesktop, isHistoryPanelOpen, toggleHistoryPanel } = useUIStore();

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col">
      {/* Header */}
      <Header />

      {/* Main content area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Main content */}
        <main
          className={`
            flex-1 overflow-y-auto p-4 lg:p-6
            ${isDesktop && sidebar ? 'lg:pr-80' : ''}
          `}
        >
          <div className="max-w-6xl mx-auto">
            {main}
          </div>
        </main>

        {/* Desktop sidebar - top value accounts for header height (title + tabs + padding) */}
        {isDesktop && sidebar && (
          <aside className="w-80 fixed right-0 top-[5.5rem] bottom-0 border-l border-gray-700 bg-gray-800/50 overflow-y-auto">
            <div className="p-4">
              <h2 className="text-sm font-medium text-gray-400 uppercase tracking-wide mb-4">
                History
              </h2>
              {sidebar}
            </div>
          </aside>
        )}
      </div>

      {/* Mobile bottom sheet for history */}
      {isMobile && sidebar && (
        <BottomSheet
          isOpen={isHistoryPanelOpen}
          onOpenChange={toggleHistoryPanel}
          title="Generation History"
          maxHeight="70vh"
        >
          {sidebar}
        </BottomSheet>
      )}

      {/* Model management - sidebar for desktop, modal for mobile */}
      <ModelSidebar />
      <ModelManagementPanel />

      {/* Notifications */}
      <Notifications />

      {/* Mobile safe area padding */}
      {isMobile && <div className="h-12" />}
    </div>
  );
}
