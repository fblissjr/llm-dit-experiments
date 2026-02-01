/**
 * AppShell Component
 *
 * Root layout component that orchestrates the overall app structure.
 * Handles responsive layout switching between desktop and mobile.
 */

import { useEffect } from 'react';
import { cn } from '@/utils';
import { useAppStore } from '@/stores';
import { useIsMobile, useIsDesktop } from '@/hooks';
import { TabBar } from './TabBar';
import { Sidebar } from './Sidebar';
import { BottomSheet } from './BottomSheet';

interface AppShellProps {
  children: React.ReactNode;
}

export function AppShell({ children }: AppShellProps) {
  const isMobile = useIsMobile();
  const isDesktop = useIsDesktop();
  const isHistoryOpen = useAppStore((s) => s.isHistoryOpen);
  const setIsMobile = useAppStore((s) => s.setIsMobile);

  // Sync mobile state with store
  useEffect(() => {
    setIsMobile(isMobile);
  }, [isMobile, setIsMobile]);

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      {/* Top navigation */}
      <header className="fixed top-0 left-0 right-0 bg-gray-900/95 backdrop-blur border-b border-gray-800 z-30">
        <TabBar />
      </header>

      {/* Main content area */}
      <main
        className={cn(
          'pt-14 min-h-screen',
          // Adjust for sidebar on desktop
          isDesktop && isHistoryOpen && 'mr-80'
        )}
      >
        <div className="max-w-6xl mx-auto p-4">
          {children}
        </div>
      </main>

      {/* History panel - desktop sidebar or mobile bottom sheet */}
      {isDesktop ? (
        <Sidebar />
      ) : (
        <BottomSheet />
      )}
    </div>
  );
}
