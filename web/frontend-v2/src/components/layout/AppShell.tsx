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
import { LeftNav } from './LeftNav';
import { MobileNav } from './MobileNav';
import { Sidebar } from './Sidebar';
import { BottomSheet } from './BottomSheet';

interface AppShellProps {
  children: React.ReactNode;
}

export function AppShell({ children }: AppShellProps) {
  const isMobile = useIsMobile();
  const isDesktop = useIsDesktop();
  const isHistoryOpen = useAppStore((s) => s.isHistoryOpen);
  const isLeftNavOpen = useAppStore((s) => s.isLeftNavOpen);
  const setIsMobile = useAppStore((s) => s.setIsMobile);

  // Sync mobile state with store
  useEffect(() => {
    setIsMobile(isMobile);
  }, [isMobile, setIsMobile]);

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      {/* Left navigation (desktop only) */}
      {isDesktop && <LeftNav />}

      {/* Main content area */}
      <main
        className={cn(
          'min-h-screen transition-all',
          // Padding for mobile bottom nav
          isMobile && 'pb-16',
          // Adjust for left nav on desktop
          isDesktop && isLeftNavOpen && 'ml-72',
          // Adjust for history sidebar on desktop
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

      {/* Mobile bottom navigation */}
      {isMobile && <MobileNav />}
    </div>
  );
}
