/**
 * TabBar Component
 *
 * Top-level navigation between Image and Video tabs.
 */

import { cn } from '@/utils';
import { useAppStore } from '@/stores';

export function TabBar() {
  const activeTab = useAppStore((s) => s.activeTab);
  const setActiveTab = useAppStore((s) => s.setActiveTab);
  const vram = useAppStore((s) => s.vram);

  return (
    <div className="flex items-center justify-between px-4 py-2 border-b border-gray-700">
      {/* Tab buttons */}
      <div className="flex gap-1">
        <button
          onClick={() => setActiveTab('image')}
          className={cn(
            'tab',
            activeTab === 'image' ? 'tab-active' : 'tab-inactive'
          )}
        >
          Image
        </button>
        <button
          onClick={() => setActiveTab('video')}
          className={cn(
            'tab',
            activeTab === 'video' ? 'tab-active' : 'tab-inactive'
          )}
        >
          Video
        </button>
      </div>

      {/* VRAM indicator */}
      {vram && (
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <span>VRAM:</span>
          <div className="w-24 h-2 bg-gray-700 rounded-full overflow-hidden">
            <div
              className={cn(
                'h-full rounded-full transition-all',
                vram.utilizationPercent < 50 && 'bg-green-500',
                vram.utilizationPercent >= 50 && vram.utilizationPercent < 75 && 'bg-yellow-500',
                vram.utilizationPercent >= 75 && vram.utilizationPercent < 90 && 'bg-orange-500',
                vram.utilizationPercent >= 90 && 'bg-red-500'
              )}
              style={{ width: `${vram.utilizationPercent}%` }}
            />
          </div>
          <span className="font-mono text-xs">
            {(vram.usedMb / 1024).toFixed(1)}/{(vram.totalMb / 1024).toFixed(0)}GB
          </span>
        </div>
      )}
    </div>
  );
}
