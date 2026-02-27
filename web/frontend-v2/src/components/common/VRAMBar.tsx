/**
 * VRAMBar Component
 *
 * Shared VRAM usage progress bar with consistent thresholds (70/90).
 * Accepts values in GB. Callers convert from MB if needed.
 */

import { cn } from '@/utils';

interface VRAMBarProps {
  usedGb: number;
  totalGb: number;
  /** Bar height class. Default 'h-1.5' for compact, use 'h-2' for larger. */
  height?: string;
  /** Whether to show the text label. Default true. */
  showLabel?: boolean;
}

export function VRAMBar({ usedGb, totalGb, height = 'h-1.5', showLabel = true }: VRAMBarProps) {
  const percent = totalGb > 0 ? (usedGb / totalGb) * 100 : 0;

  return (
    <div className="flex items-center gap-2 text-xs">
      <div className={cn('flex-1 min-w-[5rem] bg-gray-700 rounded-full overflow-hidden', height)}>
        <div
          className={cn(
            'h-full transition-all',
            percent > 90 ? 'bg-red-500' : percent > 70 ? 'bg-yellow-500' : 'bg-blue-500'
          )}
          style={{ width: `${Math.min(100, percent)}%` }}
        />
      </div>
      {showLabel && (
        <span className="text-gray-400 tabular-nums whitespace-nowrap">
          {usedGb.toFixed(1)}/{totalGb.toFixed(1)}
        </span>
      )}
    </div>
  );
}
