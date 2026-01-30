/**
 * VRAM Bar
 *
 * Visual indicator of VRAM usage. Shows at-a-glance status.
 */

import { formatVRAM, getUtilizationLevel } from '@/types/model';

interface VRAMBarProps {
  usedMB: number;
  totalMB: number;
  onClick?: () => void;
  compact?: boolean;
  className?: string;
}

export function VRAMBar({
  usedMB,
  totalMB,
  onClick,
  compact = false,
  className = '',
}: VRAMBarProps) {
  const percent = totalMB > 0 ? (usedMB / totalMB) * 100 : 0;
  const level = getUtilizationLevel(percent);

  const levelColors = {
    low: 'bg-green-500',
    medium: 'bg-yellow-500',
    high: 'bg-orange-500',
    critical: 'bg-red-500',
  };

  if (compact) {
    return (
      <div
        className={`flex items-center gap-2 ${onClick ? 'cursor-pointer' : ''} ${className}`}
        onClick={onClick}
      >
        <div className="w-24 h-2 bg-gray-700 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-300 ${levelColors[level]}`}
            style={{ width: `${percent}%` }}
          />
        </div>
        <span className="text-xs text-gray-400">
          {formatVRAM(usedMB)}/{formatVRAM(totalMB)}
        </span>
      </div>
    );
  }

  return (
    <div
      className={`flex items-center gap-3 ${onClick ? 'cursor-pointer hover:opacity-80' : ''} ${className}`}
      onClick={onClick}
      title="Click to manage models"
    >
      {/* Visual bar */}
      <div className="w-32 h-3 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-300 ${levelColors[level]}`}
          style={{ width: `${percent}%` }}
        />
      </div>

      {/* Text label */}
      <span className="text-sm text-gray-300 whitespace-nowrap">
        {formatVRAM(usedMB)}/{formatVRAM(totalMB)}
      </span>
    </div>
  );
}
