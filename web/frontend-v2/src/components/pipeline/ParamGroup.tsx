/**
 * ParamGroup Component
 *
 * Collapsible section for grouping related parameters.
 * "basic" group is always expanded and non-collapsible.
 */

import { useState } from 'react';
import { cn } from '@/utils';
import type { GroupType } from '@/api/types';

// Group labels for display
const groupLabels: Record<GroupType, string> = {
  basic: 'Basic',
  advanced: 'Advanced',
  expert: 'Expert',
  scheduler: 'Scheduler',
  optimization: 'Optimization',
  enhancement: 'Enhancement',
};

interface ParamGroupProps {
  groupId: GroupType;
  children: React.ReactNode;
  defaultExpanded?: boolean;
}

export function ParamGroup({
  groupId,
  children,
  defaultExpanded,
}: ParamGroupProps) {
  const [isExpanded, setIsExpanded] = useState(
    defaultExpanded ?? groupId === 'basic'
  );

  // Basic group is always expanded and non-collapsible
  if (groupId === 'basic') {
    return <div className="space-y-4">{children}</div>;
  }

  const label = groupLabels[groupId] ?? groupId;

  return (
    <div className="border border-gray-700 rounded-lg overflow-hidden">
      {/* Header */}
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full section-header px-4 py-3 bg-gray-800/50 hover:bg-gray-800 transition-colors"
      >
        <span>{label}</span>
        <svg
          className={cn(
            'w-4 h-4 transition-transform',
            isExpanded && 'rotate-180'
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

      {/* Content */}
      <div
        className={cn(
          'section-content px-4',
          isExpanded ? 'py-4 max-h-[2000px]' : 'max-h-0 overflow-hidden'
        )}
      >
        <div className="space-y-4">{children}</div>
      </div>
    </div>
  );
}
