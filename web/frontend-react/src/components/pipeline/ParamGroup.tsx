/**
 * Parameter Group
 *
 * Collapsible section for grouping related parameters.
 * Implements progressive disclosure.
 */

import { useState } from 'react';
import type { ParamGroup as ParamGroupType } from '@/types';

interface ParamGroupProps {
  group: ParamGroupType;
  children: React.ReactNode;
  defaultExpanded?: boolean;
}

export function ParamGroup({
  group,
  children,
  defaultExpanded,
}: ParamGroupProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded ?? group.defaultExpanded);

  // Basic group is always expanded and non-collapsible
  if (group.id === 'basic') {
    return (
      <div className="space-y-4">
        {children}
      </div>
    );
  }

  return (
    <div className="border border-gray-700 rounded-lg overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full section-header px-4 py-3 bg-gray-800/50"
      >
        <span>{group.label}</span>
        <svg
          className={`w-4 h-4 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Content */}
      <div
        className={`
          section-content px-4
          ${isExpanded ? 'py-4 max-h-[2000px]' : 'max-h-0'}
        `}
      >
        <div className="space-y-4">
          {children}
        </div>
      </div>
    </div>
  );
}
