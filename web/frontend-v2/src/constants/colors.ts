/**
 * Shared pipeline color map.
 *
 * Maps color name strings (from pipeline schemas) to CSS hex values.
 * Used by appStore (getPipelineColor) and HistoryCard (badge coloring).
 */

export const PIPELINE_COLOR_MAP: Record<string, string> = {
  blue: '#3b82f6',
  purple: '#a855f7',
  orange: '#f97316',
  teal: '#14b8a6',
  green: '#22c55e',
  pink: '#ec4899',
};
