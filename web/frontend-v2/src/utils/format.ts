/**
 * Formatting utilities shared across components.
 */

/**
 * Format uptime seconds into a human-readable duration string.
 *
 * Examples: "3h 12m", "45m", "<1m"
 */
export function formatUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m`;
  return '<1m';
}
