/**
 * Generation Types
 *
 * Types for tracking generation state, progress, and results.
 */

// Generation status
export type GenerationStatus =
  | 'idle'
  | 'loading'      // Loading model
  | 'generating'   // Running inference
  | 'completed'
  | 'error'
  | 'cancelled';

/**
 * Progress update during generation
 */
export interface GenerationProgress {
  step: number;
  totalSteps: number;
  percent: number;
  elapsedMs: number;
  estimatedRemainingMs?: number;
  message?: string;
}

/**
 * Result of a successful generation
 */
export interface GenerationResult {
  id: string;
  pipelineId: string;
  outputType: 'image' | 'video' | 'layers';
  urls: string[];           // Output file URLs
  thumbnailUrl?: string;    // For history display
  params: Record<string, unknown>;  // Parameters used
  seed: number;             // Actual seed used
  durationMs: number;       // Total generation time
  timestamp: number;        // Unix timestamp
}

/**
 * Generation error
 */
export interface GenerationError {
  message: string;
  code?: string;
  details?: unknown;
  recoverable: boolean;
}

/**
 * History item for display
 */
export interface HistoryItem {
  id: string;
  pipelineId: string;
  pipelineName: string;
  pipelineColor: string;
  thumbnailUrl: string;
  prompt: string;
  shortPrompt: string;      // First ~50 chars
  keyParams: string;        // e.g., "30 steps · CFG 3.0"
  timestamp: number;
  relativeTime: string;     // e.g., "5m ago"
  params: Record<string, unknown>;
  result: GenerationResult;
}

/**
 * Comparison between two history items
 */
export interface ParameterDiff {
  key: string;
  label: string;
  valueA: unknown;
  valueB: unknown;
}

/**
 * Time estimate for a generation
 */
export interface TimeEstimate {
  estimatedSeconds: number;
  basedOn: 'model' | 'history' | 'default';
  confidence: 'low' | 'medium' | 'high';
}

/**
 * Format relative time for history display
 */
export function formatRelativeTime(timestamp: number): string {
  const now = Date.now();
  const diff = now - timestamp;

  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) return `${days}d ago`;
  if (hours > 0) return `${hours}h ago`;
  if (minutes > 0) return `${minutes}m ago`;
  return 'just now';
}

/**
 * Format duration for display
 */
export function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);

  if (minutes > 0) {
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  }
  return `${seconds}s`;
}

/**
 * Extract key params for history card display
 */
export function extractKeyParams(params: Record<string, unknown>): string {
  const parts: string[] = [];

  // Steps
  const steps = params.steps ?? params.num_inference_steps ?? params.num_steps;
  if (steps !== undefined) {
    parts.push(`${steps} steps`);
  }

  // CFG/Guidance
  const cfg = params.guidance_scale ?? params.guidance;
  if (cfg !== undefined) {
    parts.push(`CFG ${cfg}`);
  }

  // Dimensions
  const width = params.width;
  const height = params.height;
  if (width && height) {
    parts.push(`${width}×${height}`);
  }

  return parts.join(' · ') || 'Default settings';
}

/**
 * Truncate prompt for display
 */
export function truncatePrompt(prompt: string, maxLength = 50): string {
  if (!prompt) return '';
  if (prompt.length <= maxLength) return prompt;
  return prompt.substring(0, maxLength - 3) + '...';
}
