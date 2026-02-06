/**
 * API Response Types
 *
 * Type definitions for all API responses from the FastAPI backend.
 *
 * last updated: 2026-01-25
 */

/**
 * Generation result for image pipelines
 */
export interface ImageGenerationResult {
  /** Path to generated image (relative to server) */
  image_path: string;

  /** Full URL to the image */
  image_url: string;

  /** Seed used for generation */
  seed: number;

  /** Generation time in seconds */
  generation_time: number;

  /** Parameters used for generation */
  params: Record<string, unknown>;
}

/**
 * Generation result for video pipelines (LTX-2)
 */
export interface VideoGenerationResult {
  /** Path to generated video (relative to server) */
  video_path: string;

  /** Full URL to the video */
  video_url: string;

  /** Seed used for generation */
  seed: number;

  /** Generation time in seconds */
  generation_time: number;

  /** Number of frames generated */
  num_frames: number;

  /** Frames per second */
  fps: number;
}

/**
 * Layer decomposition result (Qwen-Image Layered)
 */
export interface LayerDecompositionResult {
  /** Paths to layer images */
  layer_paths: string[];

  /** URLs to layer images */
  layer_urls: string[];

  /** Path to composite image */
  composite_path: string;

  /** Seed used */
  seed: number;

  /** Generation time */
  generation_time: number;
}

/**
 * SSE progress event for streaming generation
 */
export interface ProgressEvent {
  type: "progress";
  step: number;
  total_steps: number;
  percentage: number;
  message?: string;
}

/**
 * SSE complete event for streaming generation
 */
export interface CompleteEvent {
  type: "complete";
  result: VideoGenerationResult;
}

/**
 * SSE error event for streaming generation
 */
export interface ErrorEvent {
  type: "error";
  error: string;
  detail?: string;
}

/**
 * Union type for all SSE events
 */
export type SSEEvent = ProgressEvent | CompleteEvent | ErrorEvent;

/**
 * History item from /api/history
 */
export interface HistoryItem {
  id: string;
  type: "image" | "video" | "layers";
  url: string;
  thumbnail_url?: string;
  prompt: string;
  seed: number;
  timestamp: string;
  params: Record<string, unknown>;
  pipeline: string;
}

/**
 * System status from /api/system/status
 */
export interface SystemStatus {
  /** Server uptime in seconds */
  uptime: number;

  /** VRAM usage in MB */
  vram_used_mb: number;

  /** Total VRAM in MB */
  vram_total_mb: number;

  /** Currently loaded pipeline */
  loaded_pipeline: string | null;

  /** GPU name */
  gpu_name: string;

  /** GPU memory fraction used */
  gpu_memory_fraction: number;
}

/**
 * VRAM status from /api/vram/status
 */
export interface VRAMStatus {
  /** Total VRAM in GB */
  total_vram_gb: number;

  /** Used VRAM in GB */
  used_vram_gb: number;

  /** Free VRAM in GB */
  free_vram_gb: number;

  /** Loaded models */
  loaded_models: {
    zimage: boolean;
    qwenimage: boolean;
    qwenimage_t2i: boolean;
    ltx2: boolean;
    flux2: boolean;
  };
}

/**
 * Error response from API
 */
export interface APIError {
  detail: string;
  status_code?: number;
}

/**
 * Type guard for API errors
 */
export function isAPIError(obj: unknown): obj is APIError {
  return (
    typeof obj === "object" &&
    obj !== null &&
    "detail" in obj &&
    typeof (obj as APIError).detail === "string"
  );
}
