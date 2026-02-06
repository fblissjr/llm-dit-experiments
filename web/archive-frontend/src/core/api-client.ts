/**
 * Typed API Client
 *
 * Handles all communication with the FastAPI backend.
 * Provides typed methods for each endpoint.
 *
 * last updated: 2026-01-25
 */

import type {
  PipelinesResponse,
  PipelineSchema,
  FormValues,
  ImageGenerationResult,
  VideoGenerationResult,
  LayerDecompositionResult,
  SSEEvent,
  HistoryItem,
  SystemStatus,
  VRAMStatus,
} from "@/types/index.ts";

/**
 * API client configuration
 */
interface APIClientConfig {
  baseUrl: string;
  timeout: number;
}

const defaultConfig: APIClientConfig = {
  baseUrl: "", // Empty for same-origin requests (uses Vite proxy in dev)
  timeout: 120000, // 2 minutes for long generations
};

/**
 * Make a typed fetch request
 */
async function fetchJSON<T>(
  url: string,
  options: RequestInit = {},
  config: APIClientConfig = defaultConfig
): Promise<T> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), config.timeout);

  try {
    const response = await fetch(`${config.baseUrl}${url}`, {
      ...options,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        detail: `HTTP ${response.status}: ${response.statusText}`,
      }));
      throw new APIClientError(
        error.detail || "Unknown error",
        response.status
      );
    }

    return (await response.json()) as T;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof APIClientError) {
      throw error;
    }
    if (error instanceof Error && error.name === "AbortError") {
      throw new APIClientError("Request timeout", 408);
    }
    throw new APIClientError(
      error instanceof Error ? error.message : "Unknown error",
      0
    );
  }
}

/**
 * Custom error class for API errors
 */
export class APIClientError extends Error {
  constructor(
    message: string,
    public statusCode: number
  ) {
    super(message);
    this.name = "APIClientError";
  }
}

// =============================================================================
// Pipeline Schema API
// =============================================================================

/**
 * Fetch all pipeline schemas
 */
export async function fetchPipelines(): Promise<PipelinesResponse> {
  return fetchJSON<PipelinesResponse>("/api/pipelines");
}

/**
 * Fetch schema for a specific pipeline
 */
export async function fetchPipelineSchema(
  pipelineId: string
): Promise<PipelineSchema> {
  return fetchJSON<PipelineSchema>(`/api/pipelines/${pipelineId}`);
}

/**
 * Fetch defaults for a specific pipeline
 */
export async function fetchPipelineDefaults(
  pipelineId: string
): Promise<FormValues> {
  return fetchJSON<FormValues>(`/api/pipelines/${pipelineId}/defaults`);
}

// =============================================================================
// Generation API
// =============================================================================

/**
 * Generate using any pipeline (non-streaming)
 */
export async function generate(
  endpoint: string,
  params: FormValues
): Promise<ImageGenerationResult | LayerDecompositionResult> {
  return fetchJSON(endpoint, {
    method: "POST",
    body: JSON.stringify(params),
  });
}

/**
 * Generate video with streaming progress (LTX-2)
 */
export async function generateVideoStream(
  endpoint: string,
  params: FormValues,
  onProgress: (event: SSEEvent) => void
): Promise<VideoGenerationResult> {
  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(params),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({
      detail: `HTTP ${response.status}`,
    }));
    throw new APIClientError(error.detail, response.status);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new APIClientError("No response body", 500);
  }

  const decoder = new TextDecoder();
  let buffer = "";
  let result: VideoGenerationResult | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const event = JSON.parse(line.slice(6)) as SSEEvent;
          onProgress(event);

          if (event.type === "complete") {
            result = event.result;
          } else if (event.type === "error") {
            throw new APIClientError(event.error, 500);
          }
        } catch (e) {
          if (e instanceof APIClientError) throw e;
          console.warn("Failed to parse SSE event:", line);
        }
      }
    }
  }

  if (!result) {
    throw new APIClientError("No result received from stream", 500);
  }

  return result;
}

// =============================================================================
// Image Upload
// =============================================================================

/**
 * Upload an image and get a URL back
 */
export async function uploadImage(file: File): Promise<string> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("/api/upload", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Upload failed" }));
    throw new APIClientError(error.detail, response.status);
  }

  const result = (await response.json()) as { url: string };
  return result.url;
}

/**
 * Convert a data URL (base64) to a File object
 */
export function dataURLtoFile(dataUrl: string, filename: string): File {
  const [header, data] = dataUrl.split(",");
  const mime = header?.match(/:(.*?);/)?.[1] ?? "image/png";
  const bytes = atob(data ?? "");
  const buffer = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) {
    buffer[i] = bytes.charCodeAt(i);
  }
  return new File([buffer], filename, { type: mime });
}

// =============================================================================
// History API
// =============================================================================

/**
 * Fetch generation history
 */
export async function fetchHistory(): Promise<HistoryItem[]> {
  return fetchJSON<HistoryItem[]>("/api/history");
}

// =============================================================================
// System API
// =============================================================================

/**
 * Fetch system status
 */
export async function fetchSystemStatus(): Promise<SystemStatus> {
  return fetchJSON<SystemStatus>("/api/system/status");
}

/**
 * Fetch VRAM status
 */
export async function fetchVRAMStatus(): Promise<VRAMStatus> {
  return fetchJSON<VRAMStatus>("/api/vram/status");
}

/**
 * Load a pipeline on the server
 */
export async function loadPipeline(pipelineId: string): Promise<void> {
  const endpointMap: Record<string, string> = {
    zimage: "/api/vram/load-zimage",
    "qwenimage-layered": "/api/vram/load-qwen-image",
    "qwenimage-t2i": "/api/vram/load-qwen-image-t2i",
    "qwenimage-edit": "/api/vram/load-qwen-image",
    ltx2: "/api/vram/load-ltx2",
    flux2: "/api/vram/load-flux2",
  };

  const endpoint = endpointMap[pipelineId];
  if (!endpoint) {
    throw new APIClientError(`Unknown pipeline: ${pipelineId}`, 400);
  }

  await fetchJSON(endpoint, { method: "POST" });
}

/**
 * Unload a pipeline from the server
 */
export async function unloadPipeline(pipelineId: string): Promise<void> {
  const endpointMap: Record<string, string> = {
    zimage: "/api/vram/unload-zimage",
    "qwenimage-layered": "/api/vram/unload-qwen-image",
    "qwenimage-t2i": "/api/vram/unload-qwen-image-t2i",
    "qwenimage-edit": "/api/vram/unload-qwen-image",
    ltx2: "/api/vram/unload-ltx2",
    flux2: "/api/vram/unload-flux2",
  };

  const endpoint = endpointMap[pipelineId];
  if (!endpoint) {
    throw new APIClientError(`Unknown pipeline: ${pipelineId}`, 400);
  }

  await fetchJSON(endpoint, { method: "POST" });
}

// =============================================================================
// Utility
// =============================================================================

/**
 * Build a full URL for a media file
 */
export function getMediaUrl(path: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  return `${defaultConfig.baseUrl}${path}`;
}
