/**
 * API Client - Fetch wrapper with error handling
 *
 * Provides type-safe API calls with consistent error handling.
 * All endpoints are proxied through Vite to the backend at :7860.
 */

import type {
  PipelinesResponse,
  PresetsResponse,
  VRAMStatus,
  GenerationResult,
  FormValues,
  ModelStatusResponse,
  GenerationContext,
  LoRAListResponse,
  ClearCacheResponse,
} from './types';
import { logger } from '@/utils/logger';

const log = logger('API');

class APIError extends Error {
  constructor(
    message: string,
    public status: number,
    public details?: unknown
  ) {
    super(message);
    this.name = 'APIError';
  }
}

async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const response = await fetch(endpoint, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  });

  if (!response.ok) {
    let message = `Request failed: ${response.statusText}`;
    let details: unknown;

    try {
      const data = await response.json();
      message = data.error || data.message || data.detail || message;
      details = data;
    } catch {
      // Response wasn't JSON
    }

    log.error(`${options.method ?? 'GET'} ${endpoint} -> ${response.status}:`, message);
    throw new APIError(message, response.status, details);
  }

  return response.json();
}

/**
 * Fetch all pipeline schemas
 */
export async function fetchPipelines(): Promise<PipelinesResponse> {
  return request<PipelinesResponse>('/api/pipelines');
}

/**
 * Fetch defaults for a pipeline (schema + server config merged)
 */
export async function fetchPipelineDefaults(
  pipelineId: string
): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>(`/api/pipelines/${pipelineId}/defaults`);
}

/**
 * Fetch presets for a pipeline
 */
export async function fetchPresets(
  pipelineId: string,
  variant?: string
): Promise<PresetsResponse> {
  const url = variant
    ? `/api/presets/${pipelineId}?variant=${variant}`
    : `/api/presets/${pipelineId}`;

  return request<PresetsResponse>(url);
}

/**
 * Fetch current VRAM status
 */
export async function fetchVRAMStatus(): Promise<VRAMStatus> {
  return request<VRAMStatus>('/api/vram/status');
}

/**
 * Generate with standard POST request (for image pipelines)
 */
export async function generate(
  endpoint: string,
  params: FormValues
): Promise<GenerationResult> {
  return request<GenerationResult>(endpoint, {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

/**
 * Generate with SSE streaming (for video/progress pipelines)
 *
 * Returns an async generator that yields progress events
 * and finally the result.
 *
 * Pass an AbortSignal to cancel the generation mid-stream.
 * When aborted, the backend stops generating (saves GPU cycles).
 */
export async function* generateStream(
  endpoint: string,
  params: FormValues,
  signal?: AbortSignal
): AsyncGenerator<
  | { type: 'progress'; step: number; total: number; message?: string }
  | { type: 'result'; data: GenerationResult }
  | { type: 'error'; error: string }
> {
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
    signal,
  });

  if (!response.ok) {
    const errMsg = `Request failed: ${response.statusText}`;
    log.error(`SSE POST ${endpoint} -> ${response.status}:`, errMsg);
    yield { type: 'error', error: errMsg };
    return;
  }

  if (!response.body) {
    yield { type: 'error', error: 'No response body' };
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  // Use array-based buffering to avoid O(n^2) string concatenation for large
  // SSE events (e.g., 5MB base64 image delivered in many small chunks).
  const pending: string[] = [];

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const text = decoder.decode(value, { stream: true });

      // Fast path: no newlines means incomplete line, just accumulate
      if (!text.includes('\n')) {
        pending.push(text);
        continue;
      }

      // Join accumulated chunks + current text, then split into lines
      pending.push(text);
      const combined = pending.join('');
      pending.length = 0;

      const lines = combined.split('\n');
      const remainder = lines.pop() ?? '';
      if (remainder) pending.push(remainder);

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') continue;

          try {
            const parsed = JSON.parse(data);
            // Server sends total_steps, but we normalize to total
            const total = parsed.total ?? parsed.total_steps;

            if (parsed.step !== undefined && total !== undefined) {
              yield {
                type: 'progress',
                step: parsed.step,
                total: total,
                message: parsed.message,
              };
            } else if (parsed.output_path || parsed.urls || parsed.type === 'complete') {
              yield { type: 'result', data: parsed as GenerationResult };
            } else if (parsed.error || parsed.type === 'error') {
              log.error('SSE server error:', parsed.error || parsed.message);
              yield { type: 'error', error: parsed.error || parsed.message };
            }
          } catch {
            // Skip malformed JSON
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Model Management APIs
 */

/**
 * Get model status for a pipeline
 */
export async function fetchModelStatus(pipelineId: string): Promise<ModelStatusResponse> {
  return request<ModelStatusResponse>(`/api/models/${pipelineId}/status`);
}

/**
 * Load a model for a pipeline.
 * The load endpoint returns { success, message } -- not the ModelStatusResponse shape.
 * After a successful load, we fetch the actual model status for consistent state.
 *
 * Not called from UI (models auto-load on generate). Kept as a library function
 * for power users via curl/scripts and the /api/models/{id}/load endpoint.
 */
export async function loadModel(pipelineId: string): Promise<ModelStatusResponse> {
  await request<{ success: boolean; message: string }>(
    `/api/models/${pipelineId}/load`,
    { method: 'POST' },
  );

  return fetchModelStatus(pipelineId);
}

/**
 * Unload all loaded models to free VRAM.
 */
export async function unloadAllModels(): Promise<{ success: boolean; message: string }> {
  return request<{ success: boolean; message: string }>(
    '/api/models/unload-all',
    { method: 'POST' },
  );
}

/**
 * Unload a model for a pipeline.
 * Same pattern as loadModel -- fetch status after the action completes.
 */
export async function unloadModel(pipelineId: string): Promise<ModelStatusResponse> {
  await request<{ success: boolean }>(
    `/api/models/${pipelineId}/unload`,
    { method: 'POST' },
  );

  return fetchModelStatus(pipelineId);
}

/**
 * LoRA Management APIs
 */

/**
 * Fetch available LoRA files, optionally filtered by pipeline.
 */
export async function fetchAvailableLoras(pipelineId?: string): Promise<LoRAListResponse> {
  const url = pipelineId ? `/api/loras/${pipelineId}` : '/api/loras';
  return request<LoRAListResponse>(url);
}

/**
 * Generation Context & Server Management APIs
 */

/**
 * Fetch composite generation context (model variant, LoRA, VRAM, etc.)
 */
export async function fetchGenerationContext(): Promise<GenerationContext> {
  return request<GenerationContext>('/api/context');
}

/**
 * Restart the server
 */
export async function restartServer(reason?: string): Promise<{ success: boolean; message: string }> {
  return request<{ success: boolean; message: string }>(
    '/api/server/restart',
    {
      method: 'POST',
      body: JSON.stringify({ reason: reason ?? 'user_request' }),
    },
  );
}

/**
 * Clear CUDA cache and garbage collect
 */
export async function clearCache(): Promise<ClearCacheResponse> {
  return request<ClearCacheResponse>(
    '/api/system/clear-cache',
    { method: 'POST' },
  );
}

/**
 * Upsample a FLUX.2 prompt via the configured LLM API.
 * Returns the upsampled prompt string. Passes reference images when provided
 * so the vision model can write context-aware editing instructions.
 */
export async function upsamplePrompt(
  prompt: string,
  referenceImages?: string[]
): Promise<string> {
  const result = await request<{ prompt: string }>('/api/flux2/upsample-prompt', {
    method: 'POST',
    body: JSON.stringify({
      prompt,
      reference_images: referenceImages ?? null,
    }),
  });
  return result.prompt;
}

export { APIError };
