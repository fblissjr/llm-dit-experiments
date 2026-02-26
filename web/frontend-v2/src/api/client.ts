/**
 * API Client - Fetch wrapper with error handling
 *
 * Provides type-safe API calls with consistent error handling.
 * All endpoints are proxied through Vite to the backend at :7860.
 */

import type {
  PipelinesResponse,
  PipelineSchema,
  GenerationPreset,
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
 * Fetch a single pipeline schema
 */
export async function fetchPipeline(pipelineId: string): Promise<PipelineSchema> {
  return request<PipelineSchema>(`/api/pipelines/${pipelineId}`);
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
 * Fetch a specific preset by name
 */
export async function fetchPresetByName(name: string): Promise<GenerationPreset> {
  return request<GenerationPreset>(`/api/presets/preset/${name}`);
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
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

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
 */
export async function loadModel(pipelineId: string): Promise<ModelStatusResponse> {
  await request<{ success: boolean; message: string }>(
    `/api/models/${pipelineId}/load`,
    { method: 'POST' },
  );

  return fetchModelStatus(pipelineId);
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
 * Fetch all available LoRA files from configured directories
 */
export async function fetchAvailableLoras(): Promise<LoRAListResponse> {
  return request<LoRAListResponse>('/api/loras');
}

/**
 * Fetch LoRA files for a specific pipeline
 */
export async function fetchLorasForPipeline(pipelineId: string): Promise<LoRAListResponse> {
  return request<LoRAListResponse>(`/api/loras/${pipelineId}`);
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

export { APIError };
