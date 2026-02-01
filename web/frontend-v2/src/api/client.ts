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
  VRAMStatus,
  GenerationResult,
  FormValues,
} from './types';

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
): Promise<GenerationPreset[]> {
  const url = variant
    ? `/api/presets/${pipelineId}?variant=${variant}`
    : `/api/presets/${pipelineId}`;

  const response = await request<{ presets: GenerationPreset[] }>(url);
  return response.presets;
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
  const data = await request<Record<string, unknown>>('/api/vram/status');

  return {
    usedMB: (data.used_mb ?? data.usedMB ?? 0) as number,
    totalMB: (data.total_mb ?? data.totalMB ?? 24576) as number,
    freeMB: (data.free_mb ?? data.freeMB ?? 24576) as number,
    utilizationPercent: (data.utilization_percent ?? data.utilizationPercent ?? 0) as number,
    breakdown: (data.breakdown ?? []) as VRAMStatus['breakdown'],
  };
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
 */
export async function* generateStream(
  endpoint: string,
  params: FormValues
): AsyncGenerator<
  | { type: 'progress'; step: number; total: number; message?: string }
  | { type: 'result'; data: GenerationResult }
  | { type: 'error'; error: string }
> {
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });

  if (!response.ok) {
    yield { type: 'error', error: `Request failed: ${response.statusText}` };
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
 * Upload an image and get a URL back
 */
export async function uploadImage(file: File): Promise<string> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('/api/upload', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new APIError('Upload failed', response.status);
  }

  const data = await response.json();
  return data.url || data.path;
}

export { APIError };
