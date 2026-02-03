/**
 * API Types - Mirrors Python schemas for type safety
 *
 * These types match the backend dataclasses in:
 * - src/llm_dit/pipelines/schemas/__init__.py (ParamSchema, PipelineSchema)
 * - src/llm_dit/presets/schema.py (GenerationPreset)
 */

// Param type - determines which control renders
export type ParamType = 'textarea' | 'slider' | 'number' | 'checkbox' | 'select' | 'image' | 'color' | 'lora_list';

// Output type - affects result display
export type OutputType = 'image' | 'video' | 'layers';

// Param group - for progressive disclosure
export type GroupType = 'basic' | 'advanced' | 'expert' | 'scheduler' | 'optimization' | 'enhancement';

/**
 * ParamSchema - Describes a single UI control
 */
export interface ParamSchema {
  id: string;
  type: ParamType;
  label: string;
  default?: unknown;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  options_endpoint?: string;
  group: GroupType;
  tooltip?: string;
  conditional?: Record<string, unknown>;
  placeholder?: string;
  rows?: number;
  required?: boolean;
  max_count?: number;
  // LoRA-specific constraints
  scale_min?: number;
  scale_max?: number;
}

/**
 * PipelineSchema - Complete pipeline definition
 */
export interface PipelineSchema {
  id: string;
  name: string;
  description: string;
  output_type: OutputType;
  color: string;
  icon?: string;
  params: ParamSchema[];
  supports_history: boolean;
  supports_img2img: boolean;
  supports_reference_images: boolean;
  supports_streaming: boolean;
  endpoint: string;
  category: string;
}

/**
 * GenerationPreset - Reusable generation configuration
 */
export interface GenerationPreset {
  name: string;
  description: string;
  category: string;
  pipelines: string[];
  variant: string | null;
  params: Record<string, unknown>;
}

/**
 * VRAM Status from /api/vram/status
 */
export interface VRAMStatus {
  usedMB: number;
  totalMB: number;
  freeMB: number;
  utilizationPercent: number;
  breakdown: {
    label: string;
    sizeMB: number;
    color: string;
  }[];
}

/**
 * Generation progress update (SSE)
 */
export interface GenerationProgress {
  step: number;
  total: number;
  percent?: number;
  message?: string;
}

/**
 * Generation result
 */
export interface GenerationResult {
  id: string;
  pipelineId: string;
  outputType: OutputType;
  urls: string[];
  thumbnailUrl?: string;
  params: Record<string, unknown>;
  seed: number;
  durationMs: number;
  timestamp: number;
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
  shortPrompt: string;
  keyParams: string;
  timestamp: number;
  params: Record<string, unknown>;
  result: GenerationResult;
}

/**
 * API response from /api/pipelines
 */
export interface PipelinesResponse {
  pipelines: Record<string, PipelineSchema>;
  defaults: Record<string, unknown>;
  loaded_pipeline: string | null;
}

/**
 * API response from /api/presets/{pipeline_id}
 */
export interface PresetsResponse {
  presets: GenerationPreset[];
}

/**
 * Form values - generic key-value for any pipeline
 */
export type FormValues = Record<string, unknown>;

/**
 * Validation error
 */
export interface ValidationError {
  paramId: string;
  message: string;
}
