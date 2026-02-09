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
  // Value-dependent defaults: when trigger param changes, this param's default updates
  // Format: { trigger_param_id: { trigger_value: new_default } }
  dependent_defaults?: Record<string, Record<string, unknown>>;
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
  usedMb: number;
  totalMb: number;
  freeMb: number;
  utilizationPercent: number;
  breakdown: {
    label: string;
    sizeMb: number;
    color: string;
  }[];
}

/**
 * Config metadata from /api/models/{pipeline_id}/status
 */
export interface ConfigTag {
  key: string;
  label: string;
  color: string;
}

export interface ConfigWarning {
  severity: 'error' | 'warning';
  message: string;
}

/**
 * Model status from /api/models/{pipeline_id}/status
 */
export type ModelStatus = 'loaded' | 'unloaded' | 'loading' | 'error';

export interface ModelStatusResponse {
  pipelineId?: string;
  status: ModelStatus;
  components?: { name: string; sizeMb: number; device: string }[];
  totalVramMb?: number;
  vramMb?: number;
  modelVariant?: string | null;
  displayName?: string | null;
  loras?: { name: string; scale: number; layersUpdated: number }[];
  loraSummary?: string | null;
  configTags?: ConfigTag[];
  configWarnings?: ConfigWarning[];
  // Frontend-only: set locally when a load/unload operation fails
  error?: string;
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
  fullImageUrl?: string; // Full resolution URL (only available in current session for base64 images)
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
  defaultPreset: string;
}

/**
 * LoRA file on disk from /api/loras
 */
export interface LoRAFile {
  path: string;
  name: string;
  directory: string;
  sizeMb: number;
}

/**
 * LoRA list response from /api/loras
 */
export interface LoRAListResponse {
  loras: LoRAFile[];
  directories: string[];
  count: number;
  pipelineId?: string;
}

/**
 * Clear cache response from /api/system/clear-cache
 */
export interface ClearCacheResponse {
  success: boolean;
  freedGb: number;
  message: string;
}

/**
 * LoRA fusion info from /api/context
 */
export interface LoRAInfo {
  name: string;
  path: string;
  scale: number;
  layersUpdated: number;
}

/**
 * Composite generation context from /api/context
 *
 * Aggregates model variant, LoRA state, VRAM, quantization,
 * compile, and session state into a single response.
 */
export interface GenerationContext {
  uptimeSeconds: number | null;
  profile: string;
  activePipeline: string | null;
  pipelineDisplayName: string | null;
  modelVariant: string | null;
  loras: LoRAInfo[];
  loraSummary: string | null;
  quantization: Record<string, string>;
  compileEnabled: boolean;
  compileMode: string | null;
  blockOffload: boolean;
  vramUsedGb: number | null;
  vramTotalGb: number | null;
  vramPercent: number | null;
  pendingRestartFields: string[];
  sessionModifiedFields: string[];
  fmttCached: boolean;
  historyCount: number;
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
