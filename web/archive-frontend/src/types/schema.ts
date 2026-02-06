/**
 * Pipeline Schema Types
 *
 * These types mirror the Python dataclasses in src/llm_dit/pipelines/schemas/__init__.py.
 * Keep in sync with the Python definitions - any changes there should be reflected here.
 *
 * last updated: 2026-01-25
 */

/**
 * Parameter control types - maps to form element types
 */
export type ParamType =
  | "textarea" // Multi-line text input (prompts)
  | "slider" // Range slider with min/max/step
  | "number" // Numeric input
  | "checkbox" // Boolean toggle
  | "select" // Dropdown with options
  | "image" // Image upload/display
  | "color"; // Color picker (future)

/**
 * Output types - affects result display component
 */
export type OutputType = "image" | "video" | "layers";

/**
 * Parameter groups for progressive disclosure
 */
export type GroupType =
  | "basic" // Always visible
  | "advanced" // Collapsible, for power users
  | "expert" // Hidden by default, for debugging
  | "scheduler" // Scheduler-specific params
  | "optimization" // Performance tuning
  | "enhancement"; // Quality enhancements

/**
 * Schema for a single form parameter
 *
 * Maps to a specific Web Component for rendering.
 * The `conditional` field enables dynamic show/hide based on other params.
 */
export interface ParamSchema {
  /** Maps to API field name (e.g., "guidance_scale") */
  id: string;

  /** Control type - determines which component renders this param */
  type: ParamType;

  /** Human-readable label shown in UI */
  label: string;

  /** Default value for the control */
  default?: unknown;

  /** Minimum value for slider/number inputs */
  min?: number;

  /** Maximum value for slider/number inputs */
  max?: number;

  /** Step increment for slider/number inputs */
  step?: number;

  /** List of valid options for select inputs */
  options?: string[];

  /** Grouping for progressive disclosure */
  group: GroupType;

  /** Help text shown on hover/focus */
  tooltip?: string;

  /**
   * Show only when another field matches condition.
   * Example: { "dype_enabled": true } - show only when dype_enabled is checked
   */
  conditional?: Record<string, unknown>;

  /** Placeholder text for textarea/text inputs */
  placeholder?: string;

  /** Number of rows for textarea inputs */
  rows?: number;

  /** Whether the field is required for generation */
  required?: boolean;
}

/**
 * Complete schema for a pipeline's UI
 *
 * The frontend fetches these at startup and uses them to build
 * forms dynamically. No pipeline-specific code needed in the frontend.
 */
export interface PipelineSchema {
  /** Unique identifier matching SUPPORTED_MODEL_TYPES */
  id: string;

  /** Human-readable display name */
  name: string;

  /** Brief description of the pipeline's purpose */
  description: string;

  /** What the pipeline produces - affects result display */
  output_type: OutputType;

  /** Tailwind color class for theming (e.g., "blue", "purple") */
  color: string;

  /** Icon identifier or emoji */
  icon?: string;

  /** List of form parameters */
  params: ParamSchema[];

  /** Whether to show generation history */
  supports_history: boolean;

  /** Whether pipeline accepts input images */
  supports_img2img: boolean;

  /** Whether pipeline uses reference images (FLUX.2) */
  supports_reference_images: boolean;

  /** Whether generation uses SSE streaming (LTX-2) */
  supports_streaming: boolean;

  /** API endpoint for generation */
  endpoint: string;

  /** Category for pipeline tabs (e.g., "image", "video") */
  category: string;
}

/**
 * Response from /api/pipelines endpoint
 */
export interface PipelinesResponse {
  /** Dict of pipeline_id -> PipelineSchema */
  pipelines: Record<string, PipelineSchema>;

  /** Current RuntimeConfig values (if loaded) */
  defaults: Record<string, unknown>;

  /** Currently loaded pipeline type (if any) */
  loaded_pipeline: string | null;
}

/**
 * Form values - maps param IDs to their current values
 */
export type FormValues = Record<string, unknown>;

/**
 * Validation result for a single parameter
 */
export interface ValidationResult {
  valid: boolean;
  message?: string;
}

/**
 * Helper type guard to check if a value is a valid ParamType
 */
export function isParamType(value: string): value is ParamType {
  return [
    "textarea",
    "slider",
    "number",
    "checkbox",
    "select",
    "image",
    "color",
  ].includes(value);
}

/**
 * Helper to get params by group from a schema
 */
export function getParamsByGroup(
  schema: PipelineSchema,
  group: GroupType
): ParamSchema[] {
  return schema.params.filter((p) => p.group === group);
}

/**
 * Helper to get default values from a schema
 */
export function getDefaultValues(schema: PipelineSchema): FormValues {
  const defaults: FormValues = {};
  for (const param of schema.params) {
    if (param.default !== undefined) {
      defaults[param.id] = param.default;
    }
  }
  return defaults;
}

/**
 * Check if a param should be visible based on conditional rules
 */
export function isParamVisible(
  param: ParamSchema,
  formValues: FormValues
): boolean {
  if (!param.conditional) {
    return true;
  }

  for (const [key, expectedValue] of Object.entries(param.conditional)) {
    const actualValue = formValues[key];
    if (actualValue !== expectedValue) {
      return false;
    }
  }

  return true;
}
