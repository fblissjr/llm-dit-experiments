/**
 * Pipeline Types - TypeScript mirrors of Python schema dataclasses
 *
 * These types are consumed by the frontend to dynamically render forms.
 * They match the JSON output of PipelineSchema.to_dict() in Python.
 */

// Control types supported by the form builder
export type ParamType =
  | 'textarea'
  | 'slider'
  | 'number'
  | 'checkbox'
  | 'select'
  | 'image'
  | 'color';

// What the pipeline produces
export type OutputType = 'image' | 'video' | 'layers';

// Progressive disclosure groups
export type GroupType =
  | 'basic'
  | 'scheduler'
  | 'optimization'
  | 'enhancement'
  | 'advanced'
  | 'expert';

// Pipeline accent colors
export type PipelineColor =
  | 'blue'
  | 'purple'
  | 'orange'
  | 'teal'
  | 'green'
  | 'pink';

/**
 * Static Tailwind class maps for pipeline colors.
 * IMPORTANT: Must be static strings for Tailwind JIT to detect them.
 * Dynamic interpolation like `bg-${color}-500` won't work in production.
 */
export const PIPELINE_COLOR_CLASSES = {
  border: {
    blue: 'border-blue-500',
    purple: 'border-purple-500',
    orange: 'border-orange-500',
    teal: 'border-teal-500',
    green: 'border-green-500',
    pink: 'border-pink-500',
  } as const satisfies Record<PipelineColor, string>,

  bg: {
    blue: 'bg-blue-500',
    purple: 'bg-purple-500',
    orange: 'bg-orange-500',
    teal: 'bg-teal-500',
    green: 'bg-green-500',
    pink: 'bg-pink-500',
  } as const satisfies Record<PipelineColor, string>,

  bgSubtle: {
    blue: 'bg-blue-500/80',
    purple: 'bg-purple-500/80',
    orange: 'bg-orange-500/80',
    teal: 'bg-teal-500/80',
    green: 'bg-green-500/80',
    pink: 'bg-pink-500/80',
  } as const satisfies Record<PipelineColor, string>,

  text: {
    blue: 'text-blue-500',
    purple: 'text-purple-500',
    orange: 'text-orange-500',
    teal: 'text-teal-500',
    green: 'text-green-500',
    pink: 'text-pink-500',
  } as const satisfies Record<PipelineColor, string>,

  borderAndText: {
    blue: 'border-blue-500 text-blue-500',
    purple: 'border-purple-500 text-purple-500',
    orange: 'border-orange-500 text-orange-500',
    teal: 'border-teal-500 text-teal-500',
    green: 'border-green-500 text-green-500',
    pink: 'border-pink-500 text-pink-500',
  } as const satisfies Record<PipelineColor, string>,

  // Full button styles (bg + hover + focus ring)
  button: {
    blue: 'bg-blue-600 hover:bg-blue-500 focus:ring-blue-500',
    purple: 'bg-purple-600 hover:bg-purple-500 focus:ring-purple-500',
    orange: 'bg-orange-600 hover:bg-orange-500 focus:ring-orange-500',
    teal: 'bg-teal-600 hover:bg-teal-500 focus:ring-teal-500',
    green: 'bg-green-600 hover:bg-green-500 focus:ring-green-500',
    pink: 'bg-pink-600 hover:bg-pink-500 focus:ring-pink-500',
  } as const satisfies Record<PipelineColor, string>,
};

/**
 * Schema for a single parameter/control
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
  options_endpoint?: string;  // API endpoint for dynamic options (presets)
  group: GroupType;
  tooltip?: string;
  conditional?: Record<string, unknown>;
  placeholder?: string;
  rows?: number;
  required?: boolean;
  max_count?: number;  // Maximum number of images (for image type controls)
}

/**
 * Generation preset for a pipeline
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
 * Complete pipeline schema
 */
export interface PipelineSchema {
  id: string;
  name: string;
  description: string;
  output_type: OutputType;
  color: PipelineColor;
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
 * Form values for a pipeline - keyed by param ID
 */
export type FormValues = Record<string, unknown>;

/**
 * Group metadata for rendering
 */
export interface ParamGroup {
  id: GroupType;
  label: string;
  params: ParamSchema[];
  defaultExpanded: boolean;
}

/**
 * Map groups to human-readable labels
 */
export const GROUP_LABELS: Record<GroupType, string> = {
  basic: 'Basic',
  scheduler: 'Scheduler',
  optimization: 'Optimization',
  enhancement: 'Enhancement',
  advanced: 'Advanced',
  expert: 'Expert',
};

/**
 * Default expansion state for groups
 */
export const GROUP_DEFAULT_EXPANDED: Record<GroupType, boolean> = {
  basic: true,
  scheduler: false,
  optimization: false,
  enhancement: false,
  advanced: false,
  expert: false,
};

/**
 * Check if a parameter should be visible based on conditional
 */
export function isParamVisible(
  param: ParamSchema,
  values: FormValues
): boolean {
  if (!param.conditional) return true;
  return Object.entries(param.conditional).every(
    ([key, val]) => values[key] === val
  );
}

/**
 * Group parameters by their group type
 */
export function groupParams(params: ParamSchema[]): ParamGroup[] {
  const groups: Map<GroupType, ParamSchema[]> = new Map();

  // Initialize all groups to maintain order
  const groupOrder: GroupType[] = [
    'basic',
    'scheduler',
    'optimization',
    'enhancement',
    'advanced',
    'expert',
  ];
  groupOrder.forEach((g) => groups.set(g, []));

  // Distribute params to groups
  params.forEach((param) => {
    const list = groups.get(param.group) || [];
    list.push(param);
    groups.set(param.group, list);
  });

  // Filter out empty groups and build result
  return groupOrder
    .filter((g) => (groups.get(g)?.length ?? 0) > 0)
    .map((g) => ({
      id: g,
      label: GROUP_LABELS[g],
      params: groups.get(g)!,
      defaultExpanded: GROUP_DEFAULT_EXPANDED[g],
    }));
}
