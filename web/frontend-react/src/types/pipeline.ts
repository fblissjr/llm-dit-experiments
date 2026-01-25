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
  group: GroupType;
  tooltip?: string;
  conditional?: Record<string, unknown>;
  placeholder?: string;
  rows?: number;
  required?: boolean;
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
