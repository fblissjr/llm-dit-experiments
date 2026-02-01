/**
 * PipelineForm Component
 *
 * Schema-driven form that renders all parameters for a pipeline.
 * Groups parameters by their group field and handles preset loading.
 */

import { useMemo, useCallback } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useAppStore, useFormStore } from '@/stores';
import type { GroupType, ParamSchema, FormValues, ValidationError } from '@/api/types';
import { ParamGroup } from './ParamGroup';
import { ParamControl } from './ParamControl';

// Define group order for consistent rendering
const groupOrder: GroupType[] = [
  'basic',
  'scheduler',
  'advanced',
  'expert',
  'optimization',
  'enhancement',
];

// Empty array/object constants to avoid creating new references
const EMPTY_PRESETS: never[] = [];
const EMPTY_ERRORS: ValidationError[] = [];
const EMPTY_FORM_VALUES: FormValues = {};

export function PipelineForm() {
  const selectedPipelineId = useAppStore((s) => s.selectedPipelineId);

  // Use useShallow for object/array selectors to prevent infinite re-renders
  // In Zustand v5, selectors returning new objects trigger re-renders due to reference inequality
  const pipeline = useAppStore(
    useShallow((s) => (selectedPipelineId ? s.pipelines[selectedPipelineId] : null))
  );
  const presets = useAppStore(
    useShallow((s) => s.presets[selectedPipelineId ?? ''] ?? EMPTY_PRESETS)
  );

  // Function references are stable - no useShallow needed
  const getPipelineColor = useAppStore((s) => s.getPipelineColor);

  // Use useShallow for computed objects that return new references
  const formValues = useFormStore(
    useShallow((s) =>
      selectedPipelineId ? s.getResolvedValues(selectedPipelineId) : EMPTY_FORM_VALUES
    )
  );
  const errors = useFormStore(
    useShallow((s) =>
      selectedPipelineId ? (s.errors[selectedPipelineId] ?? EMPTY_ERRORS) : EMPTY_ERRORS
    )
  );

  const setValue = useFormStore((s) => s.setValue);
  const applyPreset = useFormStore((s) => s.applyPreset);

  // Handle value change
  const handleChange = useCallback(
    (paramId: string) => (value: unknown) => {
      if (!selectedPipelineId) return;
      setValue(selectedPipelineId, paramId, value);
    },
    [selectedPipelineId, setValue]
  );

  // Handle preset selection
  const handlePresetChange = useCallback(
    (presetName: string) => {
      if (!selectedPipelineId || !presetName) return;

      const preset = presets.find((p) => p.name === presetName);
      if (preset) {
        // Apply preset params first
        applyPreset(selectedPipelineId, preset.params);
        // Then set the preset field itself
        setValue(selectedPipelineId, 'preset', presetName);
      }
    },
    [selectedPipelineId, presets, applyPreset, setValue]
  );

  // Group parameters by their group field
  const groupedParams = useMemo(() => {
    if (!pipeline) return new Map<GroupType, ParamSchema[]>();

    const groups = new Map<GroupType, ParamSchema[]>();

    for (const param of pipeline.params) {
      // Special handling for preset - we intercept it
      if (param.id === 'preset') continue;

      const group = param.group || 'basic';
      if (!groups.has(group)) {
        groups.set(group, []);
      }
      groups.get(group)!.push(param);
    }

    return groups;
  }, [pipeline]);

  // Find preset param if it exists
  const presetParam = pipeline?.params.find((p) => p.id === 'preset');

  if (!pipeline || !selectedPipelineId) {
    return (
      <div className="text-gray-400 text-center py-8">
        Select a pipeline to get started
      </div>
    );
  }

  // Set CSS variable for pipeline color
  const pipelineColor = getPipelineColor(selectedPipelineId);

  return (
    <form
      className="space-y-6"
      style={{ '--pipeline-color': pipelineColor } as React.CSSProperties}
      onSubmit={(e) => e.preventDefault()}
    >
      {/* Preset selector - special handling outside groups */}
      {presetParam && presets.length > 0 && (
        <div className="form-control">
          <label className="form-label">{presetParam.label}</label>
          <select
            value={(formValues.preset as string) ?? ''}
            onChange={(e) => handlePresetChange(e.target.value)}
            className="form-select"
          >
            <option value="">Select a preset...</option>
            {presets.map((preset) => (
              <option key={preset.name} value={preset.name}>
                {preset.name} - {preset.description}
              </option>
            ))}
          </select>
          {presetParam.tooltip && (
            <p className="text-xs text-gray-500 mt-1">{presetParam.tooltip}</p>
          )}
        </div>
      )}

      {/* Render groups in order */}
      {groupOrder.map((groupId) => {
        const params = groupedParams.get(groupId);
        if (!params || params.length === 0) return null;

        return (
          <ParamGroup key={groupId} groupId={groupId}>
            {params.map((param) => (
              <ParamControl
                key={param.id}
                param={param}
                value={formValues[param.id]}
                onChange={handleChange(param.id)}
                formValues={formValues}
                errors={errors}
              />
            ))}
          </ParamGroup>
        );
      })}
    </form>
  );
}
