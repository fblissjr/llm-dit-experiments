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
import { PresetBrowser } from '@/components/preset';

// FLUX.2 distilled models have fixed num_steps and guidance.
// Distilled = all variants without "base" in the name.
const FLUX2_FIXED_PARAMS = new Set(['num_steps', 'guidance']);

function isFixedParam(pipelineId: string, paramId: string, formValues: FormValues): boolean {
  if (pipelineId !== 'flux2') return false;
  if (!FLUX2_FIXED_PARAMS.has(paramId)) return false;
  const modelName = String(formValues.model_name ?? '');
  // Base models allow overriding; distilled models do not
  return !modelName.includes('base');
}

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
const EMPTY_ERRORS: ValidationError[] = [];
const EMPTY_FORM_VALUES: FormValues = {};

export function PipelineForm() {
  const selectedPipelineId = useAppStore((s) => s.selectedPipelineId);

  // Use useShallow for object/array selectors to prevent infinite re-renders
  // In Zustand v5, selectors returning new objects trigger re-renders due to reference inequality
  const pipeline = useAppStore(
    useShallow((s) => (selectedPipelineId ? s.pipelines[selectedPipelineId] : null))
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
  const applyDependentDefaults = useFormStore((s) => s.applyDependentDefaults);

  // Handle value change with dimension_preset <-> width/height sync.
  // Reads both formValues and pipeline from store directly (getState) to
  // avoid closing over reactive selectors -- keeps deps minimal so this
  // callback reference stays stable across renders.
  const handleChange = useCallback(
    (paramId: string, value: unknown) => {
      if (!selectedPipelineId) return;

      // When dimension_preset changes, parse "WIDTHxHEIGHT" and update width/height
      if (paramId === 'dimension_preset' && typeof value === 'string') {
        const match = value.match(/^(\d+)x(\d+)$/);
        if (match) {
          const w = parseInt(match[1], 10);
          const h = parseInt(match[2], 10);
          setValue(selectedPipelineId, 'width', w);
          setValue(selectedPipelineId, 'height', h);
        }
      }

      // When width or height changes manually, clear preset to show it's custom
      if (paramId === 'width' || paramId === 'height') {
        const currentValues = useFormStore.getState()
          .getResolvedValues(selectedPipelineId);
        const currentPreset = currentValues.dimension_preset;
        if (currentPreset && currentPreset !== 'Custom') {
          const presetStr = String(currentPreset);
          const match = presetStr.match(/^(\d+)x(\d+)$/);
          if (match) {
            const presetW = parseInt(match[1], 10);
            const presetH = parseInt(match[2], 10);
            const newW = paramId === 'width' ? Number(value) : Number(currentValues.width ?? 1024);
            const newH = paramId === 'height' ? Number(value) : Number(currentValues.height ?? 1024);
            if (newW !== presetW || newH !== presetH) {
              setValue(selectedPipelineId, 'dimension_preset', 'Custom');
            }
          }
        }
      }

      setValue(selectedPipelineId, paramId, value);

      // Check if this param triggers dependent defaults on other params.
      // Read pipeline from store (getState) instead of closing over the
      // reactive selector, so this callback doesn't depend on `pipeline`.
      const currentPipeline = useAppStore.getState().pipelines[selectedPipelineId];
      if (currentPipeline) {
        const hasDependents = currentPipeline.params.some(
          (p) => p.dependent_defaults?.[paramId]
        );
        if (hasDependents) {
          applyDependentDefaults(selectedPipelineId, paramId, value);
        }
      }
    },
    [selectedPipelineId, setValue, applyDependentDefaults]
  );

  // Memoize per-param onChange callbacks so each ParamControl gets a stable
  // reference. Recomputed only when pipeline params change or handleChange
  // changes (which only depends on selectedPipelineId + action refs).
  const paramCallbacks = useMemo(() => {
    if (!pipeline) return new Map<string, (value: unknown) => void>();
    const map = new Map<string, (value: unknown) => void>();
    for (const param of pipeline.params) {
      map.set(param.id, (value: unknown) => handleChange(param.id, value));
    }
    return map;
  }, [pipeline, handleChange]);

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
      {/* Preset browser - visual card system */}
      <PresetBrowser pipelineId={selectedPipelineId} />

      {/* Render groups in order */}
      {groupOrder.map((groupId) => {
        const params = groupedParams.get(groupId);
        if (!params || params.length === 0) return null;

        return (
          <ParamGroup key={groupId} groupId={groupId}>
            {params.map((param) => {
              // Disable fixed params for distilled FLUX.2 models
              const isFixed = isFixedParam(selectedPipelineId, param.id, formValues);
              return (
                <div key={param.id} className="relative">
                  <ParamControl
                    param={isFixed ? { ...param, tooltip: `Fixed for distilled models (${param.tooltip ?? ''})` } : param}
                    value={formValues[param.id]}
                    onChange={paramCallbacks.get(param.id) ?? (() => {})}
                    formValues={formValues}
                    errors={errors}
                    disabled={isFixed}
                  />
                  {isFixed && (
                    <div className="text-xs text-amber-500/70 mt-1 ml-1">
                      Fixed for distilled models
                    </div>
                  )}
                </div>
              );
            })}
          </ParamGroup>
        );
      })}
    </form>
  );
}
