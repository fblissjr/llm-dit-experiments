/**
 * Pipeline Form
 *
 * Schema-driven form that renders controls based on the selected pipeline's schema.
 * Groups parameters and handles form state.
 *
 * NOTE: This component uses shallow selectors to get stable references from the store.
 * The formValues object is passed directly from the store, and individual param values
 * are resolved with defaults at the ParamControl level to avoid recreating objects
 * on every render (which would break controlled inputs like sliders).
 *
 * PRESETS: When a preset is selected, its params are merged into form values.
 * User edits after preset selection override the preset values.
 */

import { useEffect, useCallback, useMemo } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { usePipelineStore } from '@/stores/pipelineStore';
import { useGenerationStore, selectFormValues } from '@/stores/generationStore';
import { ParamGroup } from './ParamGroup';
import { ParamControl } from './ParamControl';
import { GenerateButton } from '../generation/GenerateButton';
import { groupParams } from '@/types';

export function PipelineForm(): JSX.Element | null {
  const { pipelines, selectedPipelineId, serverDefaults, fetchPresets, getPreset, presets } = usePipelineStore(
    useShallow((state) => ({
      pipelines: state.pipelines,
      selectedPipelineId: state.selectedPipelineId,
      serverDefaults: state.serverDefaults,
      fetchPresets: state.fetchPresets,
      getPreset: state.getPreset,
      presets: state.presets,
    }))
  );

  const pipeline = selectedPipelineId ? pipelines[selectedPipelineId] : null;

  // Get stable references from generation store
  const { setFormValue, initializeFormValues, isInitialized, status } = useGenerationStore(
    useShallow((state) => ({
      setFormValue: state.setFormValue,
      initializeFormValues: state.initializeFormValues,
      isInitialized: state.isInitialized,
      status: state.status,
    }))
  );

  // Get form values with a stable selector - only re-renders when this pipeline's values change
  // Note: selectFormValues returns a stable empty object reference when pipeline doesn't exist
  const formValues = useGenerationStore(
    useCallback(
      (state) => selectFormValues(state, pipeline?.id ?? ''),
      [pipeline?.id]
    )
  );

  // Initialize form values when pipeline changes or server defaults arrive
  useEffect(() => {
    if (pipeline && !isInitialized(pipeline.id)) {
      // For zimage, wait until serverDefaults has loaded to get variant-aware defaults
      if (pipeline.id === 'zimage' && !serverDefaults.zimage_variant) {
        return;
      }
      initializeFormValues(pipeline.id, pipeline);
    }
  }, [pipeline, serverDefaults, initializeFormValues, isInitialized]);

  // Fetch presets when pipeline changes
  useEffect(() => {
    if (pipeline) {
      fetchPresets(pipeline.id);
    }
  }, [pipeline?.id, fetchPresets]);

  // Memoize param groups with dynamically loaded preset options
  const paramGroups = useMemo(() => {
    if (!pipeline) return [];

    // Get preset names for this pipeline
    const pipelinePresets = presets[pipeline.id] || [];
    const presetNames = pipelinePresets.map((p) => p.name);

    // If we have presets, inject them into the preset param's options
    const paramsWithPresetOptions = pipeline.params.map((param) => {
      if (param.id === 'preset' && param.options_endpoint && presetNames.length > 0) {
        return {
          ...param,
          options: ['', ...presetNames],  // Empty string for "None" option
        };
      }
      return param;
    });

    return groupParams(paramsWithPresetOptions);
  }, [pipeline, presets]);

  // Stable callback for param changes
  const handleParamChange = useCallback(
    (paramId: string, value: unknown) => {
      if (!pipeline) return;
      setFormValue(pipeline.id, paramId, value);

      // Handle preset selection -> merge preset params into form values
      if (paramId === 'preset' && typeof value === 'string' && value) {
        const preset = getPreset(pipeline.id, value);
        if (preset?.params) {
          // Merge all preset params into form values
          Object.entries(preset.params).forEach(([key, val]) => {
            setFormValue(pipeline.id, key, val);
          });
        }
      }

      // Handle dimension preset -> width/height sync
      if (paramId === 'dimension_preset' && typeof value === 'string') {
        const [width, height] = value.split('x').map(Number);
        if (width && height) {
          setFormValue(pipeline.id, 'width', width);
          setFormValue(pipeline.id, 'height', height);
        }
      }

      // FLUX.2: Model change -> update steps/guidance defaults
      if (pipeline.id === 'flux2' && paramId === 'model_name' && typeof value === 'string') {
        const isBase = value.includes('base');
        // Distilled: 4 steps, 1.0 guidance | Base: 40 steps, 3.5 guidance
        setFormValue(pipeline.id, 'num_steps', isBase ? 40 : 4);
        setFormValue(pipeline.id, 'guidance', isBase ? 3.5 : 1.0);
      }
    },
    [pipeline?.id, setFormValue, getPreset]
  );

  if (!pipeline) {
    return (
      <div className="card text-center py-12">
        <p className="text-gray-400">Select a pipeline to get started</p>
      </div>
    );
  }

  const isGenerating = status === 'generating' || status === 'loading';

  return (
    <div className={`card pipeline-${pipeline.color}`}>
      {/* Pipeline header */}
      <div className="card-header">
        <div className="flex items-center gap-2">
          {pipeline.icon && <span className="text-xl">{pipeline.icon}</span>}
          <h2 className="card-title">{pipeline.name}</h2>
        </div>
        <span className="text-sm text-gray-500">{pipeline.output_type}</span>
      </div>

      {/* Form content */}
      <div className="space-y-6">
        {paramGroups.map((group) => (
          <ParamGroup key={group.id} group={group}>
            {group.params.map((param) => (
              <ParamControl
                key={param.id}
                param={param}
                value={formValues[param.id]}
                onChange={(value) => handleParamChange(param.id, value)}
                formValues={formValues}
                disabled={isGenerating}
              />
            ))}
          </ParamGroup>
        ))}

        {/* Generate button */}
        <GenerateButton
          pipelineId={pipeline.id}
          endpoint={pipeline.endpoint}
          isStreaming={pipeline.supports_streaming}
        />
      </div>
    </div>
  );
}
