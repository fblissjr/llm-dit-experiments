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
  const { pipelines, selectedPipelineId, serverDefaults } = usePipelineStore(
    useShallow((state) => ({
      pipelines: state.pipelines,
      selectedPipelineId: state.selectedPipelineId,
      serverDefaults: state.serverDefaults,
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

  // Memoize param groups to avoid recreating on every render
  const paramGroups = useMemo(
    () => (pipeline ? groupParams(pipeline.params) : []),
    [pipeline]
  );

  // Stable callback for param changes
  const handleParamChange = useCallback(
    (paramId: string, value: unknown) => {
      if (!pipeline) return;
      setFormValue(pipeline.id, paramId, value);

      // Handle dimension preset -> width/height sync
      if (paramId === 'dimension_preset' && typeof value === 'string') {
        const [width, height] = value.split('x').map(Number);
        if (width && height) {
          setFormValue(pipeline.id, 'width', width);
          setFormValue(pipeline.id, 'height', height);
        }
      }
    },
    [pipeline?.id, setFormValue]
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
