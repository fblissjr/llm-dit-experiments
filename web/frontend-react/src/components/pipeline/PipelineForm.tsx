/**
 * Pipeline Form
 *
 * Schema-driven form that renders controls based on the selected pipeline's schema.
 * Groups parameters and handles form state.
 */

import { usePipelineStore } from '@/stores/pipelineStore';
import { useGenerationStore } from '@/stores/generationStore';
import { ParamGroup } from './ParamGroup';
import { ParamControl } from './ParamControl';
import { GenerateButton } from '../generation/GenerateButton';
import { groupParams } from '@/types';

export function PipelineForm() {
  const { pipelines, selectedPipelineId } = usePipelineStore();
  const { getFormValues, setFormValue, status } = useGenerationStore();

  const pipeline = selectedPipelineId ? pipelines[selectedPipelineId] : null;

  if (!pipeline) {
    return (
      <div className="card text-center py-12">
        <p className="text-gray-400">Select a pipeline to get started</p>
      </div>
    );
  }

  const formValues = getFormValues(pipeline.id, pipeline);
  const paramGroups = groupParams(pipeline.params);
  const isGenerating = status === 'generating' || status === 'loading';

  const handleParamChange = (paramId: string, value: unknown) => {
    setFormValue(pipeline.id, paramId, value);

    // Handle dimension preset -> width/height sync
    if (paramId === 'dimension_preset' && typeof value === 'string') {
      const [width, height] = value.split('x').map(Number);
      if (width && height) {
        setFormValue(pipeline.id, 'width', width);
        setFormValue(pipeline.id, 'height', height);
      }
    }
  };

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
