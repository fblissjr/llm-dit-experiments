/**
 * Generate Button
 *
 * Main action button for triggering generation.
 * Shows time estimate and handles loading state.
 * Shows "Load Model" button when no model is loaded.
 */

import { useGenerationStore } from '@/stores/generationStore';
import { useModelStore } from '@/stores/modelStore';
import { usePipelineStore } from '@/stores/pipelineStore';

interface GenerateButtonProps {
  pipelineId: string;
  endpoint: string;
  isStreaming: boolean;
}

export function GenerateButton({
  pipelineId,
  endpoint,
  isStreaming,
}: GenerateButtonProps) {
  const { status, generate, cancelGeneration, getTimeEstimate } = useGenerationStore();
  const { pipelines } = usePipelineStore();
  const { isModelLoaded, loadModel, isLoadingModel, loadingPipelineId } = useModelStore();
  const pipeline = pipelines[pipelineId];

  const isGenerating = status === 'generating' || status === 'loading';
  const estimate = getTimeEstimate(pipelineId);
  const modelLoaded = isModelLoaded(pipelineId);
  const isLoadingThisModel = isLoadingModel && loadingPipelineId === pipelineId;

  const handleClick = () => {
    if (isGenerating) {
      cancelGeneration();
    } else {
      generate(pipelineId, endpoint, isStreaming);
    }
  };

  const handleLoadModel = () => {
    loadModel(pipelineId);
  };

  // Button color based on pipeline
  const colorClasses: Record<string, string> = {
    blue: 'bg-blue-600 hover:bg-blue-500 focus:ring-blue-500',
    purple: 'bg-purple-600 hover:bg-purple-500 focus:ring-purple-500',
    orange: 'bg-orange-600 hover:bg-orange-500 focus:ring-orange-500',
    teal: 'bg-teal-600 hover:bg-teal-500 focus:ring-teal-500',
    green: 'bg-green-600 hover:bg-green-500 focus:ring-green-500',
    pink: 'bg-pink-600 hover:bg-pink-500 focus:ring-pink-500',
  };

  const buttonColor = pipeline?.color ? colorClasses[pipeline.color] : colorClasses.blue;

  // Show "Load Model" button if model isn't loaded
  if (!modelLoaded) {
    return (
      <div className="pt-4 border-t border-gray-700 space-y-3">
        {/* Model not loaded notice */}
        <div className="p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
          <p className="text-sm text-yellow-400 text-center">
            Model not loaded. Load it to start generating.
          </p>
        </div>

        <button
          onClick={handleLoadModel}
          disabled={isLoadingModel}
          className={`
            w-full py-3 px-6 rounded-lg font-medium text-white
            transition-all duration-200
            focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900
            ${isLoadingThisModel
              ? 'bg-gray-600 cursor-wait'
              : 'bg-green-600 hover:bg-green-500 focus:ring-green-500'
            }
          `}
        >
          {isLoadingThisModel ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              Loading {pipeline?.name ?? pipelineId}...
            </span>
          ) : (
            <span className="flex items-center justify-center gap-2">
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
              </svg>
              Load {pipeline?.name ?? pipelineId}
            </span>
          )}
        </button>
      </div>
    );
  }

  return (
    <div className="pt-4 border-t border-gray-700">
      <button
        onClick={handleClick}
        className={`
          w-full py-3 px-6 rounded-lg font-medium text-white
          transition-all duration-200
          focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900
          ${isGenerating ? 'bg-red-600 hover:bg-red-500 focus:ring-red-500' : buttonColor}
          ${isGenerating ? 'generating' : ''}
        `}
      >
        {isGenerating ? (
          <span className="flex items-center justify-center gap-2">
            <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
            Cancel Generation
          </span>
        ) : (
          <span className="flex items-center justify-center gap-2">
            Generate
            {estimate.estimatedSeconds > 0 && estimate.confidence !== 'low' && (
              <span className="text-sm opacity-75">
                ~{estimate.estimatedSeconds}s
              </span>
            )}
          </span>
        )}
      </button>

      {/* Keyboard shortcut hint */}
      <p className="text-xs text-gray-500 text-center mt-2">
        Press <kbd className="px-1.5 py-0.5 bg-gray-800 rounded text-gray-400">⌘</kbd> + <kbd className="px-1.5 py-0.5 bg-gray-800 rounded text-gray-400">Enter</kbd> to generate
      </p>
    </div>
  );
}
