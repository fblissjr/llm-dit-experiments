/**
 * GenerateButton Component
 *
 * Primary action button with validation, loading state, and cancel support.
 */

import { useCallback } from 'react';
import { cn } from '@/utils';
import { useAppStore, useFormStore, useSessionStore } from '@/stores';

export function GenerateButton() {
  const selectedPipelineId = useAppStore((s) => s.selectedPipelineId);
  const getPipelineColor = useAppStore((s) => s.getPipelineColor);

  const validate = useFormStore((s) => s.validate);
  const hasErrors = useFormStore((s) =>
    selectedPipelineId ? s.hasErrors(selectedPipelineId) : false
  );

  const status = useSessionStore((s) => s.status);
  const startGeneration = useSessionStore((s) => s.startGeneration);
  const cancelGeneration = useSessionStore((s) => s.cancelGeneration);

  const isGenerating = status === 'generating';

  const handleClick = useCallback(() => {
    if (!selectedPipelineId) return;

    if (isGenerating) {
      cancelGeneration();
    } else {
      // Validate first
      const errors = validate(selectedPipelineId);
      if (errors.length === 0) {
        startGeneration(selectedPipelineId);
      }
    }
  }, [selectedPipelineId, isGenerating, validate, startGeneration, cancelGeneration]);

  const pipelineColor = selectedPipelineId
    ? getPipelineColor(selectedPipelineId)
    : '#3b82f6';

  return (
    <button
      type="button"
      onClick={handleClick}
      disabled={!selectedPipelineId || hasErrors}
      className={cn(
        'btn-primary w-full py-3 text-lg font-medium',
        'flex items-center justify-center gap-2',
        isGenerating && 'bg-red-600 hover:bg-red-700'
      )}
      style={
        !isGenerating
          ? ({ backgroundColor: pipelineColor } as React.CSSProperties)
          : undefined
      }
    >
      {isGenerating ? (
        <>
          <svg
            className="w-5 h-5 animate-spin"
            fill="none"
            viewBox="0 0 24 24"
          >
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
          Cancel
        </>
      ) : (
        <>
          <svg
            className="w-5 h-5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M13 10V3L4 14h7v7l9-11h-7z"
            />
          </svg>
          Generate
        </>
      )}
    </button>
  );
}
