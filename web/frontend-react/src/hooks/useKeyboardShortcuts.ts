/**
 * Keyboard Shortcuts Hook
 *
 * Handles global keyboard shortcuts.
 */

import { useEffect } from 'react';
import { useGenerationStore } from '@/stores/generationStore';
import { usePipelineStore } from '@/stores/pipelineStore';
import { useUIStore } from '@/stores/uiStore';

export function useKeyboardShortcuts() {
  const { generate, status } = useGenerationStore();
  const { pipelines, selectedPipelineId } = usePipelineStore();
  const { toggleCommandPalette, toggleHistoryPanel, toggleModelPanel } = useUIStore();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in input
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        e.target instanceof HTMLSelectElement
      ) {
        // Allow Cmd+Enter in textareas
        if (!(e.target instanceof HTMLTextAreaElement && e.key === 'Enter' && (e.metaKey || e.ctrlKey))) {
          return;
        }
      }

      const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
      const cmdKey = isMac ? e.metaKey : e.ctrlKey;

      // Cmd+Enter - Generate
      if (cmdKey && e.key === 'Enter') {
        e.preventDefault();
        if (status !== 'generating' && selectedPipelineId) {
          const pipeline = pipelines[selectedPipelineId];
          if (pipeline) {
            generate(selectedPipelineId, pipeline.endpoint, pipeline.supports_streaming);
          }
        }
        return;
      }

      // Cmd+K - Command palette
      if (cmdKey && e.key === 'k') {
        e.preventDefault();
        toggleCommandPalette();
        return;
      }

      // Cmd+H - Toggle history
      if (cmdKey && e.key === 'h') {
        e.preventDefault();
        toggleHistoryPanel();
        return;
      }

      // M - Toggle model management sidebar (no modifier needed)
      if (e.key === 'm' || e.key === 'M') {
        e.preventDefault();
        toggleModelPanel();
        return;
      }

      // Number keys 1-6 for pipeline switching (with Cmd)
      if (cmdKey && e.key >= '1' && e.key <= '9') {
        e.preventDefault();
        const index = parseInt(e.key) - 1;
        const pipelineList = Object.values(pipelines);
        if (index < pipelineList.length) {
          usePipelineStore.getState().selectPipeline(pipelineList[index].id);
        }
        return;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [generate, status, pipelines, selectedPipelineId, toggleCommandPalette, toggleHistoryPanel, toggleModelPanel]);
}
