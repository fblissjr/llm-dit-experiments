/**
 * PresetBrowser - Main container for the visual preset card system.
 *
 * Orchestrates:
 * - Grouping presets by category (hides "testing" category)
 * - Active/modified state from formStore
 * - Preset selection -> applyPreset + applyDependentDefaults
 * - Restore/clear actions
 *
 * Only renders when the pipeline has presets.
 */

import { useMemo, useCallback } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useAppStore, useFormStore } from '@/stores';
import type { GenerationPreset } from '@/api/types';
import { ActivePresetIndicator } from './ActivePresetIndicator';
import { PresetCategorySection } from './PresetCategorySection';

interface PresetBrowserProps {
  pipelineId: string;
}

// Categories to hide from the browser
const HIDDEN_CATEGORIES = new Set(['testing']);

export function PresetBrowser({ pipelineId }: PresetBrowserProps) {
  const presets = useAppStore(
    useShallow((s) => s.presets[pipelineId] ?? [])
  );
  const pipeline = useAppStore(
    useShallow((s) => s.pipelines[pipelineId] ?? null)
  );
  const getPipelineColor = useAppStore((s) => s.getPipelineColor);
  const pipelineColor = getPipelineColor(pipelineId);

  const activePresetName = useFormStore((s) => s.getActivePresetName(pipelineId));
  const isModified = useFormStore((s) => s.isPresetModified(pipelineId));
  const applyPreset = useFormStore((s) => s.applyPreset);
  const clearPreset = useFormStore((s) => s.clearPreset);
  const restorePreset = useFormStore((s) => s.restorePreset);
  const applyDependentDefaults = useFormStore((s) => s.applyDependentDefaults);

  // Group presets by category, filtering out hidden categories
  const categorizedPresets = useMemo(() => {
    const groups = new Map<string, GenerationPreset[]>();
    for (const preset of presets) {
      const cat = preset.category || 'general';
      if (HIDDEN_CATEGORIES.has(cat)) continue;
      if (!groups.has(cat)) {
        groups.set(cat, []);
      }
      groups.get(cat)!.push(preset);
    }
    return groups;
  }, [presets]);

  // Handle selecting a preset
  const handleSelectPreset = useCallback(
    (presetName: string) => {
      const preset = presets.find((p) => p.name === presetName);
      if (!preset) return;

      // If clicking the already-active preset, deselect it
      if (presetName === activePresetName) {
        clearPreset(pipelineId);
        return;
      }

      applyPreset(pipelineId, presetName, preset.params);

      // Fire dependent defaults for any params the preset set that are triggers
      if (pipeline) {
        for (const paramId of Object.keys(preset.params)) {
          const hasDependents = pipeline.params.some(
            (p) => p.dependent_defaults?.[paramId]
          );
          if (hasDependents) {
            applyDependentDefaults(pipelineId, paramId, preset.params[paramId]);
          }
        }
      }
    },
    [pipelineId, presets, activePresetName, pipeline, applyPreset, clearPreset, applyDependentDefaults]
  );

  const handleRestore = useCallback(() => {
    restorePreset(pipelineId);
  }, [pipelineId, restorePreset]);

  const handleClear = useCallback(() => {
    clearPreset(pipelineId);
  }, [pipelineId, clearPreset]);

  // Don't render if no presets
  if (presets.length === 0 || categorizedPresets.size === 0) {
    return null;
  }

  return (
    <div className="space-y-3">
      <ActivePresetIndicator
        presetName={activePresetName}
        isModified={isModified}
        pipelineColor={pipelineColor}
        onRestore={handleRestore}
        onClear={handleClear}
      />

      {Array.from(categorizedPresets.entries()).map(([category, categoryPresets]) => (
        <PresetCategorySection
          key={category}
          category={category}
          presets={categoryPresets}
          activePresetName={activePresetName}
          pipelineColor={pipelineColor}
          onSelectPreset={handleSelectPreset}
        />
      ))}
    </div>
  );
}
