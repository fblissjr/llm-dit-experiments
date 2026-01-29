/**
 * DefaultsTab Component
 *
 * Generation defaults editor:
 * - Per-pipeline default values
 * - Stored in localStorage
 * - Reset to server defaults option
 */

import { useState, useEffect, useCallback } from 'react';
import { usePipelineStore } from '@/stores/pipelineStore';
import type { PipelineColor, ParamSchema } from '@/types';
import { PIPELINE_COLOR_CLASSES } from '@/types';

// Server defaults type helper
interface ServerDefaultsForPipeline {
  steps?: number;
  guidance_scale?: number;
  shift?: number;
  [key: string]: unknown;
}

// localStorage key for generation defaults
const DEFAULTS_STORAGE_KEY = 'llm-dit-generation-defaults';

interface PipelineDefaults {
  steps?: number;
  guidance_scale?: number;
  shift?: number;
  width?: number;
  height?: number;
}

type AllDefaults = Record<string, PipelineDefaults>;

// Load defaults from localStorage
function loadDefaults(): AllDefaults {
  try {
    const stored = localStorage.getItem(DEFAULTS_STORAGE_KEY);
    return stored ? JSON.parse(stored) : {};
  } catch {
    return {};
  }
}

// Save defaults to localStorage
function saveDefaults(defaults: AllDefaults): void {
  localStorage.setItem(DEFAULTS_STORAGE_KEY, JSON.stringify(defaults));
}

export function DefaultsTab() {
  const { pipelines, serverDefaults } = usePipelineStore();
  const [defaults, setDefaults] = useState<AllDefaults>(loadDefaults);
  const [selectedPipeline, setSelectedPipeline] = useState<string | null>(null);

  const pipelineList = Object.values(pipelines);

  // Set initial selection
  useEffect(() => {
    if (!selectedPipeline && pipelineList.length > 0) {
      setSelectedPipeline(pipelineList[0].id);
    }
  }, [pipelineList, selectedPipeline]);

  // Save when defaults change
  useEffect(() => {
    saveDefaults(defaults);
  }, [defaults]);

  const handleValueChange = useCallback((pipelineId: string, field: string, value: number) => {
    setDefaults((prev) => ({
      ...prev,
      [pipelineId]: {
        ...prev[pipelineId],
        [field]: value,
      },
    }));
  }, []);

  const handleReset = useCallback((pipelineId: string) => {
    setDefaults((prev) => {
      const next = { ...prev };
      delete next[pipelineId];
      return next;
    });
  }, []);

  const handleResetAll = useCallback(() => {
    setDefaults({});
  }, []);

  const currentPipeline = selectedPipeline ? pipelines[selectedPipeline] : null;
  const currentDefaults = selectedPipeline ? defaults[selectedPipeline] : {};
  const currentServerDefaults = selectedPipeline
    ? (serverDefaults?.[selectedPipeline] as ServerDefaultsForPipeline | undefined)
    : undefined;

  const hasCustomDefaults = Object.keys(defaults).length > 0;

  return (
    <div className="space-y-6">
      {/* Description */}
      <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
        <h3 className="text-sm font-medium text-gray-200 mb-2">About Generation Defaults</h3>
        <p className="text-sm text-gray-400">
          Set default values for generation parameters. These are saved in your browser
          and become the starting values when you open the app. Server defaults are
          always used as the fallback.
        </p>
      </div>

      {/* Pipeline selector */}
      <div className="flex flex-wrap items-center gap-2">
        {pipelineList.map((pipeline) => {
          const hasOverrides = !!defaults[pipeline.id] && Object.keys(defaults[pipeline.id]).length > 0;
          return (
            <button
              key={pipeline.id}
              onClick={() => setSelectedPipeline(pipeline.id)}
              className={`
                px-3 py-2 text-sm font-medium rounded-lg transition-colors
                flex items-center gap-2
                ${
                  selectedPipeline === pipeline.id
                    ? `${PIPELINE_COLOR_CLASSES.bg[pipeline.color as PipelineColor]} text-white`
                    : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                }
              `}
            >
              {pipeline.icon && <span>{pipeline.icon}</span>}
              {pipeline.name}
              {hasOverrides && (
                <span className="w-2 h-2 rounded-full bg-yellow-400" title="Has custom defaults" />
              )}
            </button>
          );
        })}
      </div>

      {/* Reset all button */}
      {hasCustomDefaults && (
        <div className="flex justify-end">
          <button
            onClick={handleResetAll}
            className="px-3 py-1.5 text-sm font-medium rounded-lg
              bg-red-600/20 text-red-400 hover:bg-red-600/30 border border-red-600/30 transition-colors"
          >
            Reset All to Server Defaults
          </button>
        </div>
      )}

      {/* Parameter editor */}
      {currentPipeline && selectedPipeline && (
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <h3 className={`text-lg font-medium ${PIPELINE_COLOR_CLASSES.text[currentPipeline.color as PipelineColor]}`}>
              {currentPipeline.icon} {currentPipeline.name} Defaults
            </h3>
            {currentDefaults && Object.keys(currentDefaults).length > 0 && (
              <button
                onClick={() => handleReset(selectedPipeline)}
                className="text-sm text-gray-400 hover:text-gray-300"
              >
                Reset to server defaults
              </button>
            )}
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <DefaultSlider
              label="Steps"
              field="steps"
              value={currentDefaults?.steps}
              serverDefault={currentServerDefaults?.steps}
              schemaDefault={getSchemaDefaultFromParams(currentPipeline.params, 'steps')}
              min={1}
              max={50}
              step={1}
              onChange={(v) => handleValueChange(selectedPipeline, 'steps', v)}
            />
            <DefaultSlider
              label="Guidance Scale (CFG)"
              field="guidance_scale"
              value={currentDefaults?.guidance_scale}
              serverDefault={currentServerDefaults?.guidance_scale}
              schemaDefault={getSchemaDefaultFromParams(currentPipeline.params, 'guidance_scale')}
              min={0}
              max={20}
              step={0.1}
              onChange={(v) => handleValueChange(selectedPipeline, 'guidance_scale', v)}
            />
            <DefaultSlider
              label="Shift"
              field="shift"
              value={currentDefaults?.shift}
              serverDefault={currentServerDefaults?.shift}
              schemaDefault={getSchemaDefaultFromParams(currentPipeline.params, 'shift')}
              min={0}
              max={10}
              step={0.1}
              onChange={(v) => handleValueChange(selectedPipeline, 'shift', v)}
            />
          </div>
        </div>
      )}
    </div>
  );
}

// Get default value from pipeline params array
function getSchemaDefaultFromParams(params: ParamSchema[] | undefined, fieldId: string): number | undefined {
  if (!params) return undefined;
  const param = params.find((p) => p.id === fieldId);
  return param?.default as number | undefined;
}

// Default slider component
interface DefaultSliderProps {
  label: string;
  field: string;
  value: number | undefined;
  serverDefault: number | undefined;
  schemaDefault: number | undefined;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}

function DefaultSlider({
  label,
  value,
  serverDefault,
  schemaDefault,
  min,
  max,
  step,
  onChange,
}: DefaultSliderProps) {
  const effectiveDefault = serverDefault ?? schemaDefault ?? min;
  const currentValue = value ?? effectiveDefault;
  const hasOverride = value !== undefined;

  return (
    <div className="space-y-2">
      <div className="flex justify-between text-sm">
        <label className="text-gray-300">{label}</label>
        <span className={hasOverride ? 'text-yellow-400' : 'text-gray-500'}>
          {currentValue.toFixed(step < 1 ? 1 : 0)}
          {hasOverride && ' (custom)'}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={currentValue}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer
          [&::-webkit-slider-thumb]:appearance-none
          [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4
          [&::-webkit-slider-thumb]:rounded-full
          [&::-webkit-slider-thumb]:bg-blue-500
          [&::-webkit-slider-thumb]:cursor-pointer"
      />
      <div className="flex justify-between text-xs text-gray-500">
        <span>{min}</span>
        {serverDefault !== undefined && (
          <span className="text-gray-400">Server: {serverDefault}</span>
        )}
        <span>{max}</span>
      </div>
    </div>
  );
}
