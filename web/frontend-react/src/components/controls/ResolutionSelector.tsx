/**
 * Resolution Selector
 *
 * Shared component for selecting image/video dimensions.
 * Supports presets and custom values with VAE rounding.
 */

import { useState, useEffect } from 'react';
import { Select } from './Select';
import { NumberInput } from './Number';

interface ResolutionSelectorProps {
  width: number;
  height: number;
  onWidthChange: (value: number) => void;
  onHeightChange: (value: number) => void;
  presets?: string[];  // e.g., ["1024x1024", "768x512"]
  minWidth?: number;
  maxWidth?: number;
  minHeight?: number;
  maxHeight?: number;
  roundTo?: number;    // VAE pixel rounding (16, 32, 64)
  tooltip?: string;
  disabled?: boolean;
  className?: string;
}

export function ResolutionSelector({
  width,
  height,
  onWidthChange,
  onHeightChange,
  presets = ['1024x1024', '1152x896', '1216x832', '768x1344', '832x1216'],
  minWidth = 256,
  maxWidth = 4096,
  minHeight = 256,
  maxHeight = 4096,
  roundTo = 64,
  tooltip,
  disabled = false,
  className = '',
}: ResolutionSelectorProps) {
  const [selectedPreset, setSelectedPreset] = useState<string>('custom');

  // Sync preset selection with current width/height
  useEffect(() => {
    const currentPreset = `${width}x${height}`;
    if (presets.includes(currentPreset)) {
      setSelectedPreset(currentPreset);
    } else {
      setSelectedPreset('custom');
    }
  }, [width, height, presets]);

  const handlePresetChange = (preset: string) => {
    if (preset === 'custom') {
      setSelectedPreset('custom');
      return;
    }

    const [w, h] = preset.split('x').map(Number);
    if (w && h) {
      onWidthChange(roundToNearest(w, roundTo));
      onHeightChange(roundToNearest(h, roundTo));
      setSelectedPreset(preset);
    }
  };

  const handleWidthChange = (value: number) => {
    const rounded = roundToNearest(value, roundTo);
    onWidthChange(Math.max(minWidth, Math.min(maxWidth, rounded)));
    setSelectedPreset('custom');
  };

  const handleHeightChange = (value: number) => {
    const rounded = roundToNearest(value, roundTo);
    onHeightChange(Math.max(minHeight, Math.min(maxHeight, rounded)));
    setSelectedPreset('custom');
  };

  return (
    <div className={`space-y-3 ${className}`} title={tooltip}>
      {/* Preset selector */}
      <Select
        label="Dimension Preset"
        value={selectedPreset}
        onChange={handlePresetChange}
        options={['custom', ...presets]}
        disabled={disabled}
      />

      {/* Custom dimensions */}
      <div className="grid grid-cols-2 gap-3">
        <NumberInput
          label="Width"
          value={width}
          onChange={handleWidthChange}
          min={minWidth}
          max={maxWidth}
          step={roundTo}
          disabled={disabled}
          tooltip={`Width in pixels. Rounded to ${roundTo}px for VAE.`}
        />
        <NumberInput
          label="Height"
          value={height}
          onChange={handleHeightChange}
          min={minHeight}
          max={maxHeight}
          step={roundTo}
          disabled={disabled}
          tooltip={`Height in pixels. Rounded to ${roundTo}px for VAE.`}
        />
      </div>

      {/* Aspect ratio indicator */}
      <div className="text-xs text-gray-500">
        Aspect ratio: {getAspectRatio(width, height)}
      </div>
    </div>
  );
}

function roundToNearest(value: number, nearest: number): number {
  return Math.round(value / nearest) * nearest;
}

function gcd(a: number, b: number): number {
  return b === 0 ? a : gcd(b, a % b);
}

function getAspectRatio(width: number, height: number): string {
  const divisor = gcd(width, height);
  const ratioW = width / divisor;
  const ratioH = height / divisor;

  // Simplify common ratios
  if (ratioW === ratioH) return '1:1 (Square)';
  if (ratioW === 16 && ratioH === 9) return '16:9 (Widescreen)';
  if (ratioW === 9 && ratioH === 16) return '9:16 (Vertical)';
  if (ratioW === 4 && ratioH === 3) return '4:3 (Standard)';
  if (ratioW === 3 && ratioH === 4) return '3:4 (Portrait)';
  if (ratioW === 3 && ratioH === 2) return '3:2 (Photo)';
  if (ratioW === 2 && ratioH === 3) return '2:3 (Portrait Photo)';

  return `${ratioW}:${ratioH}`;
}
