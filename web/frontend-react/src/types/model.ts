/**
 * Model Management Types
 *
 * Types for tracking loaded models, VRAM usage, and component status.
 */

// Component types within a pipeline
export type ComponentType = 'encoder' | 'transformer' | 'vae' | 'scheduler';

// Quantization options
export type QuantizationType = 'fp32' | 'fp16' | 'bf16' | 'fp8' | 'int8' | 'int4' | 'Q4_K_M' | 'Q8_0';

// Component loading status
export type LoadStatus = 'unloaded' | 'loading' | 'loaded' | 'error';

// Device location
export type DeviceLocation = 'cpu' | 'cuda:0' | 'cuda:1' | 'offloaded';

/**
 * Status of a single model component
 */
export interface ComponentStatus {
  name: ComponentType;
  status: LoadStatus;
  device: DeviceLocation;
  quantization: QuantizationType;
  vramMB: number;
  error?: string;
}

/**
 * Status of an entire pipeline's model
 */
export interface PipelineModelStatus {
  pipelineId: string;
  status: LoadStatus;
  components: ComponentStatus[];
  totalVramMB: number;
  loadTimeMs?: number;
}

/**
 * Global VRAM status
 */
export interface VRAMStatus {
  usedMB: number;
  totalMB: number;
  freeMB: number;
  utilizationPercent: number;
  // Per-component breakdown
  breakdown: {
    label: string;
    sizeMB: number;
    color: string;
  }[];
}

/**
 * Estimation result for loading a model
 */
export interface VRAMEstimate {
  requiredMB: number;
  currentFreeMB: number;
  wouldFit: boolean;
  suggestions?: string[];
}

/**
 * Get utilization level for color coding
 */
export function getUtilizationLevel(percent: number): 'low' | 'medium' | 'high' | 'critical' {
  if (percent < 50) return 'low';
  if (percent < 75) return 'medium';
  if (percent < 90) return 'high';
  return 'critical';
}

/**
 * Format VRAM size for display
 */
export function formatVRAM(mb: number): string {
  if (mb >= 1024) {
    return `${(mb / 1024).toFixed(1)} GB`;
  }
  return `${Math.round(mb)} MB`;
}
