/**
 * VRAM Warning Dialog
 *
 * Shown when attempting to load a model that may not fit in available VRAM.
 * Displays the estimate and offers options to proceed or cancel.
 */

import { formatVRAM } from '@/types/model';
import type { VRAMEstimate } from '@/types';

interface VRAMWarningDialogProps {
  isOpen: boolean;
  estimate: VRAMEstimate;
  pipelineName: string;
  onConfirm: () => void;
  onCancel: () => void;
}

export function VRAMWarningDialog({
  isOpen,
  estimate,
  pipelineName,
  onConfirm,
  onCancel,
}: VRAMWarningDialogProps) {
  if (!isOpen) return null;

  const shortfall = estimate.requiredMB - estimate.currentFreeMB;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/60 z-50"
        onClick={onCancel}
      />

      {/* Dialog */}
      <div className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-50 w-full max-w-md">
        <div className="bg-gray-800 rounded-xl border border-gray-700 shadow-2xl">
          {/* Header */}
          <div className="p-4 border-b border-gray-700 flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-yellow-500/20 flex items-center justify-center">
              <svg className="w-6 h-6 text-yellow-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
            <div>
              <h2 className="text-lg font-medium">Low VRAM Warning</h2>
              <p className="text-sm text-gray-400">May not fit in available memory</p>
            </div>
          </div>

          {/* Content */}
          <div className="p-4 space-y-4">
            <p className="text-gray-300">
              Loading <span className="font-medium text-white">{pipelineName}</span> requires approximately{' '}
              <span className="font-medium text-yellow-400">{formatVRAM(estimate.requiredMB)}</span> of VRAM.
            </p>

            <div className="p-3 bg-gray-900 rounded-lg space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Required</span>
                <span className="text-white">{formatVRAM(estimate.requiredMB)}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Available</span>
                <span className="text-white">{formatVRAM(estimate.currentFreeMB)}</span>
              </div>
              <div className="border-t border-gray-700 pt-2 flex justify-between text-sm">
                <span className="text-gray-400">Shortfall</span>
                <span className="text-red-400">~{formatVRAM(shortfall)}</span>
              </div>
            </div>

            {estimate.suggestions && estimate.suggestions.length > 0 && (
              <div className="space-y-2">
                <p className="text-sm text-gray-400">Suggestions:</p>
                <ul className="text-sm text-gray-300 space-y-1">
                  {estimate.suggestions.map((suggestion, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className="text-yellow-500">•</span>
                      {suggestion}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <p className="text-sm text-gray-500">
              You can still try to load the model. If it fails, try unloading other models first.
            </p>
          </div>

          {/* Actions */}
          <div className="p-4 border-t border-gray-700 flex gap-3 justify-end">
            <button
              onClick={onCancel}
              className="px-4 py-2 text-sm rounded-lg border border-gray-600 text-gray-300 hover:bg-gray-700 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={onConfirm}
              className="px-4 py-2 text-sm rounded-lg bg-yellow-600 hover:bg-yellow-500 text-white transition-colors"
            >
              Load Anyway
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
