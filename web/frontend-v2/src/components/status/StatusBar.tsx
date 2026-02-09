/**
 * StatusBar Component
 *
 * Persistent compact status strip showing loaded model variant,
 * active LoRAs, quantization, and VRAM usage. Expandable for details.
 *
 * Desktop: 36px tall bar with accordion expansion below.
 * Mobile: 32px tall single-line bar, tap to expand as bottom sheet.
 */

import { useState } from 'react';
import { cn, formatUptime } from '@/utils';
import { useAppStore } from '@/stores';
import { useIsMobile } from '@/hooks';
import { RestartWarning } from '@/components/common/RestartWarning';

function VRAMBar({ usedGb, totalGb }: { usedGb: number; totalGb: number }) {
  const percent = totalGb > 0 ? (usedGb / totalGb) * 100 : 0;
  return (
    <div className="flex items-center gap-2 text-xs">
      <div className="w-20 h-1.5 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={cn(
            'h-full transition-all',
            percent > 90 ? 'bg-red-500' : percent > 70 ? 'bg-yellow-500' : 'bg-blue-500'
          )}
          style={{ width: `${Math.min(100, percent)}%` }}
        />
      </div>
      <span className="text-gray-400 tabular-nums">
        {usedGb.toFixed(1)}/{totalGb.toFixed(1)}
      </span>
    </div>
  );
}

function LoRABadge({ name, scale }: { name: string; scale: number }) {
  return (
    <span className="inline-flex items-center gap-1 px-1.5 py-0.5 text-xs bg-purple-500/20 text-purple-300 rounded">
      {name}
      <span className="text-purple-400/70">@{scale.toFixed(2)}</span>
    </span>
  );
}

function QuantBadge({ method }: { method: string }) {
  if (method === 'none') return null;
  return (
    <span className="inline-flex items-center px-1.5 py-0.5 text-xs bg-orange-500/15 text-orange-300 rounded">
      {method.toUpperCase()}
    </span>
  );
}

function ExpandedDetails() {
  const ctx = useAppStore((s) => s.generationContext);

  if (!ctx) return null;

  return (
    <div className="px-4 py-3 space-y-3 text-sm border-t border-gray-700/50">
      {/* Model details */}
      <div>
        <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-1">Model</h4>
        <p className="text-gray-200">
          {ctx.pipelineDisplayName ?? 'No model loaded'}
          {ctx.modelVariant && (
            <span className="text-gray-500 ml-2">({ctx.modelVariant})</span>
          )}
        </p>
      </div>

      {/* LoRA list */}
      {ctx.loras.length > 0 && (
        <div>
          <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-1">LoRAs</h4>
          <div className="space-y-1">
            {ctx.loras.map((lora) => (
              <div key={lora.path} className="flex items-center justify-between text-gray-300">
                <span className="truncate">{lora.name}</span>
                <span className="text-gray-500 tabular-nums ml-2">
                  scale: {lora.scale.toFixed(2)} | {lora.layersUpdated} layers
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Quantization */}
      {Object.keys(ctx.quantization).length > 0 && (
        <div>
          <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-1">Quantization</h4>
          <div className="flex flex-wrap gap-2">
            {Object.entries(ctx.quantization).map(([component, method]) => (
              <span key={component} className="text-gray-300">
                <span className="text-gray-500">{component}:</span>{' '}
                {method === 'none' ? 'none' : method.toUpperCase()}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Optimization */}
      <div className="flex flex-wrap gap-3 text-gray-400">
        {ctx.compileEnabled && (
          <span>
            compiled{ctx.compileMode ? ` (${ctx.compileMode})` : ''}
          </span>
        )}
        {ctx.blockOffload && <span>block offload</span>}
        {ctx.uptimeSeconds != null && (
          <span>uptime: {formatUptime(ctx.uptimeSeconds)}</span>
        )}
        {ctx.profile !== 'default' && <span>profile: {ctx.profile}</span>}
      </div>

      {/* Pending restart warning */}
      <RestartWarning fields={ctx.pendingRestartFields} />
    </div>
  );
}

export function StatusBar() {
  const ctx = useAppStore((s) => s.generationContext);
  const isMobile = useIsMobile();
  const [isExpanded, setIsExpanded] = useState(false);

  // Status dot color
  const dotColor = ctx?.activePipeline ? '#22c55e' : '#6b7280';

  return (
    <>
      {/* Compact bar */}
      <div
        className={cn(
          'bg-gray-800/60 border-b border-gray-700 cursor-pointer select-none',
          'transition-colors hover:bg-gray-800/80',
          isMobile ? 'h-8' : 'h-9'
        )}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className={cn(
          'h-full flex items-center gap-3 px-4',
          isMobile ? 'text-xs' : 'text-sm'
        )}>
          {/* Status dot */}
          <div
            className="w-2 h-2 rounded-full flex-shrink-0"
            style={{ backgroundColor: dotColor }}
          />

          {/* Pipeline display name */}
          <span className="text-gray-200 truncate font-medium">
            {ctx?.pipelineDisplayName ?? 'No model loaded'}
          </span>

          {/* Desktop: show inline details */}
          {!isMobile && ctx?.activePipeline && (
            <>
              {/* Separator */}
              {ctx.loras.length > 0 && (
                <>
                  <span className="text-gray-600">|</span>
                  <div className="flex items-center gap-1.5 overflow-hidden">
                    {ctx.loras.map((lora) => (
                      <LoRABadge key={lora.path} name={lora.name} scale={lora.scale} />
                    ))}
                  </div>
                </>
              )}

              {/* Quantization badge */}
              {Object.values(ctx.quantization).some((m) => m !== 'none') && (
                <>
                  <span className="text-gray-600">|</span>
                  {/* Show the most significant quant method */}
                  <QuantBadge method={ctx.quantization.transformer ?? ctx.quantization.encoder ?? 'none'} />
                </>
              )}

              {/* Compile badge */}
              {ctx.compileEnabled && (
                <span className="text-xs text-blue-400/70">compiled</span>
              )}
            </>
          )}

          {/* Spacer */}
          <div className="flex-1" />

          {/* VRAM bar (desktop) */}
          {!isMobile && ctx?.vramUsedGb != null && ctx?.vramTotalGb != null && (
            <VRAMBar usedGb={ctx.vramUsedGb} totalGb={ctx.vramTotalGb} />
          )}

          {/* Pending restart indicator */}
          {ctx && ctx.pendingRestartFields.length > 0 && (
            <div className="w-2 h-2 rounded-full bg-yellow-400 flex-shrink-0" title="Restart required" />
          )}

          {/* Expand chevron */}
          <svg
            className={cn(
              'w-4 h-4 text-gray-500 transition-transform flex-shrink-0',
              isExpanded && 'rotate-180'
            )}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </div>

      {/* Expanded details */}
      {isExpanded && (
        isMobile ? (
          // Mobile: bottom sheet overlay
          <div
            className="fixed inset-0 bg-black/50 z-40 md:hidden"
            onClick={() => setIsExpanded(false)}
          >
            <div
              className="absolute bottom-0 left-0 right-0 bg-gray-900 rounded-t-2xl border-t border-gray-700 max-h-[70vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
                <h2 className="font-semibold">Status Details</h2>
                <button
                  onClick={() => setIsExpanded(false)}
                  className="p-1 hover:bg-gray-800 rounded transition-colors"
                >
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <ExpandedDetails />
            </div>
          </div>
        ) : (
          // Desktop: inline accordion
          <div className="bg-gray-800/40 border-b border-gray-700">
            <ExpandedDetails />
          </div>
        )
      )}
    </>
  );
}
