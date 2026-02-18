/**
 * ResultDisplay Component
 *
 * Shows the generated image or video result.
 */

import { useState, useCallback } from 'react';
import { useSessionStore, useFormStore } from '@/stores';
import { MediaViewer } from '@/components/viewer';

export function ResultDisplay() {
  const result = useSessionStore((s) => s.result);
  const clearResult = useSessionStore((s) => s.clearResult);
  const [showViewer, setShowViewer] = useState(false);
  const [promptExpanded, setPromptExpanded] = useState(false);
  const setFormValue = useFormStore((s) => s.setValue);

  const useEnhancedPrompt = useCallback(() => {
    if (!result?.enhancedPrompt || !result.pipelineId) return;
    setFormValue(result.pipelineId, 'prompt', result.enhancedPrompt);
  }, [result?.enhancedPrompt, result?.pipelineId, setFormValue]);

  if (!result || result.urls.length === 0) {
    return (
      <div className="card p-8 flex items-center justify-center min-h-[300px]">
        <div className="text-center text-gray-500">
          <svg
            className="w-12 h-12 mx-auto mb-4 opacity-50"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
            />
          </svg>
          <p>Generated content will appear here</p>
        </div>
      </div>
    );
  }

  const url = result.urls[0];
  const isVideo = result.outputType === 'video' || (url && (url.endsWith('.mp4') || url.endsWith('.webm')));
  const mediaType = isVideo ? 'video' as const : 'image' as const;

  return (
    <>
      <div className="card overflow-hidden">
        {/* Media display */}
        <div className="relative bg-gray-950">
          {isVideo ? (
            <>
              <video
                src={url}
                controls
                autoPlay
                loop
                className="w-full max-h-[600px] object-contain"
              />
              {/* Expand button for video (click on video itself is play/pause) */}
              <button
                onClick={() => setShowViewer(true)}
                className="absolute top-2 left-2 p-2 bg-gray-900/80 rounded-lg
                           hover:bg-gray-800 active:bg-gray-700 transition-colors"
                title="View fullscreen"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"
                  />
                </svg>
              </button>
            </>
          ) : (
            <img
              src={url}
              alt="Generated content"
              className="w-full max-h-[600px] object-contain cursor-pointer hover:opacity-90 transition-opacity"
              onClick={() => setShowViewer(true)}
              title="Click to view full screen"
            />
          )}

        {/* Close button - 44px minimum touch target for mobile */}
        <button
          onClick={clearResult}
          className="absolute top-2 right-2 p-3 bg-gray-900/80 rounded-lg
                     hover:bg-gray-800 active:bg-gray-700 transition-colors
                     min-w-touch min-h-touch flex items-center justify-center"
          title="Clear result"
        >
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
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      </div>

      {/* Warnings */}
      {result.warnings && result.warnings.length > 0 && (
        <div className="px-3 py-2 border-t border-amber-900/50 bg-amber-950/30">
          {result.warnings.map((w, i) => (
            <p key={i} className="text-xs text-amber-400">{w}</p>
          ))}
        </div>
      )}

      {/* Enhanced prompt */}
      {result.enhancedPrompt && (
        <div className="border-t border-gray-700">
          <button
            onClick={() => setPromptExpanded(!promptExpanded)}
            className="w-full px-3 py-2 flex items-center justify-between text-xs text-gray-400
                       hover:bg-gray-800/50 transition-colors"
          >
            <span className="font-medium text-gray-300">Enhanced Prompt</span>
            <svg
              className={`w-3.5 h-3.5 transition-transform ${promptExpanded ? 'rotate-180' : ''}`}
              fill="none" viewBox="0 0 24 24" stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          {promptExpanded && (
            <div className="px-3 pb-3">
              <p className="text-xs text-gray-300 leading-relaxed whitespace-pre-wrap">
                {result.enhancedPrompt}
              </p>
              <button
                onClick={useEnhancedPrompt}
                className="mt-2 text-xs text-blue-400 hover:text-blue-300 transition-colors"
              >
                Copy to prompt field
              </button>
            </div>
          )}
        </div>
      )}

      {/* Metadata footer */}
      <div className="p-3 border-t border-gray-700 flex items-center justify-between text-sm text-gray-400">
        <div className="flex items-center gap-4">
          <span>Seed: {result.seed}</span>
          <span>{(result.durationMs / 1000).toFixed(1)}s</span>
        </div>

        {/* Download button - larger touch target for mobile */}
        <a
          href={url}
          download
          className="btn-ghost px-4 py-2 text-sm flex items-center gap-2 min-h-touch"
        >
          <svg
            className="w-4 h-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
            />
          </svg>
          Download
        </a>
      </div>
      </div>

      {/* Media viewer modal */}
      {showViewer && (
        <MediaViewer
          url={url}
          alt="Generated content"
          mediaType={mediaType}
          onClose={() => setShowViewer(false)}
        />
      )}
    </>
  );
}
