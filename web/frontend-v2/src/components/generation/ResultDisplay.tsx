/**
 * ResultDisplay Component
 *
 * Shows the generated image or video result.
 */

import { useSessionStore } from '@/stores';

export function ResultDisplay() {
  const result = useSessionStore((s) => s.result);
  const clearResult = useSessionStore((s) => s.clearResult);

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
  const isVideo = result.outputType === 'video' || url.endsWith('.mp4') || url.endsWith('.webm');

  return (
    <div className="card overflow-hidden">
      {/* Media display */}
      <div className="relative bg-gray-950">
        {isVideo ? (
          <video
            src={url}
            controls
            autoPlay
            loop
            className="w-full max-h-[600px] object-contain"
          />
        ) : (
          <img
            src={url}
            alt="Generated content"
            className="w-full max-h-[600px] object-contain"
          />
        )}

        {/* Close button */}
        <button
          onClick={clearResult}
          className="absolute top-2 right-2 p-2 bg-gray-900/80 rounded-lg
                     hover:bg-gray-800 transition-colors"
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

      {/* Metadata footer */}
      <div className="p-3 border-t border-gray-700 flex items-center justify-between text-sm text-gray-400">
        <div className="flex items-center gap-4">
          <span>Seed: {result.seed}</span>
          <span>{(result.durationMs / 1000).toFixed(1)}s</span>
        </div>

        {/* Download button */}
        <a
          href={url}
          download
          className="btn-ghost px-3 py-1 text-sm flex items-center gap-1"
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
  );
}
