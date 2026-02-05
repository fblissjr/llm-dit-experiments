/**
 * ResultDisplay Component
 *
 * Shows the generated image or video result.
 */

import { useState } from 'react';
import { useSessionStore } from '@/stores';
import { ImageViewer } from '@/components/viewer/ImageViewer';

export function ResultDisplay() {
  const result = useSessionStore((s) => s.result);
  const clearResult = useSessionStore((s) => s.clearResult);
  const [showViewer, setShowViewer] = useState(false);

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
  // Check outputType first, then file extension (data URLs don't have extensions)
  const isVideo = result.outputType === 'video' || (url && (url.endsWith('.mp4') || url.endsWith('.webm')));

  const handleImageClick = () => {
    if (!isVideo) {
      setShowViewer(true);
    }
  };

  return (
    <>
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
              className="w-full max-h-[600px] object-contain cursor-pointer hover:opacity-90 transition-opacity"
              onClick={handleImageClick}
              title="Click to view full screen"
            />
          )}

        {/* Close button - 44px minimum touch target for mobile */}
        <button
          onClick={clearResult}
          className="absolute top-2 right-2 p-3 bg-gray-900/80 rounded-lg
                     hover:bg-gray-800 active:bg-gray-700 transition-colors
                     min-w-[44px] min-h-[44px] flex items-center justify-center"
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

        {/* Download button - larger touch target for mobile */}
        <a
          href={url}
          download
          className="btn-ghost px-4 py-2 text-sm flex items-center gap-2 min-h-[44px]"
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

      {/* Image viewer modal */}
      {showViewer && !isVideo && (
        <ImageViewer
          url={url}
          alt="Generated content"
          onClose={() => setShowViewer(false)}
        />
      )}
    </>
  );
}
