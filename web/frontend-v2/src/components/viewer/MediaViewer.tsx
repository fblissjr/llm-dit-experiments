/**
 * MediaViewer Component
 *
 * Thin dispatcher for fullscreen media viewing.
 * Images delegate to ImageViewer (zoom/pan/keyboard).
 * Videos render a fullscreen modal with native controls.
 */

import { useEffect, useCallback, useRef } from 'react';
import { ImageViewer } from './ImageViewer';

type MediaType = 'image' | 'video';

interface MediaViewerProps {
  url: string;
  alt?: string;
  mediaType?: MediaType;
  audioUrl?: string;
  onClose: () => void;
}

const VIDEO_EXTENSIONS = ['.mp4', '.webm'];

function detectMediaType(url: string): MediaType {
  const lower = url.toLowerCase();
  if (VIDEO_EXTENSIONS.some((ext) => lower.endsWith(ext))) {
    return 'video';
  }
  return 'image';
}

export function MediaViewer({ url, alt, mediaType, audioUrl, onClose }: MediaViewerProps) {
  const resolved = mediaType ?? detectMediaType(url);

  if (resolved === 'image') {
    return <ImageViewer url={url} alt={alt} onClose={onClose} />;
  }

  return <VideoViewer url={url} audioUrl={audioUrl} onClose={onClose} />;
}

// ---------------------------------------------------------------------------
// VideoViewer -- fullscreen video modal
// ---------------------------------------------------------------------------

function VideoViewer({ url, audioUrl, onClose }: { url: string; audioUrl?: string; onClose: () => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  // Close on Escape
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  // Prevent body scroll while open
  useEffect(() => {
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = '';
    };
  }, []);

  const handleBackdropClick = useCallback(
    (e: React.MouseEvent) => {
      if (e.target === e.currentTarget) onClose();
    },
    [onClose]
  );

  return (
    <div
      className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center"
      onClick={handleBackdropClick}
    >
      {/* Close button */}
      <div className="absolute top-4 right-4 z-10">
        <button
          onClick={onClose}
          className="p-2 bg-gray-900/80 rounded-lg hover:bg-gray-800 transition-colors"
          title="Close (Esc)"
        >
          <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      </div>

      {/* Video */}
      <video
        ref={videoRef}
        src={url}
        controls
        autoPlay
        loop
        className="max-w-[90vw] max-h-[90vh] rounded-lg"
        onPlay={() => audioRef.current?.play()}
        onPause={() => audioRef.current?.pause()}
        onSeeked={() => {
          if (audioRef.current && videoRef.current) {
            audioRef.current.currentTime = videoRef.current.currentTime;
          }
        }}
      />
      {audioUrl && (
        <audio
          ref={audioRef}
          src={audioUrl}
          loop
        />
      )}

      {/* Hint */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 text-sm text-gray-400 bg-gray-900/80 px-4 py-2 rounded-lg">
        Press Esc to close
      </div>
    </div>
  );
}
