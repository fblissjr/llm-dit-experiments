/**
 * VideoViewer Component
 *
 * Fullscreen video modal with native controls and audio sync.
 */

import { useEffect, useCallback, useRef } from 'react';
import type { MediaItem } from '@/api/types';

interface VideoViewerProps {
  item: MediaItem;
  onClose: () => void;
}

export function VideoViewer({ item, onClose }: VideoViewerProps) {
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
        src={item.url}
        controls
        autoPlay
        loop
        className="max-w-[90vw] max-h-[90vh] rounded-lg"
        onPlay={() => audioRef.current?.play()}
        onPause={() => audioRef.current?.pause()}
        onSeeked={() => {
          if (audioRef.current && videoRef.current) {
            audioRef.current.currentTime = videoRef.current.currentTime;
            if (!videoRef.current.paused) audioRef.current.play();
          }
        }}
      />
      {item.audioUrl && (
        <audio
          ref={audioRef}
          src={item.audioUrl}
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
