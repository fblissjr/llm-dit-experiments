/**
 * MediaViewer Component
 *
 * Thin dispatcher for fullscreen media viewing.
 * Images delegate to ImageViewer (zoom/pan/keyboard).
 * Videos delegate to VideoViewer (native controls + audio sync).
 */

import type { MediaItem } from '@/api/types';
import { ImageViewer } from './ImageViewer';
import { VideoViewer } from './VideoViewer';

interface MediaViewerProps {
  item: MediaItem;
  onClose: () => void;
}

export function MediaViewer({ item, onClose }: MediaViewerProps) {
  if (item.kind === 'image') {
    return <ImageViewer url={item.url} alt={item.alt} onClose={onClose} />;
  }

  return <VideoViewer item={item} onClose={onClose} />;
}
