/**
 * Media utilities -- single source of truth for media type detection
 * and conversion from GenerationResult / HistoryItem to MediaItem.
 */

import type { MediaKind, MediaItem, GenerationResult, HistoryItem } from '@/api/types';

const VIDEO_EXTENSIONS = ['.mp4', '.webm'];
const AUDIO_EXTENSIONS = ['.wav', '.mp3', '.ogg', '.flac'];

/**
 * Detect media kind from a URL by extension.
 * Falls back to 'image' for unknown extensions or data URLs.
 */
export function detectKind(url: string): MediaKind {
  const lower = url.toLowerCase();
  if (VIDEO_EXTENSIONS.some((ext) => lower.endsWith(ext))) return 'video';
  if (AUDIO_EXTENSIONS.some((ext) => lower.endsWith(ext))) return 'audio';
  return 'image';
}

/**
 * Build a MediaItem from a GenerationResult.
 * Uses outputType as authoritative source, falls back to URL extension.
 */
export function mediaItemFromResult(result: GenerationResult): MediaItem {
  const url = result.urls[0] ?? '';
  const kind: MediaKind = result.outputType === 'video' ? 'video' : detectKind(url);

  return {
    kind,
    url,
    thumbnailUrl: result.thumbnailUrl,
    audioUrl: result.audioUrl,
    alt: 'Generated content',
  };
}

/**
 * Build a MediaItem from a HistoryItem.
 * Resolves fullImageUrl vs thumbnailUrl for the display URL.
 */
export function mediaItemFromHistory(item: HistoryItem): MediaItem {
  const kind: MediaKind = item.result.outputType === 'video' ? 'video' : detectKind(item.result.urls[0] ?? '');

  // For the display URL: prefer fullImageUrl (session-only, full quality),
  // then fall back to the first stored URL from the result
  const displayUrl = item.fullImageUrl || item.result.urls[0] || '';

  return {
    kind,
    url: displayUrl,
    thumbnailUrl: item.thumbnailUrl,
    audioUrl: item.audioUrl,
    alt: item.shortPrompt || 'Generated content',
  };
}
