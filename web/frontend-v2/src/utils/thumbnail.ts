/**
 * Thumbnail Generation Utilities
 *
 * Creates compressed thumbnails from base64 images for storage efficiency.
 */

import { logger } from './logger';

const log = logger('Thumbnail');

const THUMBNAIL_SIZE = 80;
const THUMBNAIL_QUALITY = 0.6;

/**
 * Create a thumbnail from a base64 image data URL.
 * Returns a compressed base64 thumbnail suitable for history storage.
 *
 * @param dataUrl - Base64 data URL (e.g., "data:image/png;base64,...")
 * @param maxSize - Maximum width/height for thumbnail (default: 80px)
 * @param quality - JPEG compression quality 0-1 (default: 0.6)
 * @returns Promise resolving to thumbnail data URL, or empty string on error
 */
export async function createThumbnail(
  dataUrl: string,
  maxSize = THUMBNAIL_SIZE,
  quality = THUMBNAIL_QUALITY
): Promise<string> {
  return new Promise((resolve) => {
    // Create image element
    const img = new Image();

    img.onload = () => {
      try {
        // Calculate thumbnail dimensions (preserve aspect ratio)
        let width = img.width;
        let height = img.height;

        if (width > height) {
          if (width > maxSize) {
            height = (height * maxSize) / width;
            width = maxSize;
          }
        } else {
          if (height > maxSize) {
            width = (width * maxSize) / height;
            height = maxSize;
          }
        }

        // Create canvas and draw scaled image
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;

        const ctx = canvas.getContext('2d');
        if (!ctx) {
          resolve('');
          return;
        }

        // Use better image smoothing
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';

        ctx.drawImage(img, 0, 0, width, height);

        // Convert to compressed JPEG data URL
        const thumbnailUrl = canvas.toDataURL('image/jpeg', quality);
        resolve(thumbnailUrl);
      } catch (error) {
        log.error('Error creating thumbnail:', error);
        resolve('');
      }
    };

    img.onerror = () => {
      log.error('Error loading image for thumbnail');
      resolve('');
    };

    // Start loading the image
    img.src = dataUrl;
  });
}

/**
 * Check if a URL is a base64 data URL
 */
export function isBase64DataUrl(url: string): boolean {
  return url.startsWith('data:');
}

/**
 * Estimate the size in bytes of a base64 data URL
 */
export function estimateDataUrlSize(dataUrl: string): number {
  // Base64 encoding is ~4/3 the size of the original data
  // Also account for the "data:image/...;base64," prefix
  const base64Data = dataUrl.split(',')[1] || '';
  return Math.ceil((base64Data.length * 3) / 4);
}
