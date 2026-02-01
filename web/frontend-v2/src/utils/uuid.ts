/**
 * UUID generation with fallback for non-secure contexts
 *
 * crypto.randomUUID() requires a secure context (HTTPS or localhost).
 * When accessing via HTTP on a local IP, it won't be available.
 * This provides a fallback implementation.
 */

export function generateUUID(): string {
  // Use native crypto.randomUUID if available (secure context)
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }

  // Fallback: Generate a v4-like UUID using Math.random()
  // Not cryptographically secure, but fine for client-side IDs
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}
