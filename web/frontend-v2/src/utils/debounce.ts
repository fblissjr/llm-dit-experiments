/**
 * Simple debounce utility.
 *
 * Returns a debounced function that delays invocation until after `delay` ms
 * have elapsed since the last call. Includes a `.flush()` method to
 * immediately invoke the pending call (used for pointer-up commits).
 */

export function debounce<T extends (...args: Parameters<T>) => void>(
  fn: T,
  delay: number
): T & { flush: () => void } {
  let timer: ReturnType<typeof setTimeout> | null = null;
  let lastArgs: Parameters<T> | null = null;

  const debounced = ((...args: Parameters<T>) => {
    lastArgs = args;
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      lastArgs = null;
      fn(...args);
    }, delay);
  }) as T & { flush: () => void };

  debounced.flush = () => {
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }
    if (lastArgs) {
      fn(...(lastArgs as Parameters<T>));
      lastArgs = null;
    }
  };

  return debounced;
}
