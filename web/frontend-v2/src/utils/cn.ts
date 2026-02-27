/**
 * Tailwind class name utility
 *
 * Combines class names, filtering out falsy values.
 * A lightweight alternative to clsx/classnames.
 */

export function cn(...classes: (string | false | null | undefined)[]): string {
  return classes.filter(Boolean).join(' ');
}
