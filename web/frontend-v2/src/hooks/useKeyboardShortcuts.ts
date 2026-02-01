/**
 * useKeyboardShortcuts hook
 *
 * Register global keyboard shortcuts with modifier key support.
 */

import { useEffect, useCallback } from 'react';

interface ShortcutConfig {
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  meta?: boolean;
  action: () => void;
  description?: string;
}

export function useKeyboardShortcuts(shortcuts: ShortcutConfig[]): void {
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      // Ignore if typing in an input
      const target = event.target as HTMLElement;
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.isContentEditable
      ) {
        return;
      }

      for (const shortcut of shortcuts) {
        const keyMatch = event.key.toLowerCase() === shortcut.key.toLowerCase();
        const ctrlMatch = (shortcut.ctrl ?? false) === (event.ctrlKey || event.metaKey);
        const shiftMatch = (shortcut.shift ?? false) === event.shiftKey;
        const altMatch = (shortcut.alt ?? false) === event.altKey;

        if (keyMatch && ctrlMatch && shiftMatch && altMatch) {
          event.preventDefault();
          shortcut.action();
          return;
        }
      }
    },
    [shortcuts]
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);
}

/**
 * Common shortcuts for the app
 */
export function useAppShortcuts(config: {
  onGenerate?: () => void;
  onToggleHistory?: () => void;
  onReset?: () => void;
}): void {
  const shortcuts: ShortcutConfig[] = [];

  if (config.onGenerate) {
    shortcuts.push({
      key: 'Enter',
      ctrl: true,
      action: config.onGenerate,
      description: 'Generate',
    });
  }

  if (config.onToggleHistory) {
    shortcuts.push({
      key: 'h',
      ctrl: true,
      action: config.onToggleHistory,
      description: 'Toggle history',
    });
  }

  if (config.onReset) {
    shortcuts.push({
      key: 'r',
      ctrl: true,
      shift: true,
      action: config.onReset,
      description: 'Reset form',
    });
  }

  useKeyboardShortcuts(shortcuts);
}
