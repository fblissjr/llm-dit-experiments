/**
 * Lightweight namespaced logger.
 *
 * Usage:
 *   const log = logger('API');
 *   log.info('Request started', { endpoint });
 *   log.error('Request failed', error);
 *
 * All output is prefixed with [Namespace] for easy DevTools filtering.
 * Log levels: debug < info < warn < error < silent
 * Default level in dev: debug. In production builds: warn.
 * Override with VITE_LOG_LEVEL env var.
 */

type LogLevel = 'debug' | 'info' | 'warn' | 'error' | 'silent';

const LEVELS: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
  silent: 4,
};

const envLevel = import.meta.env.VITE_LOG_LEVEL as LogLevel | undefined;
const defaultLevel: LogLevel = import.meta.env.DEV ? 'debug' : 'warn';
const activeLevel = LEVELS[envLevel ?? defaultLevel] ?? LEVELS[defaultLevel];

export interface NamespacedLogger {
  debug: (...args: unknown[]) => void;
  info: (...args: unknown[]) => void;
  warn: (...args: unknown[]) => void;
  error: (...args: unknown[]) => void;
}

export function logger(namespace: string): NamespacedLogger {
  const prefix = `[${namespace}]`;
  return {
    debug: (...args) => { if (activeLevel <= LEVELS.debug) console.debug(prefix, ...args); },
    info:  (...args) => { if (activeLevel <= LEVELS.info)  console.info(prefix, ...args); },
    warn:  (...args) => { if (activeLevel <= LEVELS.warn)  console.warn(prefix, ...args); },
    error: (...args) => { if (activeLevel <= LEVELS.error) console.error(prefix, ...args); },
  };
}
