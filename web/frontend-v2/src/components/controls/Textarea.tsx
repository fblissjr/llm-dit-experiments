/**
 * Textarea Control
 *
 * Multi-line text input for prompts and long text.
 * Supports an optional inline action button (e.g., prompt upsample)
 * rendered in the top-right corner of the textarea.
 */

import { useId } from 'react';
import { cn } from '@/utils';

interface TextareaProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  rows?: number;
  tooltip?: string;
  disabled?: boolean;
  required?: boolean;
  error?: string;
  className?: string;
  /** Called when the inline action button is clicked. */
  onAction?: () => void;
  /** Accessible tooltip/title for the action button. */
  actionLabel?: string;
  /** When true, the action button shows a spinner instead of its icon. */
  actionLoading?: boolean;
}

/**
 * Sparkle/wand SVG icon for the prompt upsample action.
 * 16x16 viewBox, single-color fill (currentColor).
 */
function SparkleIcon() {
  return (
    <svg
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className="w-4 h-4"
    >
      {/* Central star burst */}
      <path d="M8 2 L8.7 6 L12 6.5 L8.7 7 L8 11 L7.3 7 L4 6.5 L7.3 6 Z" />
      {/* Small accent sparkles */}
      <path d="M13 2 L13.4 4 L15 4.2 L13.4 4.4 L13 6.4 L12.6 4.4 L11 4.2 L12.6 4 Z" />
      <path d="M3 10 L3.3 11.5 L4.8 11.7 L3.3 11.9 L3 13.4 L2.7 11.9 L1.2 11.7 L2.7 11.5 Z" />
    </svg>
  );
}

/**
 * CSS spinner for loading state.
 * 16x16, single-color border animation.
 */
function Spinner() {
  return (
    <span
      className="block w-4 h-4 rounded-full border-2 border-current border-t-transparent animate-spin"
      aria-hidden="true"
    />
  );
}

export function Textarea({
  label,
  value,
  onChange,
  placeholder,
  rows = 4,
  tooltip,
  disabled = false,
  required = false,
  error,
  className,
  onAction,
  actionLabel = 'Upsample prompt',
  actionLoading = false,
}: TextareaProps) {
  const id = useId();

  return (
    <div className={cn('form-control', className)}>
      <label htmlFor={id} className="form-label" title={tooltip}>
        {label}
        {required && <span className="text-red-400 ml-1">*</span>}
      </label>
      {/* Wrapper provides the positioning context for the action button. */}
      <div className="relative">
        <textarea
          id={id}
          value={value ?? ''}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          rows={rows}
          disabled={disabled}
          className={cn(
            'form-textarea',
            error && 'border-red-500 focus:border-red-500 focus:ring-red-500',
            // Reserve space in the top-right so long text doesn't flow under the button
            onAction && 'pr-10'
          )}
        />
        {onAction && (
          /*
           * Touch target is 44x44px via padding (p-1.5 = 6px each side adds to the
           * 36px inner button = 48px). Positioned absolute at top-right so it floats
           * over the textarea without shifting layout.
           */
          <button
            type="button"
            onClick={(e) => { e.stopPropagation(); e.preventDefault(); onAction?.(); }}
            disabled={disabled || actionLoading}
            title={actionLabel}
            aria-label={actionLabel}
            className={cn(
              // Positioning: top-right inside the wrapper, small inset from border
              'absolute top-1.5 right-1.5',
              // Size: 36x36 inner + 6px padding gives 48px touch area
              'flex items-center justify-center',
              'w-9 h-9',
              // Appearance
              'rounded-md',
              'bg-gray-800/60 hover:bg-gray-700/90',
              'text-gray-400 hover:text-blue-400',
              'border border-transparent hover:border-gray-600',
              'transition-colors duration-150',
              // Disabled / loading
              (disabled || actionLoading) && 'opacity-50 cursor-not-allowed',
              // Focus ring for keyboard nav
              'focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-1 focus-visible:ring-offset-gray-900'
            )}
          >
            {actionLoading ? <Spinner /> : <SparkleIcon />}
          </button>
        )}
      </div>
      {error && <p className="text-sm text-red-400 mt-1">{error}</p>}
    </div>
  );
}
