/**
 * Textarea Control
 *
 * Multi-line text input for prompts and long text.
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
}: TextareaProps) {
  const id = useId();

  return (
    <div className={cn('form-control', className)}>
      <label htmlFor={id} className="form-label" title={tooltip}>
        {label}
        {required && <span className="text-red-400 ml-1">*</span>}
      </label>
      <textarea
        id={id}
        value={value ?? ''}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        rows={rows}
        disabled={disabled}
        className={cn(
          'form-textarea',
          error && 'border-red-500 focus:border-red-500 focus:ring-red-500'
        )}
      />
      {error && <p className="text-sm text-red-400 mt-1">{error}</p>}
    </div>
  );
}
