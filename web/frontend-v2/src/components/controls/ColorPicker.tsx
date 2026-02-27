/**
 * ColorPicker Control
 *
 * Color selection input with preview.
 */

import { useId } from 'react';
import { cn } from '@/utils';

interface ColorPickerProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  tooltip?: string;
  disabled?: boolean;
  className?: string;
}

export function ColorPicker({
  label,
  value,
  onChange,
  tooltip,
  disabled = false,
  className,
}: ColorPickerProps) {
  const id = useId();

  return (
    <div className={cn('form-control', className)}>
      <label htmlFor={id} className="form-label" title={tooltip}>
        {label}
      </label>
      <div className="flex items-center gap-3">
        <input
          type="color"
          id={id}
          value={value ?? '#ffffff'}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          className="w-10 h-10 rounded-lg border border-gray-700 cursor-pointer
                     disabled:opacity-50 disabled:cursor-not-allowed"
        />
        <input
          type="text"
          value={value ?? '#ffffff'}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          placeholder="#ffffff"
          className="form-input flex-1 font-mono text-sm"
        />
      </div>
    </div>
  );
}
