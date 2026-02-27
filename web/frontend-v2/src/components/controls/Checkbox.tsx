/**
 * Checkbox Control
 *
 * Boolean toggle with label.
 */

import { useId } from 'react';
import { cn } from '@/utils';

interface CheckboxProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  tooltip?: string;
  disabled?: boolean;
  className?: string;
}

export function Checkbox({
  label,
  checked,
  onChange,
  tooltip,
  disabled = false,
  className,
}: CheckboxProps) {
  const id = useId();

  return (
    <div className={cn('flex items-center gap-3', className)}>
      <input
        type="checkbox"
        id={id}
        checked={checked ?? false}
        onChange={(e) => onChange(e.target.checked)}
        disabled={disabled}
        className="form-checkbox"
      />
      <label
        htmlFor={id}
        className="form-label cursor-pointer select-none"
        title={tooltip}
      >
        {label}
      </label>
    </div>
  );
}
