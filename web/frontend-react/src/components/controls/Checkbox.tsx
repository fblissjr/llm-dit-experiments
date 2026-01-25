/**
 * Checkbox
 *
 * Toggle for boolean values. Used for feature toggles.
 */

import { useId } from 'react';

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
  className = '',
}: CheckboxProps) {
  const id = useId();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(e.target.checked);
  };

  return (
    <div className={`flex items-center gap-3 ${className}`}>
      <input
        type="checkbox"
        id={id}
        checked={checked}
        onChange={handleChange}
        disabled={disabled}
        className="w-4 h-4 rounded border-gray-600 bg-gray-700 text-blue-500 focus:ring-blue-500 focus:ring-offset-gray-900 cursor-pointer"
      />
      <label
        htmlFor={id}
        className="text-sm text-gray-300 cursor-pointer"
        title={tooltip}
      >
        {label}
      </label>
    </div>
  );
}
