/**
 * Select Dropdown
 *
 * For selecting from a list of options.
 */

import { useId } from 'react';

interface SelectProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: string[];
  tooltip?: string;
  disabled?: boolean;
  className?: string;
}

export function Select({
  label,
  value,
  onChange,
  options,
  tooltip,
  disabled = false,
  className = '',
}: SelectProps) {
  const id = useId();

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    onChange(e.target.value);
  };

  return (
    <div className={`form-control ${className}`}>
      <label
        htmlFor={id}
        className="form-label"
        title={tooltip}
      >
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={handleChange}
        disabled={disabled}
        className="form-input cursor-pointer"
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </div>
  );
}
