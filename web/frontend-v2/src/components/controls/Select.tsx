/**
 * Select Control
 *
 * Dropdown select with options from schema or dynamic endpoint.
 */

import { useId, useEffect, useState } from 'react';
import { cn } from '@/utils';

interface SelectProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: string[];
  optionsEndpoint?: string;
  tooltip?: string;
  disabled?: boolean;
  error?: string;
  className?: string;
}

export function Select({
  label,
  value,
  onChange,
  options: staticOptions,
  optionsEndpoint,
  tooltip,
  disabled = false,
  error,
  className,
}: SelectProps) {
  const id = useId();
  const [dynamicOptions, setDynamicOptions] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // Fetch dynamic options if endpoint provided
  useEffect(() => {
    if (!optionsEndpoint) return;

    setIsLoading(true);
    fetch(optionsEndpoint)
      .then((res) => res.json())
      .then((data) => {
        // Handle both array of strings and array of objects with 'name' field
        const options = Array.isArray(data.presets)
          ? data.presets.map((p: { name: string }) => p.name)
          : Array.isArray(data)
            ? data.map((item) => (typeof item === 'string' ? item : item.name))
            : [];
        setDynamicOptions(options);
      })
      .catch(() => {
        setDynamicOptions([]);
      })
      .finally(() => {
        setIsLoading(false);
      });
  }, [optionsEndpoint]);

  const options = optionsEndpoint ? dynamicOptions : staticOptions;

  return (
    <div className={cn('form-control', className)}>
      <label htmlFor={id} className="form-label" title={tooltip}>
        {label}
      </label>
      <select
        id={id}
        value={value ?? ''}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled || isLoading}
        className={cn(
          'form-select',
          error && 'border-red-500 focus:border-red-500 focus:ring-red-500'
        )}
      >
        {isLoading ? (
          <option value="">Loading...</option>
        ) : options.length === 0 ? (
          <option value="">No options available</option>
        ) : (
          options.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))
        )}
      </select>
      {error && <p className="text-sm text-red-400 mt-1">{error}</p>}
    </div>
  );
}
