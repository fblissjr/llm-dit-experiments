/**
 * Prompt Textarea
 *
 * Shared textarea for prompts with configurable max length and placeholder.
 */

import { useId } from 'react';

interface PromptTextareaProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  rows?: number;
  maxLength?: number;
  label?: string;
  tooltip?: string;
  disabled?: boolean;
  required?: boolean;
  className?: string;
}

export function PromptTextarea({
  value,
  onChange,
  placeholder = 'Describe what you want to generate...',
  rows = 4,
  maxLength,
  label = 'Prompt',
  tooltip,
  disabled = false,
  required = false,
  className = '',
}: PromptTextareaProps) {
  const id = useId();

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = e.target.value;
    if (maxLength && newValue.length > maxLength) {
      return;
    }
    onChange(newValue);
  };

  return (
    <div className={`form-control ${className}`}>
      <div className="flex items-center justify-between">
        <label
          htmlFor={id}
          className="form-label"
          title={tooltip}
        >
          {label}
          {required && <span className="text-red-500 ml-1">*</span>}
        </label>
        {maxLength && (
          <span className="text-xs text-gray-500">
            {value.length}/{maxLength}
          </span>
        )}
      </div>
      <textarea
        id={id}
        value={value}
        onChange={handleChange}
        placeholder={placeholder}
        rows={rows}
        disabled={disabled}
        required={required}
        className="form-textarea"
      />
    </div>
  );
}
