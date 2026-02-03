/**
 * Image Upload
 *
 * Shared component for uploading reference/input images.
 * Supports drag-and-drop, validation, and preview.
 */

import { useState, useRef, useCallback } from 'react';

interface ImageUploadProps {
  value: string | string[] | null;  // Base64 or URL(s)
  onChange: (value: string | string[] | null) => void;
  maxCount?: number;
  maxSizeMB?: number;
  acceptedFormats?: string[];
  label?: string;
  tooltip?: string;
  disabled?: boolean;
  className?: string;
}

export function ImageUpload({
  value,
  onChange,
  maxCount = 1,
  maxSizeMB = 10,
  acceptedFormats = ['image/png', 'image/jpeg', 'image/webp'],
  label = 'Image',
  tooltip,
  disabled = false,
  className = '',
}: ImageUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const images = Array.isArray(value) ? value : value ? [value] : [];

  const handleFiles = useCallback(
    async (files: FileList) => {
      setError(null);
      const newImages: string[] = [...images];

      for (let i = 0; i < files.length; i++) {
        if (newImages.length >= maxCount) {
          setError(`Maximum ${maxCount} image(s) allowed`);
          break;
        }

        const file = files[i];

        // Validate format
        if (!acceptedFormats.includes(file.type)) {
          setError(`Invalid format. Accepted: ${acceptedFormats.join(', ')}`);
          continue;
        }

        // Validate size
        if (file.size > maxSizeMB * 1024 * 1024) {
          setError(`File too large. Maximum ${maxSizeMB}MB`);
          continue;
        }

        // Convert to base64
        const base64 = await fileToBase64(file);
        newImages.push(base64);
      }

      if (maxCount === 1) {
        onChange(newImages[0] ?? null);
      } else {
        onChange(newImages.length > 0 ? newImages : null);
      }
    },
    [images, maxCount, maxSizeMB, acceptedFormats, onChange]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      if (disabled) return;
      if (e.dataTransfer.files) {
        handleFiles(e.dataTransfer.files);
      }
    },
    [disabled, handleFiles]
  );

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      handleFiles(e.target.files);
    }
  };

  const removeImage = (index: number) => {
    const newImages = images.filter((_, i) => i !== index);
    if (maxCount === 1) {
      onChange(null);
    } else {
      onChange(newImages.length > 0 ? newImages : null);
    }
  };

  const canAddMore = images.length < maxCount;

  return (
    <div className={`form-control ${className}`}>
      <label className="form-label" title={tooltip}>
        {label}
        {maxCount > 1 && (
          <span className="text-gray-500 font-normal ml-2">
            ({images.length}/{maxCount})
          </span>
        )}
      </label>

      {/* Preview existing images */}
      {images.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-2">
          {images.map((img, index) => (
            <div
              key={index}
              className="relative group w-20 h-20 rounded-lg overflow-hidden border border-gray-700"
            >
              <img
                src={img}
                alt={`Upload ${index + 1}`}
                className="w-full h-full object-cover"
              />
              <button
                type="button"
                onClick={() => removeImage(index)}
                className="absolute top-1 right-1 w-5 h-5 bg-gray-900/80 rounded-full flex items-center justify-center text-gray-300 hover:text-white hover:bg-red-600 transition-colors opacity-0 group-hover:opacity-100"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Drop zone */}
      {canAddMore && (
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => inputRef.current?.click()}
          className={`
            border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors
            ${isDragging ? 'border-blue-500 bg-blue-500/10' : 'border-gray-700 hover:border-gray-600'}
            ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
          `}
        >
          <input
            ref={inputRef}
            type="file"
            accept={acceptedFormats.join(',')}
            multiple={maxCount > 1}
            onChange={handleInputChange}
            disabled={disabled}
            className="hidden"
          />
          <div className="text-gray-400">
            <svg
              className="w-8 h-8 mx-auto mb-2"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
              />
            </svg>
            <p className="text-sm">
              Drop image here or <span className="text-blue-500">browse</span>
            </p>
            <p className="text-xs text-gray-500 mt-1">
              Max {maxSizeMB}MB · {acceptedFormats.map((f) => f.split('/')[1]).join(', ')}
            </p>
          </div>
        </div>
      )}

      {/* Error message */}
      {error && <p className="text-sm text-red-400 mt-1">{error}</p>}
    </div>
  );
}

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}
