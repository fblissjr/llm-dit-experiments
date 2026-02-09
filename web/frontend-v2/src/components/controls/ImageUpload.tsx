/**
 * Image Upload
 *
 * Shared component for uploading reference/input images.
 * Supports drag-and-drop, validation, and preview.
 * Reports image dimensions via onDimensionsChange for auto-matching output size.
 */

import { useState, useRef, useCallback, useEffect, useMemo } from 'react';

interface ImageUploadProps {
  value: string | string[] | null;  // Base64 or URL(s)
  onChange: (value: string | string[] | null) => void;
  onDimensionsChange?: (dimensions: { width: number; height: number } | null) => void;
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
  onDimensionsChange,
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
  const [allDimensions, setAllDimensions] = useState<({ width: number; height: number } | null)[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  // Stabilize images array -- without useMemo, a string value creates a new [value]
  // array every render, which triggers the dimension-loading effect in an infinite loop.
  const images = useMemo(
    () => Array.isArray(value) ? value : value ? [value] : [],
    [value]
  );

  // Keep a ref to the callback so the effect doesn't re-fire when the parent
  // passes an unstable onDimensionsChange identity.
  const onDimensionsChangeRef = useRef(onDimensionsChange);
  onDimensionsChangeRef.current = onDimensionsChange;

  // Read dimensions of all images when images change
  useEffect(() => {
    if (images.length === 0) {
      setAllDimensions([]);
      onDimensionsChangeRef.current?.(null);
      return;
    }

    const loadDimensions = async () => {
      const dims: ({ width: number; height: number } | null)[] = [];
      for (const src of images) {
        const d = await new Promise<{ width: number; height: number } | null>((resolve) => {
          const img = new Image();
          img.onload = () => {
            const width = Math.round(img.width / 16) * 16;
            const height = Math.round(img.height / 16) * 16;
            resolve({ width, height });
          };
          img.onerror = () => resolve(null);
          img.src = src;
        });
        dims.push(d);
      }
      setAllDimensions(dims);
      onDimensionsChangeRef.current?.(dims[0] ?? null);
    };
    loadDimensions();
  }, [images]);

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

  const canAddMore = images.length < maxCount;

  // Extract image files from clipboard data (sync path for desktop browsers).
  // Returns the files found, or empty array if no images present.
  const extractImagesFromClipboard = useCallback(
    (items: DataTransferItemList | undefined): File[] => {
      if (!items) return [];
      const files: File[] = [];
      for (let i = 0; i < items.length; i++) {
        if (items[i].type.startsWith('image/')) {
          const file = items[i].getAsFile();
          if (file) files.push(file);
        }
      }
      return files;
    },
    []
  );

  // Async clipboard API fallback (iOS Safari: "Copy Photo" / "Copy Subject").
  const extractImagesFromAsyncClipboard = useCallback(async (): Promise<File[]> => {
    if (typeof navigator?.clipboard?.read !== 'function') return [];
    try {
      const clipboardItems = await navigator.clipboard.read();
      const files: File[] = [];
      for (const clipboardItem of clipboardItems) {
        for (const type of clipboardItem.types) {
          if (type.startsWith('image/')) {
            const blob = await clipboardItem.getType(type);
            const ext = type.split('/')[1] || 'png';
            files.push(new File([blob], `pasted-image.${ext}`, { type }));
          }
        }
      }
      return files;
    } catch {
      return [];
    }
  }, []);

  // Shared paste processing logic used by both the React onPaste and the
  // document-level native listener.
  const processPaste = useCallback(
    async (items: DataTransferItemList | undefined): Promise<boolean> => {
      if (disabled || !canAddMore) return false;

      // Desktop browsers (Chrome, Firefox): images available synchronously
      const syncFiles = extractImagesFromClipboard(items);
      if (syncFiles.length > 0) {
        const dt = new DataTransfer();
        syncFiles.forEach((f) => dt.items.add(f));
        handleFiles(dt.files);
        return true;
      }

      // iOS Safari fallback via async Clipboard API
      const asyncFiles = await extractImagesFromAsyncClipboard();
      if (asyncFiles.length > 0) {
        const dt = new DataTransfer();
        asyncFiles.forEach((f) => dt.items.add(f));
        handleFiles(dt.files);
        return true;
      }

      return false;
    },
    [disabled, canAddMore, extractImagesFromClipboard, extractImagesFromAsyncClipboard, handleFiles]
  );

  // React synthetic paste handler (fires when the component's div has focus)
  const handlePaste = useCallback(
    async (e: React.ClipboardEvent) => {
      const handled = await processPaste(e.clipboardData?.items);
      if (handled) e.preventDefault();
    },
    [processPaste]
  );

  // Document-level paste listener so paste works from anywhere on the page,
  // matching the behavior of Discord/Slack/etc. Users don't need to click
  // the upload area first.
  useEffect(() => {
    const handler = async (e: ClipboardEvent) => {
      // Don't intercept paste when user is typing in an input/textarea
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || (e.target as HTMLElement)?.isContentEditable) {
        return;
      }

      const handled = await processPaste(e.clipboardData?.items);
      if (handled) e.preventDefault();
    };

    document.addEventListener('paste', handler);
    return () => document.removeEventListener('paste', handler);
  }, [processPaste]);

  const removeImage = (index: number) => {
    const newImages = images.filter((_, i) => i !== index);
    if (maxCount === 1) {
      onChange(null);
    } else {
      onChange(newImages.length > 0 ? newImages : null);
    }
  };

  return (
    <div
      className={`form-control ${className}`}
      onPaste={handlePaste}
      tabIndex={0}
      style={{ outline: 'none' }}
    >
      <label className="form-label" title={tooltip}>
        {label}
        {maxCount > 1 && (
          <span className="text-gray-500 font-normal ml-2">
            ({images.length}/{maxCount})
          </span>
        )}
      </label>

      {/* Preview existing images with per-image dimensions */}
      {images.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-2">
          {images.map((img, index) => {
            const dims = allDimensions[index];
            return (
              <div key={index} className="flex flex-col items-center gap-1">
                <div
                  className="relative group w-20 h-20 rounded-lg overflow-hidden border border-gray-700"
                >
                  <img
                    src={img}
                    alt={`Upload ${index + 1}`}
                    className="w-full h-full object-cover"
                  />
                  {/* Remove button - always visible on mobile for accessibility */}
                  <button
                    type="button"
                    onClick={() => removeImage(index)}
                    className="absolute -top-2 -right-2 w-7 h-7 bg-gray-800 border border-gray-600 rounded-full flex items-center justify-center text-gray-300 hover:text-white hover:bg-red-600 hover:border-red-500 active:bg-red-700 transition-colors opacity-100 md:opacity-0 md:group-hover:opacity-100 z-10"
                    title="Remove image"
                  >
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
                {dims && (
                  <span className="text-[10px] text-gray-500">
                    {dims.width}x{dims.height}
                  </span>
                )}
              </div>
            );
          })}
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
              Drop or paste image here, or <span className="text-blue-500">browse</span>
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
