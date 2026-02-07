/**
 * PresetCategorySection - Horizontal scroll strip of cards for one category.
 *
 * Features:
 * - Scroll snap for clean card alignment
 * - Nav arrows on desktop hover (only when content overflows)
 * - Fade edges via CSS mask when scrollable
 */

import { useRef, useState, useCallback, useEffect } from 'react';
import type { GenerationPreset } from '@/api/types';
import { PresetCard } from './PresetCard';

interface PresetCategorySectionProps {
  category: string;
  presets: GenerationPreset[];
  activePresetName: string | null;
  pipelineColor: string;
  onSelectPreset: (name: string) => void;
}

export function PresetCategorySection({
  category,
  presets,
  activePresetName,
  pipelineColor,
  onSelectPreset,
}: PresetCategorySectionProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(false);
  const [isHovered, setIsHovered] = useState(false);

  const updateScrollState = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    setCanScrollLeft(el.scrollLeft > 4);
    setCanScrollRight(el.scrollLeft + el.clientWidth < el.scrollWidth - 4);
  }, []);

  useEffect(() => {
    updateScrollState();
    const el = scrollRef.current;
    if (!el) return;

    const observer = new ResizeObserver(updateScrollState);
    observer.observe(el);
    return () => observer.disconnect();
  }, [updateScrollState, presets.length]);

  const scroll = useCallback((direction: 'left' | 'right') => {
    const el = scrollRef.current;
    if (!el) return;
    const amount = direction === 'left' ? -180 : 180;
    el.scrollBy({ left: amount, behavior: 'smooth' });
  }, []);

  return (
    <div
      className="space-y-2"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Category header */}
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">
          {category}
        </span>
        <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-gray-700 text-gray-400">
          {presets.length}
        </span>
      </div>

      {/* Scroll container with nav arrows */}
      <div className="relative">
        {/* Left arrow */}
        {isHovered && canScrollLeft && (
          <button
            type="button"
            onClick={() => scroll('left')}
            className="absolute left-0 top-0 bottom-2 z-10 w-8 flex items-center justify-center
                       bg-gradient-to-r from-gray-900/90 to-transparent text-gray-300
                       hover:text-white transition-colors"
            aria-label="Scroll left"
          >
            &#8249;
          </button>
        )}

        {/* Scrollable area */}
        <div
          ref={scrollRef}
          className={`preset-scroll ${
            (canScrollLeft || canScrollRight) ? 'preset-scroll-container' : ''
          }`}
          onScroll={updateScrollState}
        >
          {presets.map((preset) => (
            <PresetCard
              key={preset.name}
              preset={preset}
              isActive={preset.name === activePresetName}
              pipelineColor={pipelineColor}
              onClick={() => onSelectPreset(preset.name)}
            />
          ))}
        </div>

        {/* Right arrow */}
        {isHovered && canScrollRight && (
          <button
            type="button"
            onClick={() => scroll('right')}
            className="absolute right-0 top-0 bottom-2 z-10 w-8 flex items-center justify-center
                       bg-gradient-to-l from-gray-900/90 to-transparent text-gray-300
                       hover:text-white transition-colors"
            aria-label="Scroll right"
          >
            &#8250;
          </button>
        )}
      </div>
    </div>
  );
}
