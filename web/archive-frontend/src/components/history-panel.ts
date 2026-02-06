/**
 * History Panel Component
 *
 * Displays generation history with thumbnails.
 * Works as bottom sheet on mobile, sidebar on desktop.
 *
 * last updated: 2026-01-25
 */

import type { HistoryItem } from "@/types/index.ts";
import { history, formValues, selectPipeline, currentPipeline } from "@/core/state.ts";
import { createElement, clearElement } from "@/core/dom-utils.ts";

/**
 * Render the history panel content
 */
export function renderHistoryPanel(container: HTMLElement): void {
  // Subscribe to history changes
  history.subscribe((items) => {
    renderHistoryItems(container, items);
  });
}

/**
 * Render history items into the container
 */
function renderHistoryItems(container: HTMLElement, items: HistoryItem[]): void {
  clearElement(container);

  if (items.length === 0) {
    const empty = createElement("div", {
      className: "text-center text-gray-500 py-8 col-span-full",
    });
    empty.textContent = "No generations yet";
    container.appendChild(empty);
    return;
  }

  for (const item of items) {
    const historyItem = renderHistoryItem(item);
    container.appendChild(historyItem);
  }
}

/**
 * Render a single history item
 */
function renderHistoryItem(item: HistoryItem): HTMLElement {
  const wrapper = createElement("div", {
    className: "history-item",
    "data-history-id": item.id,
  });

  // Thumbnail
  if (item.type === "video") {
    const video = createElement("video", {
      className: "w-full h-24 object-cover",
    }) as HTMLVideoElement;
    video.src = item.url;
    video.muted = true;
    video.loop = true;
    // Play on hover
    wrapper.addEventListener("mouseenter", () => video.play());
    wrapper.addEventListener("mouseleave", () => video.pause());
    wrapper.appendChild(video);
  } else {
    const img = createElement("img", {
      className: "w-full h-24 object-cover",
    }) as HTMLImageElement;
    img.src = item.thumbnail_url ?? item.url;
    img.alt = item.prompt.slice(0, 50);
    wrapper.appendChild(img);
  }

  // Overlay with info
  const overlay = createElement("div", { className: "overlay" });

  const info = createElement("div", { className: "text-xs text-white truncate w-full" });
  info.textContent = item.prompt.slice(0, 30) + (item.prompt.length > 30 ? "..." : "");
  overlay.appendChild(info);

  wrapper.appendChild(overlay);

  // Type badge
  const badge = createElement("span", {
    className: `absolute top-1 right-1 px-1.5 py-0.5 text-xs rounded ${
      item.type === "video" ? "bg-purple-500/80" :
      item.type === "layers" ? "bg-pink-500/80" : "bg-blue-500/80"
    }`,
  });
  badge.textContent = item.type === "video" ? "🎬" : item.type === "layers" ? "🎭" : "🖼️";
  wrapper.appendChild(badge);

  // Click to restore params
  wrapper.addEventListener("click", () => {
    // Switch to the pipeline used for this item
    if (item.pipeline !== currentPipeline.get()) {
      selectPipeline(item.pipeline);
    }

    // Restore form values
    formValues.set(item.params);
  });

  return wrapper;
}

/**
 * Format a timestamp for display
 */
export function formatTimestamp(isoString: string): string {
  const date = new Date(isoString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);

  if (diffMins < 1) return "Just now";
  if (diffMins < 60) return `${diffMins}m ago`;

  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;

  return date.toLocaleDateString();
}
