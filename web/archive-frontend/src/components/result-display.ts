/**
 * Result Display Component
 *
 * Displays generated images, videos, or layer compositions.
 * Supports zoom, download, and copy-to-clipboard.
 *
 * last updated: 2026-01-25
 */

import { lastResultUrl, lastResultType, currentSchema } from "@/core/state.ts";
import { createElement, setVisible, Icons } from "@/core/dom-utils.ts";

/**
 * Create a result display container with auto-updates
 */
export function createResultDisplay(): HTMLElement {
  const container = createElement("div", {
    className: "hidden mt-6",
  });

  // Result frame
  const frame = createElement("div", {
    className: "result-container",
  });

  // Image element (hidden by default)
  const image = createElement("img", {
    className: "result-image hidden cursor-zoom-in",
    alt: "Generated result",
  }) as HTMLImageElement;

  // Video element (hidden by default)
  const video = createElement("video", {
    className: "result-video hidden",
  }) as HTMLVideoElement;
  video.controls = true;
  video.loop = true;

  frame.appendChild(image);
  frame.appendChild(video);
  container.appendChild(frame);

  // Action bar
  const actionBar = createElement("div", {
    className: "flex items-center justify-between mt-3",
  });

  const info = createElement("span", {
    className: "text-sm text-gray-400",
  });

  const actions = createElement("div", {
    className: "flex gap-2",
  });

  // Download button
  const downloadBtn = createElement("button", {
    className: "btn-secondary text-sm",
  });
  downloadBtn.appendChild(Icons.download());
  downloadBtn.appendChild(document.createTextNode(" Download"));

  downloadBtn.addEventListener("click", () => {
    const url = lastResultUrl.get();
    if (!url) return;

    const a = document.createElement("a");
    a.href = url;
    a.download = url.split("/").pop() ?? "result";
    a.click();
  });

  actions.appendChild(downloadBtn);
  actionBar.appendChild(info);
  actionBar.appendChild(actions);
  container.appendChild(actionBar);

  // Lightbox overlay for zoom
  const lightbox = createLightbox();
  container.appendChild(lightbox.overlay);

  // Image click to zoom
  image.addEventListener("click", () => {
    lightbox.show(image.src, "image");
  });

  // Subscribe to state updates
  lastResultUrl.subscribe((url) => {
    if (url) {
      setVisible(container, true);
      const type = lastResultType.get();
      const schema = currentSchema.get();

      if (type === "video") {
        setVisible(image, false);
        setVisible(video, true);
        video.src = url;
        video.load();
        info.textContent = "Video generated";
      } else if (type === "layers") {
        setVisible(video, false);
        setVisible(image, true);
        image.src = url;
        info.textContent = "Layers generated";
      } else {
        setVisible(video, false);
        setVisible(image, true);
        image.src = url;
        info.textContent = "Image generated";
      }

      // Apply pipeline color to frame border
      if (schema) {
        frame.className = `result-container pipeline-${schema.color} pipeline-border`;
      }
    } else {
      setVisible(container, false);
    }
  });

  return container;
}

/**
 * Create a lightbox for zoomed image viewing
 */
function createLightbox(): {
  overlay: HTMLElement;
  show: (src: string, type: "image" | "video") => void;
  hide: () => void;
} {
  const overlay = createElement("div", {
    className: "fixed inset-0 z-50 hidden bg-black/90 flex items-center justify-center cursor-zoom-out",
  });

  const content = createElement("div", {
    className: "max-w-[90vw] max-h-[90vh]",
  });

  const img = createElement("img", {
    className: "max-w-full max-h-[90vh] object-contain",
  }) as HTMLImageElement;

  content.appendChild(img);
  overlay.appendChild(content);

  // Close on click
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay || e.target === content || e.target === img) {
      hide();
    }
  });

  // Close on escape
  const handleKeydown = (e: KeyboardEvent): void => {
    if (e.key === "Escape") {
      hide();
    }
  };

  function show(src: string, type: "image" | "video"): void {
    if (type === "image") {
      img.src = src;
      setVisible(overlay, true);
      document.addEventListener("keydown", handleKeydown);
      document.body.style.overflow = "hidden";
    }
  }

  function hide(): void {
    setVisible(overlay, false);
    document.removeEventListener("keydown", handleKeydown);
    document.body.style.overflow = "";
  }

  return { overlay, show, hide };
}

/**
 * Create a layer gallery for layer decomposition results
 */
export function createLayerGallery(layerUrls: string[]): HTMLElement {
  const gallery = createElement("div", {
    className: "grid grid-cols-2 md:grid-cols-4 gap-2 mt-4",
  });

  layerUrls.forEach((url, index) => {
    const layerCard = createElement("div", {
      className: "relative group",
    });

    const img = createElement("img", {
      className: "w-full aspect-square object-cover rounded-lg border border-gray-700",
    }) as HTMLImageElement;
    img.src = url;
    img.alt = `Layer ${index + 1}`;

    const label = createElement("span", {
      className: "absolute bottom-1 left-1 px-2 py-0.5 text-xs bg-black/70 rounded",
    });
    label.textContent = `Layer ${index + 1}`;

    layerCard.appendChild(img);
    layerCard.appendChild(label);
    gallery.appendChild(layerCard);
  });

  return gallery;
}

/**
 * Create a comparison slider for before/after images
 */
export function createComparisonSlider(
  beforeUrl: string,
  afterUrl: string,
  labels: { before: string; after: string } = { before: "Before", after: "After" }
): HTMLElement {
  const container = createElement("div", {
    className: "relative overflow-hidden rounded-lg select-none",
  });

  // Before image (full width)
  const beforeImg = createElement("img", {
    className: "w-full",
  }) as HTMLImageElement;
  beforeImg.src = beforeUrl;

  // After image (clipped)
  const afterWrapper = createElement("div", {
    className: "absolute inset-0 overflow-hidden",
  });
  afterWrapper.style.clipPath = "inset(0 50% 0 0)";

  const afterImg = createElement("img", {
    className: "w-full h-full object-cover",
  }) as HTMLImageElement;
  afterImg.src = afterUrl;

  afterWrapper.appendChild(afterImg);

  // Slider handle
  const handle = createElement("div", {
    className: "absolute top-0 bottom-0 w-1 bg-white cursor-ew-resize",
  });
  handle.style.left = "50%";
  handle.style.transform = "translateX(-50%)";

  const handleCircle = createElement("div", {
    className: "absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-white rounded-full flex items-center justify-center shadow-lg",
  });
  handleCircle.textContent = "⟷";
  handle.appendChild(handleCircle);

  // Labels
  const beforeLabel = createElement("span", {
    className: "absolute top-2 left-2 px-2 py-1 text-xs bg-black/70 rounded",
  });
  beforeLabel.textContent = labels.before;

  const afterLabel = createElement("span", {
    className: "absolute top-2 right-2 px-2 py-1 text-xs bg-black/70 rounded",
  });
  afterLabel.textContent = labels.after;

  container.appendChild(beforeImg);
  container.appendChild(afterWrapper);
  container.appendChild(handle);
  container.appendChild(beforeLabel);
  container.appendChild(afterLabel);

  // Drag interaction
  let isDragging = false;

  const updatePosition = (clientX: number): void => {
    const rect = container.getBoundingClientRect();
    const x = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    afterWrapper.style.clipPath = `inset(0 ${(1 - x) * 100}% 0 0)`;
    handle.style.left = `${x * 100}%`;
  };

  handle.addEventListener("mousedown", () => {
    isDragging = true;
  });

  document.addEventListener("mousemove", (e) => {
    if (isDragging) {
      updatePosition(e.clientX);
    }
  });

  document.addEventListener("mouseup", () => {
    isDragging = false;
  });

  // Touch support
  handle.addEventListener("touchstart", () => {
    isDragging = true;
  });

  container.addEventListener("touchmove", (e) => {
    if (isDragging && e.touches[0]) {
      updatePosition(e.touches[0].clientX);
    }
  });

  document.addEventListener("touchend", () => {
    isDragging = false;
  });

  return container;
}
