/**
 * Progress Bar Component
 *
 * Animated progress indicator for generation.
 * Supports both determinate (percentage) and indeterminate states.
 *
 * last updated: 2026-01-25
 */

import { generationProgress, progressMessage, loadingState, currentSchema } from "@/core/state.ts";
import { createElement, setVisible } from "@/core/dom-utils.ts";

/**
 * Create a progress bar element with auto-updates
 */
export function createProgressBar(): HTMLElement {
  const container = createElement("div", {
    className: "hidden mt-4",
  });

  // Header row
  const header = createElement("div", {
    className: "flex items-center justify-between mb-2",
  });

  const label = createElement("span", {
    className: "text-sm text-gray-400",
  });
  label.textContent = "Generating...";

  const percent = createElement("span", {
    className: "text-sm font-mono text-gray-400",
  });
  percent.textContent = "0%";

  header.appendChild(label);
  header.appendChild(percent);
  container.appendChild(header);

  // Progress bar track
  const track = createElement("div", {
    className: "progress-bar",
  });

  const fill = createElement("div", {
    className: "fill",
  });
  fill.style.width = "0%";

  track.appendChild(fill);
  container.appendChild(track);

  // Subscribe to state updates
  loadingState.subscribe((state) => {
    setVisible(container, state === "generating");

    if (state === "generating") {
      // Apply pipeline color to progress bar
      const schema = currentSchema.get();
      if (schema) {
        fill.className = `fill bg-pipeline-${schema.color}`;
      }
    }
  });

  generationProgress.subscribe((progress) => {
    fill.style.width = `${progress}%`;
    percent.textContent = `${Math.round(progress)}%`;
  });

  progressMessage.subscribe((message) => {
    label.textContent = message || "Generating...";
  });

  return container;
}

/**
 * Create an indeterminate progress indicator (spinner)
 */
export function createSpinner(size: "sm" | "md" | "lg" = "md"): SVGSVGElement {
  const sizes = {
    sm: "w-4 h-4",
    md: "w-6 h-6",
    lg: "w-8 h-8",
  };

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", `${sizes[size]} animate-spin`);
  svg.setAttribute("fill", "none");
  svg.setAttribute("viewBox", "0 0 24 24");

  const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  circle.setAttribute("class", "opacity-25");
  circle.setAttribute("cx", "12");
  circle.setAttribute("cy", "12");
  circle.setAttribute("r", "10");
  circle.setAttribute("stroke", "currentColor");
  circle.setAttribute("stroke-width", "4");
  svg.appendChild(circle);

  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("class", "opacity-75");
  path.setAttribute("fill", "currentColor");
  path.setAttribute(
    "d",
    "M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
  );
  svg.appendChild(path);

  return svg;
}

/**
 * Create a step-based progress indicator
 */
export function createStepProgress(steps: string[]): HTMLElement {
  const container = createElement("div", {
    className: "flex items-center gap-2",
  });

  for (let i = 0; i < steps.length; i++) {
    // Step circle
    const circle = createElement("div", {
      className: "w-6 h-6 rounded-full border-2 border-gray-600 flex items-center justify-center text-xs",
      "data-step": String(i),
    });
    circle.textContent = String(i + 1);
    container.appendChild(circle);

    // Connector line (except after last step)
    if (i < steps.length - 1) {
      const line = createElement("div", {
        className: "flex-1 h-0.5 bg-gray-600",
        "data-connector": String(i),
      });
      container.appendChild(line);
    }
  }

  return container;
}

/**
 * Update step progress indicator
 */
export function updateStepProgress(container: HTMLElement, currentStep: number, completed: boolean = false): void {
  const circles = container.querySelectorAll("[data-step]");
  const connectors = container.querySelectorAll("[data-connector]");

  circles.forEach((circle, i) => {
    if (i < currentStep || completed) {
      // Completed step
      circle.className = "w-6 h-6 rounded-full bg-green-500 flex items-center justify-center text-xs text-white";
      circle.textContent = "✓";
    } else if (i === currentStep && !completed) {
      // Current step
      circle.className = "w-6 h-6 rounded-full border-2 border-blue-500 flex items-center justify-center text-xs text-blue-500";
      circle.textContent = String(i + 1);
    } else {
      // Future step
      circle.className = "w-6 h-6 rounded-full border-2 border-gray-600 flex items-center justify-center text-xs text-gray-500";
      circle.textContent = String(i + 1);
    }
  });

  connectors.forEach((connector, i) => {
    if (i < currentStep || completed) {
      connector.className = "flex-1 h-0.5 bg-green-500";
    } else {
      connector.className = "flex-1 h-0.5 bg-gray-600";
    }
  });
}
