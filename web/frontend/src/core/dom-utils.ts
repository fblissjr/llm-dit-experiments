/**
 * Safe DOM Utilities
 *
 * Helper functions for creating DOM elements without innerHTML.
 * Prevents XSS vulnerabilities by using typed DOM APIs.
 *
 * last updated: 2026-01-25
 */

/**
 * Create an element with optional attributes and children
 */
export function createElement<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs?: Record<string, string>,
  ...children: (Node | string)[]
): HTMLElementTagNameMap[K] {
  const element = document.createElement(tag);

  if (attrs) {
    for (const [key, value] of Object.entries(attrs)) {
      if (key === "className") {
        element.className = value;
      } else if (key.startsWith("data-")) {
        element.dataset[key.slice(5)] = value;
      } else {
        element.setAttribute(key, value);
      }
    }
  }

  for (const child of children) {
    if (typeof child === "string") {
      element.appendChild(document.createTextNode(child));
    } else {
      element.appendChild(child);
    }
  }

  return element;
}

/**
 * Create an SVG element
 */
export function createSvg(
  attrs: Record<string, string>,
  ...paths: { d: string; attrs?: Record<string, string> }[]
): SVGSVGElement {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");

  for (const [key, value] of Object.entries(attrs)) {
    svg.setAttribute(key, value);
  }

  for (const path of paths) {
    const pathEl = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "path"
    );
    pathEl.setAttribute("d", path.d);
    if (path.attrs) {
      for (const [key, value] of Object.entries(path.attrs)) {
        pathEl.setAttribute(key, value);
      }
    }
    svg.appendChild(pathEl);
  }

  return svg;
}

/**
 * Common SVG icons used throughout the app
 */
export const Icons = {
  lightning: (): SVGSVGElement =>
    createSvg(
      {
        class: "w-5 h-5",
        fill: "none",
        stroke: "currentColor",
        viewBox: "0 0 24 24",
      },
      {
        d: "M13 10V3L4 14h7v7l9-11h-7z",
        attrs: {
          "stroke-linecap": "round",
          "stroke-linejoin": "round",
          "stroke-width": "2",
        },
      }
    ),

  spinner: (): SVGSVGElement => {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "w-5 h-5 animate-spin");
    svg.setAttribute("fill", "none");
    svg.setAttribute("viewBox", "0 0 24 24");

    const circle = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "circle"
    );
    circle.setAttribute("class", "opacity-25");
    circle.setAttribute("cx", "12");
    circle.setAttribute("cy", "12");
    circle.setAttribute("r", "10");
    circle.setAttribute("stroke", "currentColor");
    circle.setAttribute("stroke-width", "4");

    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("class", "opacity-75");
    path.setAttribute("fill", "currentColor");
    path.setAttribute(
      "d",
      "M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
    );

    svg.appendChild(circle);
    svg.appendChild(path);
    return svg;
  },

  download: (): SVGSVGElement =>
    createSvg(
      {
        class: "w-4 h-4",
        fill: "none",
        stroke: "currentColor",
        viewBox: "0 0 24 24",
      },
      {
        d: "M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4",
        attrs: {
          "stroke-linecap": "round",
          "stroke-linejoin": "round",
          "stroke-width": "2",
        },
      }
    ),

  check: (): SVGSVGElement =>
    createSvg(
      {
        class: "w-2 h-2",
        fill: "currentColor",
        viewBox: "0 0 8 8",
      },
      { d: "M0 4a4 4 0 118 0 4 4 0 01-8 0z" }
    ),

  chevronDown: (): SVGSVGElement =>
    createSvg(
      {
        class: "w-4 h-4 icon",
        fill: "none",
        stroke: "currentColor",
        viewBox: "0 0 24 24",
      },
      {
        d: "M19 9l-7 7-7-7",
        attrs: {
          "stroke-linecap": "round",
          "stroke-linejoin": "round",
          "stroke-width": "2",
        },
      }
    ),
};

/**
 * Clear all children from an element
 */
export function clearElement(element: Element): void {
  while (element.firstChild) {
    element.removeChild(element.firstChild);
  }
}

/**
 * Show/hide an element using the hidden class
 */
export function setVisible(element: Element, visible: boolean): void {
  if (visible) {
    element.classList.remove("hidden");
  } else {
    element.classList.add("hidden");
  }
}

/**
 * Add/remove a class based on a condition
 */
export function setClass(
  element: Element,
  className: string,
  add: boolean
): void {
  if (add) {
    element.classList.add(className);
  } else {
    element.classList.remove(className);
  }
}
