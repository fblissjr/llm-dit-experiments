/**
 * Form Builder Component
 *
 * Renders a complete form from a PipelineSchema.
 * Uses Web Components pattern for each param type.
 *
 * last updated: 2026-01-25
 */

import type { PipelineSchema, ParamSchema, GroupType } from "@/types/index.ts";
import { formValues, setFormValue, expandedGroups, toggleGroup } from "@/core/state.ts";
import { createElement, Icons, clearElement, setVisible, setClass } from "@/core/dom-utils.ts";
import { isParamVisible } from "@/types/schema.ts";

/**
 * Render a complete form from a pipeline schema
 */
export function renderForm(container: HTMLElement, schema: PipelineSchema): void {
  clearElement(container);

  // Apply pipeline color class
  container.className = `pipeline-${schema.color} space-y-6`;

  // Group params by their group type
  const groups: Record<GroupType, ParamSchema[]> = {
    basic: [],
    advanced: [],
    expert: [],
    scheduler: [],
    optimization: [],
    enhancement: [],
  };

  for (const param of schema.params) {
    groups[param.group].push(param);
  }

  // Render each group
  const groupOrder: GroupType[] = ["basic", "scheduler", "advanced", "optimization", "enhancement", "expert"];
  for (const groupType of groupOrder) {
    const params = groups[groupType];
    if (params.length === 0) continue;

    const groupElement = renderGroup(groupType, params, schema);
    container.appendChild(groupElement);
  }
}

/**
 * Render a group of parameters
 */
function renderGroup(groupType: GroupType, params: ParamSchema[], schema: PipelineSchema): HTMLElement {
  const group = createElement("div", { className: "card" });

  // Group header (collapsible for non-basic groups)
  const isCollapsible = groupType !== "basic";
  const header = createElement("div", {
    className: `group-header ${isCollapsible ? "cursor-pointer" : ""}`,
  });

  const headerText = createElement("span", {}, formatGroupName(groupType));
  header.appendChild(headerText);

  if (isCollapsible) {
    header.appendChild(Icons.chevronDown());

    // Subscribe to expanded state
    expandedGroups.subscribe((groups) => {
      const isExpanded = groups.has(groupType);
      setClass(header, "expanded", isExpanded);
      setVisible(content, isExpanded);
    });

    header.addEventListener("click", () => toggleGroup(groupType));
  }

  group.appendChild(header);

  // Group content
  const content = createElement("div", {
    className: "space-y-4 mt-4",
  });

  // Render each param
  for (const param of params) {
    const paramElement = renderParam(param, schema);
    content.appendChild(paramElement);
  }

  group.appendChild(content);

  // Set initial visibility
  const isExpanded = expandedGroups.get().has(groupType);
  setVisible(content, isExpanded);
  setClass(header, "expanded", isExpanded);

  return group;
}

/**
 * Render a single parameter control
 */
function renderParam(param: ParamSchema, _schema: PipelineSchema): HTMLElement {
  const wrapper = createElement("div", {
    className: "form-control",
    "data-param-id": param.id,
  });

  // Handle conditional visibility
  if (param.conditional) {
    formValues.subscribe((values) => {
      const visible = isParamVisible(param, values);
      setVisible(wrapper, visible);
    });
  }

  // Label with optional tooltip
  const label = createElement("label", {
    className: "form-label",
  });
  label.textContent = param.label;

  if (param.required) {
    const requiredMark = createElement("span", { className: "text-red-400" });
    requiredMark.textContent = "*";
    label.appendChild(requiredMark);
  }

  if (param.tooltip) {
    const tooltipContainer = createElement("div", {
      className: "relative has-tooltip inline-block ml-1",
    });
    const questionMark = createElement("span", {
      className: "text-gray-500 cursor-help",
    });
    questionMark.textContent = "?";
    const tooltip = createElement("div", {
      className: "tooltip left-0 top-full mt-1 w-48",
    });
    tooltip.textContent = param.tooltip;
    tooltipContainer.appendChild(questionMark);
    tooltipContainer.appendChild(tooltip);
    label.appendChild(tooltipContainer);
  }

  wrapper.appendChild(label);

  // Render the appropriate control type
  let control: HTMLElement;
  switch (param.type) {
    case "textarea":
      control = renderTextarea(param);
      break;
    case "slider":
      control = renderSlider(param);
      break;
    case "number":
      control = renderNumber(param);
      break;
    case "checkbox":
      control = renderCheckbox(param);
      break;
    case "select":
      control = renderSelect(param);
      break;
    case "image":
      control = renderImageUpload(param);
      break;
    default:
      control = renderTextarea(param);
  }

  wrapper.appendChild(control);
  return wrapper;
}

/**
 * Render a textarea control
 */
function renderTextarea(param: ParamSchema): HTMLElement {
  const textarea = createElement("textarea", {
    className: "form-textarea",
    id: `param-${param.id}`,
  }) as HTMLTextAreaElement;

  if (param.placeholder) {
    textarea.placeholder = param.placeholder;
  }
  if (param.rows) {
    textarea.rows = param.rows;
  }
  if (param.default !== undefined) {
    textarea.value = String(param.default);
  }

  textarea.addEventListener("input", () => {
    setFormValue(param.id, textarea.value);
  });

  // Sync with state
  formValues.subscribe((values) => {
    const value = values[param.id];
    if (value !== undefined && textarea.value !== String(value)) {
      textarea.value = String(value);
    }
  });

  return textarea;
}

/**
 * Render a slider control with value display
 */
function renderSlider(param: ParamSchema): HTMLElement {
  const container = createElement("div", { className: "slider-container" });

  const slider = createElement("input", {
    className: "slider-track",
    type: "range",
  }) as HTMLInputElement;

  slider.min = String(param.min ?? 0);
  slider.max = String(param.max ?? 100);
  slider.step = String(param.step ?? 1);
  slider.value = String(param.default ?? param.min ?? 0);

  const valueDisplay = createElement("span", { className: "slider-value" });
  const initialValue = typeof param.default === "number" ? param.default : (param.min ?? 0);
  valueDisplay.textContent = formatValue(initialValue, param.step);

  slider.addEventListener("input", () => {
    const value = parseFloat(slider.value);
    setFormValue(param.id, value);
    valueDisplay.textContent = formatValue(value, param.step);
  });

  // Sync with state
  formValues.subscribe((values) => {
    const value = values[param.id];
    if (value !== undefined && slider.value !== String(value)) {
      slider.value = String(value);
      valueDisplay.textContent = formatValue(value as number, param.step);
    }
  });

  container.appendChild(slider);
  container.appendChild(valueDisplay);
  return container;
}

/**
 * Render a number input
 */
function renderNumber(param: ParamSchema): HTMLElement {
  const input = createElement("input", {
    className: "form-input",
    type: "number",
  }) as HTMLInputElement;

  if (param.min !== undefined) input.min = String(param.min);
  if (param.max !== undefined) input.max = String(param.max);
  if (param.step !== undefined) input.step = String(param.step);
  if (param.default !== undefined) input.value = String(param.default);

  input.addEventListener("input", () => {
    const value = parseFloat(input.value);
    if (!isNaN(value)) {
      setFormValue(param.id, value);
    }
  });

  // Sync with state
  formValues.subscribe((values) => {
    const value = values[param.id];
    if (value !== undefined && input.value !== String(value)) {
      input.value = String(value);
    }
  });

  return input;
}

/**
 * Render a checkbox control
 */
function renderCheckbox(param: ParamSchema): HTMLElement {
  const container = createElement("div", {
    className: "flex items-center gap-3",
  });

  const checkbox = createElement("input", {
    type: "checkbox",
    className: "w-5 h-5 rounded border-gray-600 bg-gray-700 text-blue-500 focus:ring-blue-500 focus:ring-offset-gray-900",
    id: `param-${param.id}`,
  }) as HTMLInputElement;

  if (param.default === true) {
    checkbox.checked = true;
  }

  checkbox.addEventListener("change", () => {
    setFormValue(param.id, checkbox.checked);
  });

  // Sync with state
  formValues.subscribe((values) => {
    const value = values[param.id];
    if (value !== undefined && checkbox.checked !== Boolean(value)) {
      checkbox.checked = Boolean(value);
    }
  });

  container.appendChild(checkbox);
  return container;
}

/**
 * Render a select dropdown
 */
function renderSelect(param: ParamSchema): HTMLElement {
  const select = createElement("select", {
    className: "form-input",
    id: `param-${param.id}`,
  }) as HTMLSelectElement;

  for (const option of param.options ?? []) {
    const optionEl = createElement("option", { value: option });
    optionEl.textContent = option;
    select.appendChild(optionEl);
  }

  if (param.default !== undefined) {
    select.value = String(param.default);
  }

  select.addEventListener("change", () => {
    setFormValue(param.id, select.value);
  });

  // Sync with state
  formValues.subscribe((values) => {
    const value = values[param.id];
    if (value !== undefined && select.value !== String(value)) {
      select.value = String(value);
    }
  });

  return select;
}

/**
 * Render an image upload control
 */
function renderImageUpload(param: ParamSchema): HTMLElement {
  const container = createElement("div", {
    className: "relative",
  });

  // Drop zone
  const dropZone = createElement("div", {
    className: "border-2 border-dashed border-gray-600 rounded-lg p-4 text-center cursor-pointer hover:border-gray-500 transition-colors min-h-touch",
  });

  const dropText = createElement("p", { className: "text-gray-400" });
  dropText.textContent = "Click or drag to upload image";
  dropZone.appendChild(dropText);

  // Hidden file input
  const fileInput = createElement("input", {
    type: "file",
    accept: "image/*",
    className: "hidden",
  }) as HTMLInputElement;

  // Preview image
  const preview = createElement("img", {
    className: "hidden max-h-32 mx-auto rounded-lg",
  }) as HTMLImageElement;

  // Handle file selection
  const handleFile = (file: File): void => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target?.result as string;
      preview.src = dataUrl;
      setVisible(preview, true);
      setVisible(dropText, false);
      setFormValue(param.id, dataUrl);
    };
    reader.readAsDataURL(file);
  };

  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) handleFile(file);
  });

  dropZone.addEventListener("click", () => fileInput.click());

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("border-blue-500");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("border-blue-500");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("border-blue-500");
    const file = e.dataTransfer?.files[0];
    if (file && file.type.startsWith("image/")) {
      handleFile(file);
    }
  });

  dropZone.appendChild(preview);
  container.appendChild(dropZone);
  container.appendChild(fileInput);

  return container;
}

/**
 * Format a group name for display
 */
function formatGroupName(group: GroupType): string {
  const names: Record<GroupType, string> = {
    basic: "Settings",
    advanced: "Advanced",
    expert: "Expert",
    scheduler: "Scheduler",
    optimization: "Performance",
    enhancement: "Enhancements",
  };
  return names[group];
}

/**
 * Format a numeric value for display
 */
function formatValue(value: number, step?: number): string {
  if (step && step < 1) {
    const decimals = step.toString().split(".")[1]?.length ?? 2;
    return value.toFixed(decimals);
  }
  return String(Math.round(value));
}
