/**
 * Reactive State Management
 *
 * Simple signal-like pattern for managing UI state without a framework.
 * Uses subscription pattern for reactive updates.
 *
 * last updated: 2026-01-25
 */

import type {
  PipelineSchema,
  FormValues,
  HistoryItem,
  SystemStatus,
} from "@/types/index.ts";

/**
 * Subscriber callback type
 */
type Subscriber<T> = (value: T) => void;

/**
 * Signal - a reactive value container
 */
export class Signal<T> {
  private value: T;
  private subscribers: Set<Subscriber<T>> = new Set();

  constructor(initialValue: T) {
    this.value = initialValue;
  }

  /** Get current value */
  get(): T {
    return this.value;
  }

  /** Set new value and notify subscribers */
  set(newValue: T): void {
    if (this.value !== newValue) {
      this.value = newValue;
      this.notify();
    }
  }

  /** Update value using a function */
  update(fn: (current: T) => T): void {
    this.set(fn(this.value));
  }

  /** Subscribe to value changes */
  subscribe(callback: Subscriber<T>): () => void {
    this.subscribers.add(callback);
    // Immediately call with current value
    callback(this.value);
    // Return unsubscribe function
    return () => this.subscribers.delete(callback);
  }

  private notify(): void {
    for (const subscriber of this.subscribers) {
      subscriber(this.value);
    }
  }
}

/**
 * Computed signal - derives value from other signals
 */
export class Computed<T> {
  private value: T;
  private subscribers: Set<Subscriber<T>> = new Set();

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  constructor(fn: () => T, deps: Array<Signal<any>>) {
    this.value = fn();

    // Subscribe to all dependencies
    for (const dep of deps) {
      dep.subscribe(() => {
        const newValue = fn();
        if (this.value !== newValue) {
          this.value = newValue;
          this.notify();
        }
      });
    }
  }

  get(): T {
    return this.value;
  }

  subscribe(callback: Subscriber<T>): () => void {
    this.subscribers.add(callback);
    callback(this.value);
    return () => this.subscribers.delete(callback);
  }

  private notify(): void {
    for (const subscriber of this.subscribers) {
      subscriber(this.value);
    }
  }
}

// =============================================================================
// Application State
// =============================================================================

/**
 * Application loading state
 */
export type LoadingState = "idle" | "loading" | "generating" | "error";

/**
 * All available pipeline schemas (loaded from /api/pipelines)
 */
export const pipelineSchemas = new Signal<Record<string, PipelineSchema>>({});

/**
 * Currently selected pipeline ID
 */
export const currentPipeline = new Signal<string | null>(null);

/**
 * Currently loaded pipeline on the server (may differ from selected)
 */
export const loadedPipeline = new Signal<string | null>(null);

/**
 * Current form values for the selected pipeline
 */
export const formValues = new Signal<FormValues>({});

/**
 * Loading/generating state
 */
export const loadingState = new Signal<LoadingState>("idle");

/**
 * Error message (if any)
 */
export const errorMessage = new Signal<string | null>(null);

/**
 * Generation progress (0-100)
 */
export const generationProgress = new Signal<number>(0);

/**
 * Progress message (for streaming)
 */
export const progressMessage = new Signal<string>("");

/**
 * Generation history
 */
export const history = new Signal<HistoryItem[]>([]);

/**
 * History panel visibility (for mobile bottom sheet)
 */
export const historyPanelOpen = new Signal<boolean>(false);

/**
 * System status (VRAM, loaded models, etc.)
 */
export const systemStatus = new Signal<SystemStatus | null>(null);

/**
 * Advanced settings visibility per group
 */
export const expandedGroups = new Signal<Set<string>>(new Set(["basic"]));

/**
 * Last generated result URL
 */
export const lastResultUrl = new Signal<string | null>(null);

/**
 * Last generated result type
 */
export const lastResultType = new Signal<"image" | "video" | "layers" | null>(
  null
);

// =============================================================================
// Computed Values
// =============================================================================

/**
 * Current pipeline schema (derived from currentPipeline + pipelineSchemas)
 */
export const currentSchema = new Computed<PipelineSchema | null>(
  () => {
    const id = currentPipeline.get();
    const schemas = pipelineSchemas.get();
    return id ? (schemas[id] ?? null) : null;
  },
  [currentPipeline, pipelineSchemas]
);

/**
 * Is generation in progress?
 */
export const isGenerating = new Computed<boolean>(
  () => loadingState.get() === "generating",
  [loadingState]
);

/**
 * Is the currently selected pipeline loaded on the server?
 */
export const isPipelineLoaded = new Computed<boolean>(
  () => {
    const selected = currentPipeline.get();
    const loaded = loadedPipeline.get();
    return selected !== null && selected === loaded;
  },
  [currentPipeline, loadedPipeline]
);

// =============================================================================
// State Actions
// =============================================================================

/**
 * Select a pipeline and initialize form values
 */
export function selectPipeline(pipelineId: string): void {
  const schemas = pipelineSchemas.get();
  const schema = schemas[pipelineId];

  if (!schema) {
    console.error(`Pipeline "${pipelineId}" not found`);
    return;
  }

  currentPipeline.set(pipelineId);

  // Initialize form with schema defaults
  const defaults: FormValues = {};
  for (const param of schema.params) {
    if (param.default !== undefined) {
      defaults[param.id] = param.default;
    }
  }
  formValues.set(defaults);

  // Ensure basic group is expanded
  expandedGroups.update((groups) => {
    groups.add("basic");
    return groups;
  });

  // Update URL
  const url = new URL(window.location.href);
  url.searchParams.set("pipeline", pipelineId);
  window.history.replaceState({}, "", url.toString());
}

/**
 * Update a single form value
 */
export function setFormValue(paramId: string, value: unknown): void {
  formValues.update((current) => ({
    ...current,
    [paramId]: value,
  }));
}

/**
 * Toggle a group's expanded state
 */
export function toggleGroup(group: string): void {
  expandedGroups.update((groups) => {
    const newGroups = new Set(groups);
    if (newGroups.has(group)) {
      newGroups.delete(group);
    } else {
      newGroups.add(group);
    }
    return newGroups;
  });
}

/**
 * Set error state
 */
export function setError(message: string): void {
  errorMessage.set(message);
  loadingState.set("error");
}

/**
 * Clear error state
 */
export function clearError(): void {
  errorMessage.set(null);
  if (loadingState.get() === "error") {
    loadingState.set("idle");
  }
}

/**
 * Start generation
 */
export function startGeneration(): void {
  loadingState.set("generating");
  generationProgress.set(0);
  progressMessage.set("Starting generation...");
  clearError();
}

/**
 * Update generation progress
 */
export function updateProgress(progress: number, message?: string): void {
  generationProgress.set(progress);
  if (message) {
    progressMessage.set(message);
  }
}

/**
 * Complete generation
 */
export function completeGeneration(
  resultUrl: string,
  resultType: "image" | "video" | "layers"
): void {
  loadingState.set("idle");
  generationProgress.set(100);
  progressMessage.set("Complete!");
  lastResultUrl.set(resultUrl);
  lastResultType.set(resultType);
}

/**
 * Add item to history
 */
export function addToHistory(item: HistoryItem): void {
  history.update((current) => {
    const newHistory = [item, ...current];
    // Keep last 50 items
    return newHistory.slice(0, 50);
  });
}
