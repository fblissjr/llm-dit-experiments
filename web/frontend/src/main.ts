/**
 * Main Entry Point
 *
 * Initializes the LLM-DiT Studio application.
 *
 * last updated: 2026-01-25
 */

import {
  pipelineSchemas,
  currentPipeline,
  currentSchema,
  loadedPipeline,
  loadingState,
  errorMessage,
  generationProgress,
  progressMessage,
  lastResultUrl,
  lastResultType,
  historyPanelOpen,
  selectPipeline,
  startGeneration,
  updateProgress,
  completeGeneration,
  setError,
  clearError,
  formValues,
  addToHistory,
} from "@/core/state.ts";

import {
  fetchPipelines,
  generate,
  generateVideoStream,
  getMediaUrl,
  APIClientError,
} from "@/core/api-client.ts";

import { initRouter } from "@/core/router.ts";
import { renderForm } from "@/components/form-builder.ts";
import {
  createElement,
  Icons,
  clearElement,
  setVisible,
  setClass,
} from "@/core/dom-utils.ts";

import type { SSEEvent } from "@/types/index.ts";

/**
 * Initialize the application
 */
async function init(): Promise<void> {
  console.log("LLM-DiT Studio initializing...");

  // Set up UI event listeners
  setupUIListeners();

  // Load pipeline schemas from API
  try {
    const response = await fetchPipelines();

    // Store schemas in state
    pipelineSchemas.set(response.pipelines);
    loadedPipeline.set(response.loaded_pipeline);

    // Render pipeline tabs
    renderPipelineTabs(response.pipelines);

    // Initialize router (reads pipeline from URL)
    initRouter();

    // If no pipeline selected, select the first one
    if (!currentPipeline.get()) {
      const firstPipeline = Object.keys(response.pipelines)[0];
      if (firstPipeline) {
        selectPipeline(firstPipeline);
      }
    }

    // Enable generate button
    const generateBtn = document.getElementById("generateBtn");
    if (generateBtn) {
      generateBtn.removeAttribute("disabled");
    }

    console.log(
      `Loaded ${Object.keys(response.pipelines).length} pipeline schemas`
    );
  } catch (error) {
    console.error("Failed to load pipeline schemas:", error);
    setError(
      `Failed to connect to server: ${error instanceof Error ? error.message : "Unknown error"}`
    );
  }

  // Subscribe to state changes
  setupStateSubscriptions();
}

/**
 * Set up UI event listeners
 */
function setupUIListeners(): void {
  // Generate button
  const generateBtn = document.getElementById("generateBtn");
  if (generateBtn) {
    generateBtn.addEventListener("click", handleGenerate);
  }

  // Error dismiss button
  const dismissError = document.getElementById("dismissError");
  if (dismissError) {
    dismissError.addEventListener("click", () => clearError());
  }

  // History toggle (mobile)
  const historyToggle = document.getElementById("historyToggle");
  if (historyToggle) {
    historyToggle.addEventListener("click", () => {
      historyPanelOpen.update((open) => !open);
    });
  }

  // History panel swipe (mobile)
  const historyPanel = document.getElementById("historyPanel");
  if (historyPanel) {
    let startY = 0;
    historyPanel.addEventListener("touchstart", (e) => {
      startY = e.touches[0]?.clientY ?? 0;
    });
    historyPanel.addEventListener("touchmove", (e) => {
      const currentY = e.touches[0]?.clientY ?? 0;
      const diff = currentY - startY;
      // Swipe down to close
      if (diff > 50) {
        historyPanelOpen.set(false);
      }
      // Swipe up to open
      if (diff < -50) {
        historyPanelOpen.set(true);
      }
    });
  }

  // System status modal
  const systemStatusBtn = document.getElementById("systemStatusBtn");
  const systemStatusModal = document.getElementById("systemStatusModal");
  const closeSystemStatus = document.getElementById("closeSystemStatus");

  if (systemStatusBtn && systemStatusModal) {
    systemStatusBtn.addEventListener("click", () => {
      systemStatusModal.classList.remove("hidden");
      loadSystemStatus();
    });
  }

  if (closeSystemStatus && systemStatusModal) {
    closeSystemStatus.addEventListener("click", () => {
      systemStatusModal.classList.add("hidden");
    });
    // Close on backdrop click
    systemStatusModal.addEventListener("click", (e) => {
      if (e.target === systemStatusModal) {
        systemStatusModal.classList.add("hidden");
      }
    });
  }

  // Download button
  const downloadBtn = document.getElementById("downloadBtn");
  if (downloadBtn) {
    downloadBtn.addEventListener("click", handleDownload);
  }
}

/**
 * Set up state subscriptions
 */
function setupStateSubscriptions(): void {
  // Re-render form when pipeline changes
  currentSchema.subscribe((schema) => {
    if (schema) {
      const formContainer = document.getElementById("formContainer");
      if (formContainer) {
        clearElement(formContainer);
        renderForm(formContainer, schema);
      }
      // Update page title
      document.title = `${schema.name} - LLM-DiT Studio`;
    }
  });

  // Update error banner
  errorMessage.subscribe((message) => {
    const banner = document.getElementById("errorBanner");
    const messageEl = document.getElementById("errorMessage");
    if (banner && messageEl) {
      if (message) {
        messageEl.textContent = message;
        setVisible(banner, true);
      } else {
        setVisible(banner, false);
      }
    }
  });

  // Update progress bar
  generationProgress.subscribe((progress) => {
    const fill = document.getElementById("progressFill");
    const percent = document.getElementById("progressPercent");
    if (fill) fill.style.width = `${progress}%`;
    if (percent) percent.textContent = `${Math.round(progress)}%`;
  });

  progressMessage.subscribe((message) => {
    const label = document.getElementById("progressLabel");
    if (label) label.textContent = message;
  });

  // Update loading state UI
  loadingState.subscribe((state) => {
    const generateBtn = document.getElementById("generateBtn");
    const progressContainer = document.getElementById("progressContainer");

    if (generateBtn) {
      if (state === "generating") {
        generateBtn.setAttribute("disabled", "");
        clearElement(generateBtn);
        generateBtn.appendChild(Icons.spinner());
        generateBtn.appendChild(document.createTextNode(" Generating..."));
      } else {
        generateBtn.removeAttribute("disabled");
        clearElement(generateBtn);
        generateBtn.appendChild(Icons.lightning());
        generateBtn.appendChild(document.createTextNode(" Generate"));
      }
    }

    if (progressContainer) {
      setVisible(progressContainer, state === "generating");
    }
  });

  // Update result display
  lastResultUrl.subscribe((url) => {
    const container = document.getElementById("resultContainer");
    const image = document.getElementById("resultImage") as HTMLImageElement;
    const video = document.getElementById("resultVideo") as HTMLVideoElement;

    if (!container || !image || !video) return;

    if (url) {
      setVisible(container, true);
      const type = lastResultType.get();

      if (type === "video") {
        setVisible(image, false);
        setVisible(video, true);
        video.src = url;
        video.load();
      } else {
        setVisible(video, false);
        setVisible(image, true);
        image.src = url;
      }
    } else {
      setVisible(container, false);
    }
  });

  // Update history panel state (mobile)
  historyPanelOpen.subscribe((open) => {
    const panel = document.getElementById("historyPanel");
    if (panel) {
      setClass(panel, "closed", !open);
      setClass(panel, "open", open);
    }
  });

  // Update loaded pipeline indicator
  loadedPipeline.subscribe((loaded) => {
    const indicator = document.getElementById("loadedPipelineIndicator");
    if (indicator) {
      if (loaded) {
        setVisible(indicator, true);
        indicator.textContent = `${loaded} loaded`;
      } else {
        setVisible(indicator, false);
      }
    }
  });
}

/**
 * Render pipeline tabs
 */
function renderPipelineTabs(
  pipelines: Record<
    string,
    { id: string; name: string; icon?: string; color: string }
  >
): void {
  const tabsContainer = document.getElementById("pipelineTabs");
  if (!tabsContainer) return;

  clearElement(tabsContainer);

  // Render tabs
  for (const [id, pipeline] of Object.entries(pipelines)) {
    const tab = createElement("button", {
      className: `pipeline-tab pipeline-${pipeline.color}`,
      "data-pipeline": id,
    });

    if (pipeline.icon) {
      const iconSpan = createElement("span", { className: "text-lg" });
      iconSpan.textContent = pipeline.icon;
      tab.appendChild(iconSpan);
    }

    const nameSpan = createElement("span");
    nameSpan.textContent = pipeline.name;
    tab.appendChild(nameSpan);

    tab.addEventListener("click", () => selectPipeline(id));

    tabsContainer.appendChild(tab);
  }

  // Subscribe to pipeline changes to update active state
  currentPipeline.subscribe((selected) => {
    const tabs = tabsContainer.querySelectorAll(".pipeline-tab");
    tabs.forEach((tab) => {
      const pipelineId = (tab as HTMLElement).dataset["pipeline"];
      setClass(tab, "active", pipelineId === selected);
      setClass(tab, "pipeline-border", pipelineId === selected);
    });
  });
}

/**
 * Handle generate button click
 */
async function handleGenerate(): Promise<void> {
  const schema = currentSchema.get();
  if (!schema) {
    setError("No pipeline selected");
    return;
  }

  const values = formValues.get();

  // Validate required fields
  for (const param of schema.params) {
    if (param.required && !values[param.id]) {
      setError(`${param.label} is required`);
      return;
    }
  }

  startGeneration();

  try {
    if (schema.supports_streaming) {
      // SSE streaming for video generation
      const result = await generateVideoStream(
        schema.endpoint,
        values,
        (event: SSEEvent) => {
          if (event.type === "progress") {
            updateProgress(event.percentage, event.message);
          }
        }
      );

      const url = getMediaUrl(result.video_url);
      completeGeneration(url, "video");

      // Add to history
      addToHistory({
        id: crypto.randomUUID(),
        type: "video",
        url,
        prompt: String(values["prompt"] ?? ""),
        seed: result.seed,
        timestamp: new Date().toISOString(),
        params: values,
        pipeline: schema.id,
      });
    } else {
      // Standard request/response
      const result = await generate(schema.endpoint, values);

      // Determine result type
      let url: string;
      let type: "image" | "video" | "layers" = "image";

      if ("image_url" in result) {
        url = getMediaUrl(result.image_url);
      } else if ("layer_urls" in result) {
        url = getMediaUrl(result.composite_path);
        type = "layers";
      } else {
        throw new Error("Unknown result format");
      }

      completeGeneration(url, type);

      // Add to history
      addToHistory({
        id: crypto.randomUUID(),
        type,
        url,
        prompt: String(values["prompt"] ?? ""),
        seed: result.seed,
        timestamp: new Date().toISOString(),
        params: values,
        pipeline: schema.id,
      });
    }
  } catch (error) {
    if (error instanceof APIClientError) {
      setError(error.message);
    } else {
      setError(error instanceof Error ? error.message : "Generation failed");
    }
    loadingState.set("idle");
  }
}

/**
 * Handle download button click
 */
function handleDownload(): void {
  const url = lastResultUrl.get();
  if (!url) return;

  const a = document.createElement("a");
  a.href = url;
  a.download = url.split("/").pop() ?? "result";
  a.click();
}

/**
 * Load system status into modal
 */
async function loadSystemStatus(): Promise<void> {
  const content = document.getElementById("systemStatusContent");
  if (!content) return;

  // Show loading spinner
  clearElement(content);
  const loadingDiv = createElement("div", {
    className: "flex justify-center py-4",
  });
  loadingDiv.appendChild(Icons.spinner());
  content.appendChild(loadingDiv);

  try {
    const response = await fetch("/api/vram/status");
    const status = (await response.json()) as {
      gpu_name?: string;
      used_vram_gb?: number;
      total_vram_gb?: number;
      loaded_models?: Record<string, boolean>;
    };

    clearElement(content);

    const container = createElement("div", { className: "space-y-3" });

    // GPU info
    const gpuRow = createElement(
      "div",
      { className: "flex justify-between" },
      createElement("span", { className: "text-gray-400" }, "GPU"),
      createElement(
        "span",
        { className: "font-mono" },
        status.gpu_name ?? "Unknown"
      )
    );
    container.appendChild(gpuRow);

    // VRAM usage
    const usedVram = status.used_vram_gb?.toFixed(1) ?? "?";
    const totalVram = status.total_vram_gb?.toFixed(1) ?? "?";
    const vramRow = createElement(
      "div",
      { className: "flex justify-between" },
      createElement("span", { className: "text-gray-400" }, "VRAM Used"),
      createElement(
        "span",
        { className: "font-mono" },
        `${usedVram}GB / ${totalVram}GB`
      )
    );
    container.appendChild(vramRow);

    // Progress bar
    const vramPercent =
      status.used_vram_gb && status.total_vram_gb
        ? (status.used_vram_gb / status.total_vram_gb) * 100
        : 0;
    const progressBar = createElement("div", {
      className: "h-2 bg-gray-700 rounded-full overflow-hidden",
    });
    const progressFill = createElement("div", {
      className: "h-full bg-blue-500 rounded-full",
    });
    progressFill.style.width = `${vramPercent.toFixed(0)}%`;
    progressBar.appendChild(progressFill);
    container.appendChild(progressBar);

    // Loaded models
    if (status.loaded_models) {
      const modelsSection = createElement("div", {
        className: "pt-2 border-t border-gray-700",
      });
      modelsSection.appendChild(
        createElement(
          "span",
          { className: "text-sm text-gray-400" },
          "Loaded Models:"
        )
      );

      const modelsList = createElement("ul", { className: "mt-2 space-y-1 text-sm" });
      for (const [model, loaded] of Object.entries(status.loaded_models)) {
        const li = createElement("li", {
          className: "flex items-center gap-2",
        });
        const dot = createElement("span", {
          className: `w-2 h-2 rounded-full ${loaded ? "bg-green-500" : "bg-gray-600"}`,
        });
        li.appendChild(dot);
        li.appendChild(createElement("span", {}, model));
        modelsList.appendChild(li);
      }
      modelsSection.appendChild(modelsList);
      container.appendChild(modelsSection);
    }

    content.appendChild(container);
  } catch {
    clearElement(content);
    content.appendChild(
      createElement(
        "div",
        { className: "text-center text-red-400 py-4" },
        "Failed to load status"
      )
    );
  }
}

// Initialize when DOM is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
