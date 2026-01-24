# web app architecture

last updated: 2026-01-24

## overview

The web application provides a single-page UI for multi-model image and video generation. It supports five model types:

- **Z-Image** - Custom text-to-image model with advanced features (DyPE, SLG, FMTT, VL conditioning)
- **Qwen-Image** - Layer-based image decomposition and editing
- **Qwen-Image T2I** - Pure text-to-image generation (2512 variant)
- **LTX-2** - Video generation with Server-Sent Events progress tracking
- **FLUX.2 Klein** - Fast distilled image generation with multiple variants

The application uses a Python FastAPI backend with a vanilla JavaScript frontend, communicating via REST and SSE.

## file structure

```
web/
├── server.py                    # FastAPI backend (4900+ lines)
├── index.html                   # Single-page HTML application
└── static/
    ├── css/                     # Modular CSS
    └── js/
        ├── api-client.js        # API abstraction layer
        ├── app.js               # Main initialization & form handling
        ├── qwen-image.js        # Qwen model switching and features
        ├── ltx2.js              # Video generation with SSE
        ├── flux2.js             # FLUX.2 image generation
        ├── history.js           # Generation history management
        ├── config-manager.js    # Server config management
        ├── vl-conditioning.js   # Vision-language conditioning
        ├── img2img.js           # Image-to-image features
        ├── layer-blend.js       # Layer weight configuration
        ├── advanced.js          # Advanced features (DyPE, SLG, FMTT)
        ├── rewriter.js          # Prompt rewriting
        ├── resolution.js        # Resolution presets
        ├── state.js             # Global app state
        ├── ui-utils.js          # UI helper functions
        ├── templates.js         # Template management
        └── image-utils.js       # Image workflow utilities
```

## backend architecture

### fastapi application structure

The backend is a single FastAPI application defined in `web/server.py`:

```python
import argparse
import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(title="Z-Image Generator")

# CORS for local development
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)

# Static files mounted at /static
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
```

### global pipeline variables

The backend maintains global references to model pipelines, loaded on-demand:

```python
# Global pipeline/encoder (loaded on startup or on-demand)
pipeline = None                  # Z-Image pipeline (DiT + VAE + encoder)
encoder = None                   # Encoder-only mode (fast)
rewriter_backend = None          # API backend for rewriting
vl_extractor = None              # Qwen3-VL embedding extractor
vl_rewriter = None               # Qwen3-VL vision rewriting
vl_embeddings_cache = {}         # VL embeddings cache (hash-keyed)
runtime_config = None            # RuntimeConfig from CLI/TOML

# Model-specific pipelines
qwen_image_pipeline = None       # Qwen-Image layer decompose/edit
qwen_image_t2i_pipeline = None   # Qwen-Image T2I (pure text-to-image)
ltx2_pipeline = None             # LTX-2 video generation (Gemma 3-12B encoder)
flux2_pipeline = None            # FLUX.2 Klein (Qwen3 encoder, 3-stage offload)

# In-memory state
generation_history = []          # Recent generations (max 50)
session_file_values = {}         # Original config file values
session_modified_fields = set()  # Modified during session
pending_restart_changes = {}     # Changes requiring restart
```

### request models (pydantic)

All endpoints use Pydantic `BaseModel` classes for request validation:

```python
class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    thinking_content: Optional[str] = None
    assistant_content: Optional[str] = None
    force_think_block: bool = False
    template: Optional[str] = None
    width: int = 1024
    height: int = 1024
    steps: int = 9
    guidance_scale: float = 0.0
    shift: Optional[float] = None
    dynamic_shift: bool = False
    d_noise: float = 1.0
    seed: Optional[int] = None
    # ... advanced features
    layer_weights: Optional[List[float]] = None
    slg_scale: Optional[float] = None
    fmtt_scale: Optional[float] = None
    # ... ~40 total fields

class Img2ImgRequest(BaseModel):
    # Same as GenerateRequest plus:
    image: str  # Base64-encoded input image
    strength: float = 0.8
    differential_mask: Optional[str] = None

class QwenImageDecomposeRequest(BaseModel):
    image: str  # Base64-encoded
    layer_num: int = 3
    cfg_scale: float = 4.0
    steps: int = 30
    resolution: int = 1024

class LTX2GenerateRequest(BaseModel):
    prompt: str
    num_frames: int = 97
    width: int = 768
    height: int = 512
    num_inference_steps: int = 40
    guidance_scale: float = 3.0
    seed: Optional[int] = None

class Flux2GenerateRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    steps: int = 4
    guidance_scale: float = 1.0
    seed: Optional[int] = None
    model_variant: str = "klein-9b"
    edit_mode: bool = False
    reference_images: Optional[List[str]] = None
```

### endpoint patterns

The API follows RESTful conventions with consistent patterns:

#### status endpoints (GET)

Check if a model/feature is available and loaded:

```python
@app.get("/api/{model}/status")
async def model_status():
    """Return availability, loaded state, and configuration."""
    return {
        "available": bool(model_pipeline),
        "configured": bool(runtime_config.model_path),
        "loaded": model_pipeline is not None,
        "model_path": runtime_config.model_path,
        # ... model-specific defaults
    }
```

Examples:
- `/api/qwen-image/status`
- `/api/qwen-image-2512/status`
- `/api/ltx2/status`
- `/api/flux2/status`
- `/api/vl/status`

#### generate endpoints (POST)

Generate content using a model:

```python
@app.post("/api/generate")
async def generate(request: GenerateRequest):
    """Generate image from prompt."""
    global pipeline

    # Load pipeline on-demand if not loaded
    if pipeline is None:
        pipeline = load_pipeline(runtime_config)

    # Run generation in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    image = await loop.run_in_executor(None, _generate_sync, request)

    # Return binary PNG with metadata in headers
    return StreamingResponse(
        io.BytesIO(image_bytes),
        media_type="image/png",
        headers={
            "X-Seed": str(seed),
            "X-Generation-Time": f"{gen_time:.2f}",
            "X-History-Id": str(history_id),
        }
    )
```

Generate endpoints:
- `/api/generate` - Z-Image text-to-image
- `/api/img2img` - Z-Image image-to-image
- `/api/vl/generate` - Z-Image with VL conditioning
- `/api/qwen-image/decompose` - Decompose image to layers
- `/api/qwen-image/edit-layer` - Edit single layer
- `/api/qwen-image/edit-multi` - Combine multiple images
- `/api/qwen-image-2512/generate` - Qwen T2I
- `/api/ltx2/generate/stream` - Video with SSE (see below)
- `/api/flux2/generate` - FLUX.2 image generation

#### vram management (POST)

Unload models to free GPU memory:

```python
@app.post("/api/vram/unload-{model}")
async def vram_unload_model():
    """Unload model pipeline from VRAM."""
    global model_pipeline
    if model_pipeline is None:
        return {"message": "Not loaded", "unloaded": False}

    # Move to CPU before deletion
    model_pipeline.transformer.to("cpu")
    model_pipeline.vae.to("cpu")
    del model_pipeline
    model_pipeline = None

    # Force garbage collection
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return {"message": "Unloaded", "unloaded": True}
```

VRAM endpoints:
- `/api/vram/unload-zimage`
- `/api/vram/unload-qwen-image`
- `/api/vram/unload-qwen-image-t2i`
- `/api/vram/unload-ltx2`
- `/api/vram/unload-flux2`

### async generation with run_in_executor

CPU-bound generation runs in a thread pool to avoid blocking the event loop:

```python
async def generate(request: GenerateRequest):
    loop = asyncio.get_event_loop()

    # Synchronous wrapper that calls the pipeline
    def _generate_sync():
        return pipeline.generate(
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            # ...
        )

    # Run in thread pool
    image = await loop.run_in_executor(None, _generate_sync)

    # Return result
    return image
```

### sse streaming for ltx-2 video progress

LTX-2 video generation uses Server-Sent Events to stream progress:

```python
@app.post("/api/ltx2/generate/stream")
async def ltx2_generate_stream(request: LTX2GenerateRequest):
    """Generate video with SSE progress."""

    async def event_stream() -> AsyncIterator[str]:
        """Yield SSE messages."""
        def progress_callback(step: int, total: int, elapsed: float):
            # Called from pipeline during generation
            msg = {
                "type": "progress",
                "step": step,
                "total": total,
                "elapsed": elapsed,
                "eta": (elapsed / step) * (total - step),
            }
            # Queue for async yield
            progress_queue.put_nowait(msg)

        # Run generation in thread pool with callback
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(None, _generate_video, request, progress_callback)

        # Yield progress events
        while not future.done():
            try:
                msg = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                yield f"data: {json.dumps(msg)}\n\n"
            except asyncio.TimeoutError:
                continue

        # Yield final result
        video_path = await future
        yield f"data: {json.dumps({'type': 'complete', 'video_url': video_path})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

## frontend architecture

### single-page html structure

The frontend is a single HTML file (`web/index.html`) with modular JavaScript. Key structural elements:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="/static/css/layout.css">
    <link rel="stylesheet" href="/static/css/components.css">
    <link rel="stylesheet" href="/static/css/forms.css">
</head>
<body class="bg-gray-900 text-gray-100">
    <div class="flex flex-col md:flex-row min-h-screen">
        <!-- Main Content -->
        <div class="flex-1 p-4 md:p-6 overflow-y-auto">
            <!-- Model Type Selector (tabs) -->
            <div class="mb-6 p-4 bg-gray-800/50 rounded-xl">
                <button id="modelTypeZImage">Z-Image</button>
                <button id="modelTypeQwenImage">Qwen-Image</button>
                <button id="modelTypeQwenImage2512">Qwen T2I</button>
                <button id="modelTypeLTX2">LTX-2</button>
                <button id="modelTypeFLUX2">FLUX.2</button>
            </div>

            <!-- Z-Image Controls (shown for Z-Image and Qwen T2I) -->
            <div id="zImageControls">
                <form id="generationForm">
                    <textarea id="prompt"></textarea>
                    <!-- Resolution, steps, guidance, advanced features... -->
                </form>
            </div>

            <!-- Qwen-Image Controls -->
            <div id="qwenImageSection" class="hidden">
                <!-- Layer decomposition, editing, combining... -->
            </div>

            <!-- LTX-2 Controls -->
            <div id="ltx2Section" class="hidden">
                <form id="ltx2Form">
                    <!-- Video generation controls -->
                </form>
            </div>

            <!-- FLUX.2 Controls -->
            <div id="flux2Section" class="hidden">
                <form id="flux2Form">
                    <!-- FLUX.2 generation controls -->
                </form>
            </div>

            <!-- Result Display -->
            <div id="result" class="hidden">
                <img id="resultImage">
                <button id="downloadBtn">Download</button>
            </div>
        </div>

        <!-- History Sidebar (desktop) / Bottom Sheet (mobile) -->
        <div class="history-panel">
            <div id="historyGrid">
                <!-- Thumbnail gallery -->
            </div>
        </div>
    </div>

    <!-- Modals (settings, image lightbox, edit dialogs) -->
    <div id="settingsModal" class="hidden">...</div>
    <div id="imageModal" class="hidden">...</div>

    <!-- JavaScript Modules (order matters) -->
    <script src="/static/js/state.js"></script>
    <script src="/static/js/api-client.js"></script>
    <script src="/static/js/ui-utils.js"></script>
    <script src="/static/js/templates.js"></script>
    <script src="/static/js/history.js"></script>
    <script src="/static/js/resolution.js"></script>
    <script src="/static/js/rewriter.js"></script>
    <script src="/static/js/vl-conditioning.js"></script>
    <script src="/static/js/img2img.js"></script>
    <script src="/static/js/layer-blend.js"></script>
    <script src="/static/js/advanced.js"></script>
    <script src="/static/js/qwen-image.js"></script>
    <script src="/static/js/ltx2.js"></script>
    <script src="/static/js/flux2.js"></script>
    <script src="/static/js/image-utils.js"></script>
    <script src="/static/js/config-manager.js"></script>
    <script src="/static/js/app.js"></script> <!-- Main entry point, init on DOMContentLoaded -->
</body>
</html>
```

### model type switching

The UI switches between five model types using tabs:

```javascript
// qwen-image.js
function getSelectedModelType() {
    const zimageBtn = document.getElementById('modelTypeZImage');
    const qwenImageBtn = document.getElementById('modelTypeQwenImage');
    const qwenImage2512Btn = document.getElementById('modelTypeQwenImage2512');
    const ltx2Btn = document.getElementById('modelTypeLTX2');
    const flux2Btn = document.getElementById('modelTypeFLUX2');

    if (qwenImageBtn?.classList.contains('bg-blue-600')) return 'qwenimage';
    if (qwenImage2512Btn?.classList.contains('bg-blue-600')) return 'qwenimage2512';
    if (ltx2Btn?.classList.contains('bg-blue-600')) return 'ltx2';
    if (flux2Btn?.classList.contains('bg-blue-600')) return 'flux2';
    return 'zimage';
}

function updateQiSections() {
    const modelType = getSelectedModelType();

    // Show/hide sections based on model type
    document.getElementById('zImageControls')?.classList.toggle('hidden',
        modelType !== 'zimage' && modelType !== 'qwenimage2512');
    document.getElementById('qwenImageSection')?.classList.toggle('hidden',
        modelType !== 'qwenimage');
    document.getElementById('ltx2Section')?.classList.toggle('hidden',
        modelType !== 'ltx2');
    document.getElementById('flux2Section')?.classList.toggle('hidden',
        modelType !== 'flux2');
}

// Model tab click handlers
document.getElementById('modelTypeZImage').addEventListener('click', () => {
    switchModelType('zimage');
    setModelDefaults('zimage');
});
```

### javascript module responsibilities

#### api-client.js - API abstraction layer

Centralizes all backend communication:

```javascript
const ApiClient = {
    // System & Status
    async getSystemStatus() {
        const response = await fetch(`${API_BASE}/api/system/status`);
        return response.json();
    },

    // VRAM Management
    async unloadZImage() {
        const response = await fetch(`${API_BASE}/api/vram/unload-zimage`, { method: 'POST' });
        return response.json();
    },

    // Z-Image Generation
    async generate(data) {
        const response = await fetch(`${API_BASE}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        return this._handleImageResponse(response);
    },

    // Binary PNG response handler
    async _handleImageResponse(response) {
        const blob = await response.blob();
        const base64 = await this._blobToBase64(blob);
        return {
            image: base64,
            seed: response.headers.get('X-Seed'),
            gen_time: parseFloat(response.headers.get('X-Generation-Time')),
            history_id: response.headers.get('X-History-Id'),
        };
    },

    // LTX-2 SSE streaming
    async _ltx2GenerateSSE(params, { onProgress, onStatus, onComplete, onError }) {
        const response = await fetch(`${API_BASE}/api/ltx2/generate/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));
                    if (data.type === 'progress') onProgress(data);
                    else if (data.type === 'complete') onComplete(data);
                }
            }
        }
    },
};
```

#### app.js - Main initialization

Entry point that wires everything together:

```javascript
async function initializeApp() {
    // Initialize DOM references (state.js)
    initDOMReferences();

    // Setup form submission
    DOM.form?.addEventListener('submit', handleFormSubmit);

    // Initialize feature modules
    initHistoryEvents();
    initTemplateEvents();
    initRewriterEvents();
    initVLEvents();
    initImg2ImgEvents();
    initLayerBlendEvents();
    initAdvancedEvents();
    initQwenImageEvents();

    // Setup modals and keyboard shortcuts
    setupSettingsModal();
    setupKeyboardShortcuts();

    // Load initial data in parallel
    await Promise.all([
        loadTemplates(),
        loadGenerationConfig(),
        loadHistory(),
        checkVLStatus(),
        checkQwenImageStatus(),
        checkQwenImage2512Status(),
        loadRewriterConfig(),
    ]);

    // Initialize sub-systems
    await ResolutionSelector.init();
    await ConfigManager.init();

    // Set defaults and update UI
    setModelDefaults('zimage');
    updateSliderDisplays();
}

document.addEventListener('DOMContentLoaded', initializeApp);
```

Form submission handler:

```javascript
async function handleFormSubmit(e) {
    e.preventDefault();

    const modelType = getSelectedModelType();
    const params = {
        prompt: document.getElementById('prompt').value,
        width: resolution.width,
        height: resolution.height,
        steps: parseInt(DOM.stepsSlider.value),
        guidance_scale: parseFloat(DOM.guidanceScaleSlider.value),
        seed: document.getElementById('seed')?.value || null,
        // ... collect all form values
    };

    // Add model-specific params
    if (modelType === 'zimage') {
        params.shift = parseFloat(DOM.shiftSlider.value);
        params.d_noise = parseFloat(DOM.dNoiseSlider.value);
        // Add advanced features (DyPE, SLG, FMTT, etc.)
    }

    // Determine endpoint
    let endpoint = 'generate';
    if (getVLParams()) endpoint = 'vl';
    else if (getImg2ImgParams()) endpoint = 'img2img';
    else if (modelType === 'qwenimage2512') endpoint = 'qwen2512';

    // Call API
    showStatus('Generating...', 30);
    const data = await ApiClient.generate(params);

    // Display result
    document.getElementById('resultImage').src = data.image;
    document.getElementById('result').classList.remove('hidden');

    // Reload history
    await loadHistory();
}
```

#### qwen-image.js - Model switching and Qwen features

Handles Qwen-specific UI and operations:

```javascript
async function checkQwenImageStatus() {
    const data = await ApiClient.getQwenImageStatus();
    AppState.features.qwenImageEnabled = data.available;
    // Show/hide Qwen-Image section
}

async function decomposeImage() {
    const imageInput = document.getElementById('qiDecomposeImage');
    const base64 = await fileToBase64(imageInput.files[0]);

    const params = {
        image: base64,
        layer_num: parseInt(document.getElementById('qiLayerNum').value),
        cfg_scale: parseFloat(document.getElementById('qiCfgScale').value),
        steps: parseInt(document.getElementById('qiSteps').value),
        resolution: parseInt(document.getElementById('qiResolution').value),
    };

    // Returns ZIP file with layers
    const blob = await ApiClient.qwenImageDecompose(params);
    downloadBlob(blob, `layers_${Date.now()}.zip`);
}
```

#### ltx2.js - Video generation with SSE

Handles video generation with real-time progress:

```javascript
async function generateVideo() {
    const params = {
        prompt: document.getElementById('ltx2Prompt').value,
        num_frames: parseInt(document.getElementById('ltx2Frames').value),
        width: parseInt(document.getElementById('ltx2Width').value),
        height: parseInt(document.getElementById('ltx2Height').value),
        num_inference_steps: parseInt(document.getElementById('ltx2Steps').value),
        guidance_scale: parseFloat(document.getElementById('ltx2Guidance').value),
        seed: document.getElementById('ltx2Seed').value || null,
    };

    const progressBar = document.getElementById('ltx2Progress');
    const statusText = document.getElementById('ltx2Status');

    await ApiClient._ltx2GenerateSSE(params, {
        onProgress: (data) => {
            const pct = (data.step / data.total) * 100;
            progressBar.style.width = `${pct}%`;
            statusText.textContent = `Step ${data.step}/${data.total} (${data.its.toFixed(2)} it/s)`;
        },
        onComplete: (data) => {
            // Display video
            const video = document.getElementById('ltx2Video');
            video.src = data.video_url;
            video.classList.remove('hidden');
            currentVideoUrl = data.video_url;
        },
        onError: (error) => {
            showError(error);
        },
    });
}
```

#### flux2.js - FLUX.2 image generation

Handles FLUX.2 variants and generation:

```javascript
const FLUX2_MODEL_DEFAULTS = {
    'klein-9b': { steps: 4, guidance: 1.0, distilled: true },
    'klein-base-9b': { steps: 50, guidance: 4.0, distilled: false },
    // ... other variants
};

function updateFlux2Defaults(model) {
    const defaults = FLUX2_MODEL_DEFAULTS[model];
    document.getElementById('flux2Steps').value = defaults.steps;
    document.getElementById('flux2Guidance').value = defaults.guidance;
}

async function generateFlux2Image() {
    const params = {
        prompt: document.getElementById('flux2Prompt').value,
        width: parseInt(document.getElementById('flux2Width').value),
        height: parseInt(document.getElementById('flux2Height').value),
        steps: parseInt(document.getElementById('flux2Steps').value),
        guidance_scale: parseFloat(document.getElementById('flux2Guidance').value),
        seed: document.getElementById('flux2Seed').value || null,
        model_variant: document.getElementById('flux2Model').value,
    };

    const data = await ApiClient.flux2Generate(params);
    document.getElementById('flux2Result').src = data.image;
}
```

#### history.js - Generation history

Manages thumbnail gallery and history actions:

```javascript
async function loadHistory() {
    const data = await ApiClient.getHistory();
    const grid = document.getElementById('historyGrid');
    grid.innerHTML = '';

    data.history.forEach((item, index) => {
        const div = document.createElement('div');
        div.className = 'history-item relative';

        const img = document.createElement('img');
        img.src = item.image;
        img.className = 'w-full h-auto rounded cursor-pointer';

        const overlay = document.createElement('div');
        overlay.className = 'history-overlay absolute inset-0 bg-black/80 opacity-0';

        const reuseBtn = document.createElement('button');
        reuseBtn.textContent = 'Reuse';
        reuseBtn.onclick = () => reloadParams(index);

        const deleteBtn = document.createElement('button');
        deleteBtn.textContent = 'Delete';
        deleteBtn.onclick = () => deleteHistoryItem(index);

        overlay.appendChild(reuseBtn);
        overlay.appendChild(deleteBtn);
        div.appendChild(img);
        div.appendChild(overlay);
        grid.appendChild(div);
    });
}

async function deleteHistoryItem(index) {
    await ApiClient.deleteHistoryItem(index);
    await loadHistory();
}
```

#### config-manager.js - Server configuration

UI for runtime config management:

```javascript
const ConfigManager = {
    async init() {
        await this.loadSessionConfig();
        this.setupEventHandlers();
    },

    async loadSessionConfig() {
        const data = await fetch('/api/config/session').then(r => r.json());
        // Populate config UI
    },

    async loadConfig(filename, profile) {
        const result = await ApiClient.loadConfig(filename, profile);
        if (result.requires_restart) {
            showRestartWarning();
        }
    },
};
```

### ui patterns

#### collapsible sections

Advanced features use collapsible sections:

```html
<div class="collapsible-section">
    <button class="collapsible-header" onclick="toggleSection(this)">
        <span>Advanced Features</span>
        <svg class="chevron">...</svg>
    </button>
    <div class="collapsible-content hidden">
        <!-- Feature controls -->
    </div>
</div>
```

```javascript
function toggleSection(button) {
    const content = button.nextElementSibling;
    const chevron = button.querySelector('.chevron');
    content.classList.toggle('hidden');
    chevron.classList.toggle('rotate-180');
}
```

#### progress bars

Generation progress uses width-based bar:

```html
<div class="w-full bg-gray-700 rounded-full h-2">
    <div id="progressFill" class="bg-blue-600 h-2 rounded-full transition-all"></div>
</div>
```

```javascript
function showStatus(message, percent) {
    document.getElementById('progressFill').style.width = `${percent}%`;
    document.getElementById('statusText').textContent = message;
}
```

#### result display with actions

Generated images show with action buttons:

```html
<div id="result" class="hidden">
    <img id="resultImage" class="w-full rounded-lg cursor-pointer">
    <div class="flex gap-2 mt-4">
        <button id="downloadBtn">Download</button>
        <button id="resultUseImg2Img">Use as Img2Img</button>
        <button id="resultUseVL">Use as VL Reference</button>
        <button id="resultUseQwenEdit">Edit in Qwen</button>
    </div>
</div>
```

## api patterns

### status endpoints (GET)

Check model availability and configuration:

```
GET /api/qwen-image/status
Response:
{
    "available": true,
    "configured": true,
    "loaded": true,
    "model_path": "/path/to/model",
    "default_layer_num": 3,
    "default_cfg_scale": 4.0
}
```

### generate endpoints (POST)

Generate content with binary response:

```
POST /api/generate
Content-Type: application/json

{
    "prompt": "a cat",
    "width": 1024,
    "height": 1024,
    "steps": 9,
    "guidance_scale": 0.0,
    "seed": null
}

Response:
Content-Type: image/png
X-Seed: 42
X-Generation-Time: 2.34
X-History-Id: 0

<binary PNG data>
```

### vram management (POST)

Unload models to free memory:

```
POST /api/vram/unload-zimage

Response:
{
    "message": "Z-Image pipeline unloaded (encoder + DiT + VAE freed ~14 GB VRAM)",
    "unloaded": true
}
```

### configuration endpoints (GET)

Get runtime configuration:

```
GET /api/generation-config

Response:
{
    "width": 1024,
    "height": 1024,
    "steps": 9,
    "guidance_scale": 0.0,
    "shift": 3.0,
    "d_noise": 1.0,
    "dynamic_shift": false,
    // ... all configurable params
}
```

## data flow

### user interaction flow

1. **User fills form** (prompt, resolution, steps, etc.)
2. **User clicks Generate** → `handleFormSubmit(e)`
3. **JavaScript collects params** from form values
4. **API call** via `ApiClient.generate(params)`
5. **Backend receives request** → validates with Pydantic
6. **Backend loads pipeline** (if not loaded)
7. **Backend runs generation** in thread pool
8. **Backend returns binary PNG** with metadata headers
9. **Frontend converts to base64** and displays
10. **UI updates** (show result, reload history)

### image handling

Images flow through base64 encoding throughout:

```javascript
// File upload → base64
async function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// API response (binary) → base64
async function _handleImageResponse(response) {
    const blob = await response.blob();
    const reader = new FileReader();
    reader.readAsDataURL(blob);
    return new Promise(resolve => {
        reader.onloadend = () => resolve({
            image: reader.result,  // "data:image/png;base64,..."
            seed: response.headers.get('X-Seed'),
        });
    });
}

// Display in DOM
document.getElementById('resultImage').src = base64;

// Download
function downloadImage(base64) {
    const link = document.createElement('a');
    link.href = base64;
    link.download = `image_${Date.now()}.png`;
    link.click();
}
```

## memory management

### model loading on-demand

Models load lazily when first accessed:

```python
@app.post("/api/generate")
async def generate(request: GenerateRequest):
    global pipeline

    if pipeline is None:
        logger.info("[Z-Image] Loading pipeline on-demand...")
        from llm_dit.pipelines.zimage import ZImagePipeline
        pipeline = ZImagePipeline.from_pretrained(
            runtime_config.model_path,
            device=runtime_config.dit_device,
            # ...
        )

    # Use pipeline...
```

### vram unloading between models

Switching models frees previous model's VRAM:

```python
def unload_zimage_pipeline() -> bool:
    global pipeline, encoder
    if pipeline is not None:
        # Move to CPU before deletion
        pipeline.transformer.to("cpu")
        pipeline.vae.to("cpu")
        del pipeline
        pipeline = None

        # Force cleanup
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        return True
    return False
```

Frontend prompts user before model switches:

```javascript
async function switchToModel(newModel) {
    const currentModel = getLoadedModel();
    if (currentModel && currentModel !== newModel) {
        const confirmUnload = confirm(
            `Switching models will unload ${currentModel}. Continue?`
        );
        if (confirmUnload) {
            await ApiClient.unloadModel(currentModel);
        }
    }
}
```

### block offload patterns

Large models use sequential CPU offload for 24GB VRAM:

```python
# LTX-2: Gemma 3-12B text encoder with sequential offload
ltx2_pipeline = LTX2Pipeline.from_pretrained(
    model_path,
    cpu_offload=True,  # Sequential CPU offload
)

# FLUX.2: Qwen3 encoder with three-stage offloading
flux2_pipeline = Flux2Pipeline.from_pretrained(
    model_path,
    cpu_offload="three_stage",  # Text encoder → DiT → VAE
)
```

## key architectural decisions

1. **Single-page vanilla JavaScript** - No build step, no framework dependencies, fast iteration
2. **Modular JS files** - Feature-based modules loaded in order, shared via `window` globals
3. **Binary PNG responses** - Efficient transfer, metadata in headers
4. **On-demand model loading** - Save VRAM by loading only when needed
5. **SSE for video progress** - Real-time feedback for long operations
6. **Base64 everywhere in frontend** - Simplifies image handling, no temp files
7. **Pydantic for validation** - Type-safe API with auto-generated docs
8. **Thread pool for generation** - Keep FastAPI async loop responsive
9. **Global pipeline variables** - Simple state management, clear ownership
10. **VRAM management endpoints** - Explicit control over GPU memory

## extending the application

### adding a new model type

1. **Add global pipeline variable** in `server.py`:
   ```python
   new_model_pipeline = None
   ```

2. **Create Pydantic request model**:
   ```python
   class NewModelRequest(BaseModel):
       prompt: str
       # ... model-specific params
   ```

3. **Add status endpoint**:
   ```python
   @app.get("/api/new-model/status")
   async def new_model_status():
       return {"available": new_model_pipeline is not None}
   ```

4. **Add generate endpoint**:
   ```python
   @app.post("/api/new-model/generate")
   async def new_model_generate(request: NewModelRequest):
       global new_model_pipeline
       # Load if needed, run generation, return result
   ```

5. **Add API client methods** in `api-client.js`:
   ```javascript
   async newModelGenerate(data) {
       const response = await fetch(`${API_BASE}/api/new-model/generate`, {
           method: 'POST',
           headers: { 'Content-Type': 'application/json' },
           body: JSON.stringify(data),
       });
       return this._handleImageResponse(response);
   }
   ```

6. **Add UI section** in `index.html`:
   ```html
   <div id="newModelSection" class="hidden">
       <form id="newModelForm">
           <!-- Model-specific controls -->
       </form>
   </div>
   ```

7. **Create JS module** `new-model.js`:
   ```javascript
   function initNewModel() {
       document.getElementById('newModelForm').addEventListener('submit', async (e) => {
           e.preventDefault();
           await generateWithNewModel();
       });
   }
   ```

8. **Add model tab** in HTML and wire switching logic.

### adding a new feature to existing model

1. **Add field to request model** in `server.py`:
   ```python
   class GenerateRequest(BaseModel):
       # ... existing fields
       new_feature_param: Optional[float] = None
   ```

2. **Update generation logic** to use new param.

3. **Add UI control** in HTML:
   ```html
   <input type="range" id="newFeatureSlider" min="0" max="1" step="0.1">
   ```

4. **Collect in form handler** in `app.js`:
   ```javascript
   params.new_feature_param = parseFloat(document.getElementById('newFeatureSlider').value);
   ```

5. **Update config files** if param should be configurable via TOML.

## troubleshooting

### common issues

**Model not loading:**
- Check `/api/{model}/status` endpoint
- Verify `model_path` in config
- Check server logs for load errors

**VRAM out of memory:**
- Use `/api/vram/unload-{model}` to free memory
- Check `/api/system/status` for current usage
- Enable CPU offload in config

**SSE progress not updating (LTX-2):**
- Check browser network tab for EventSource connection
- Verify backend is yielding progress events
- Look for JavaScript errors in console

**Images not displaying:**
- Check for base64 encoding errors
- Verify PNG binary response from backend
- Check CORS headers if using different port
