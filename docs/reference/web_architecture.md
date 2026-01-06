# Web UI Architecture

last updated: 2026-01-06

The web UI is a modular JavaScript/CSS application served by FastAPI.

## Directory Structure

```
web/
  index.html          # Main HTML (markup only, ~2100 lines)
  server.py           # FastAPI backend
  static/
    css/
      layout.css      # Scrollbar, responsive, animations
      components.css  # Buttons, cards, modals, dropzone, resolution selector
      forms.css       # Slider/range styling
    js/
      state.js        # DOM references, shared state (AppState, DOM)
      api-client.js   # ApiClient object - all backend communication
      ui-utils.js     # Helpers: escapeHtml, formatNumber, debounce, getResolution
      history.js      # Generation history panel
      templates.js    # Template loading and selection
      rewriter.js     # Prompt rewriter system
      vl-conditioning.js  # VL embedding extraction
      img2img.js      # Img2img + mask canvas
      layer-blend.js  # Multi-layer blending controls
      advanced.js     # DyPE, SLG, FMTT, CFG controls
      qwen-image.js   # Qwen-Image model controls
      resolution.js   # Resolution selector with aspect filters
      image-utils.js  # Shared image loading/workflow utilities
      config-manager.js  # Settings modal config management
      app.js          # Initialization and main form handling
```

## Script Loading Order

Order matters due to dependencies:

```html
<!-- CSS -->
<link rel="stylesheet" href="/static/css/layout.css">
<link rel="stylesheet" href="/static/css/components.css">
<link rel="stylesheet" href="/static/css/forms.css">

<!-- JS (dependencies flow downward) -->
<script src="/static/js/state.js"></script>
<script src="/static/js/api-client.js"></script>
<script src="/static/js/ui-utils.js"></script>
<script src="/static/js/history.js"></script>
<script src="/static/js/templates.js"></script>
<script src="/static/js/rewriter.js"></script>
<script src="/static/js/vl-conditioning.js"></script>
<script src="/static/js/img2img.js"></script>
<script src="/static/js/layer-blend.js"></script>
<script src="/static/js/advanced.js"></script>
<script src="/static/js/qwen-image.js"></script>
<script src="/static/js/resolution.js"></script>
<script src="/static/js/image-utils.js"></script>
<script src="/static/js/config-manager.js"></script>
<script src="/static/js/app.js"></script>
```

## Architecture Patterns

### Global State

`state.js` provides two singletons:

```javascript
// DOM element references (initialized once)
const DOM = {
    form: null,
    generateBtn: null,
    stepsSlider: null,
    // ... 50+ elements
};

// Shared application state
const AppState = {
    currentParams: null,
    img2imgImage: null,
    vlEmbeddings: null,
    // ... mutable state
};
```

### API Communication

`api-client.js` centralizes all backend calls:

```javascript
const ApiClient = {
    generate: async (params) => { ... },
    img2img: async (params) => { ... },
    formatPrompt: async (params) => { ... },
    getHistory: async () => { ... },
    // ... all endpoints
};
```

### Module Exports

Traditional scripts export via `window`:

```javascript
// In history.js
function loadHistory() { ... }
window.loadHistory = loadHistory;

// In app.js
await loadHistory();  // Available globally
```

### Event Binding

Each module has an `init*Events()` function called from `app.js`:

```javascript
// In app.js
async function initializeApp() {
    initDOMReferences();
    initHistoryEvents();
    initTemplateEvents();
    initImg2ImgEvents();
    // ...
}
```

## Key Modules

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| state.js | Shared state | `initDOMReferences()` |
| api-client.js | Backend calls | `ApiClient.*` methods |
| ui-utils.js | Helpers | `escapeHtml()`, `debounce()`, `showError()`, `getResolution()` |
| history.js | History panel | `loadHistory()`, `reuseHistoryItem()` |
| templates.js | Templates | `loadTemplates()`, `onTemplateChange()` |
| rewriter.js | Prompt rewriting | `rewritePrompt()`, `loadRewriters()` |
| vl-conditioning.js | VL embeddings | `extractVLEmbeddings()`, `getVLParams()` |
| img2img.js | Img2img | `getImg2ImgParams()`, `initMaskCanvas()` |
| layer-blend.js | Layer blending | `getLayerWeights()`, `populateLayerDropdowns()` |
| advanced.js | Advanced controls | `getDypeConfig()`, `getSlgConfig()`, `getFmttConfig()` |
| qwen-image.js | Qwen-Image | `executeDecompose()`, `executeLayerEdit()` |
| resolution.js | Resolution selector | `ResolutionSelector.init()`, `.loadConstraints()`, `.getResolution()` |
| image-utils.js | Image workflows | `useAsImg2Img()`, `useAsVLReference()`, `setupDropzoneForInternalImages()` |
| config-manager.js | Config management | `ConfigManager.init()`, `.loadSessionConfig()`, `.applyToSession()` |
| app.js | Main entry | `initializeApp()`, `handleFormSubmit()` |

## CSS Organization

| File | Contents |
|------|----------|
| layout.css | Custom scrollbar, animations, responsive utilities |
| components.css | Modal styles, cards, buttons, dropzone, tooltips |
| forms.css | Range slider styling, input focus states |

Tailwind CSS (CDN) handles utility classes. Custom CSS is minimal.

## Security

- All user content rendered via `escapeHtml()` before innerHTML
- Event delegation instead of inline onclick handlers
- Base64 image data escaped for defense in depth

## Adding New Features

1. Create `web/static/js/feature.js`
2. Add DOM elements to `state.js` if needed
3. Create `initFeatureEvents()` function
4. Export functions via `window.functionName`
5. Add script tag to `index.html` (order matters)
6. Call `initFeatureEvents()` from `app.js`
