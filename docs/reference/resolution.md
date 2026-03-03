# resolution constraints

*last updated: 2026-03-03*

Z-Image requires image dimensions divisible by 16 (VAE constraint). All preset resolutions are pre-validated.

## constants

| Constant | Value | Description |
|----------|-------|-------------|
| `VAE_SCALE_FACTOR` | 8 | Latent to pixel ratio (`latent_dim = image_dim / 8`) |
| `VAE_MULTIPLE` | 16 | Required divisibility for image dimensions |
| `MIN_RESOLUTION` | 256 | Minimum recommended resolution |
| `MAX_RESOLUTION` | 4096 | Maximum recommended resolution |
| `DEFAULT_RESOLUTION` | 1024 | Default width/height |
| `DYPE_BASE_RESOLUTION` | 1024 | Z-Image training resolution (DyPE threshold) |

## web ui resolution selector

The web UI provides a comprehensive resolution selector (`web/static/js/resolution.js`) with:

### controls

| Control | Description |
|---------|-------------|
| Width/Height inputs | Numeric inputs with VAE snapping (auto-rounds to multiple of 16) |
| Aspect filter icons | Filter presets by aspect category (square, landscape, portrait, mobile) |
| Aspect lock button | Lock ratio - changing one dimension scales the other |
| Preset chips | Quick-select buttons, filtered by active aspect category |
| DyPE hint | Shown when resolution exceeds 1024px (Z-Image only) |

### aspect categories

| Category | Ratio Range | Examples |
|----------|-------------|----------|
| square | 0.95 - 1.05 | 1:1 |
| landscape | 1.05 - 2.0 | 16:9, 3:2, 4:3, 21:9 |
| portrait | 0.5 - 0.95 | 9:16, 2:3, 3:4 |
| mobile-landscape | > 2.0 | 19.5:9, 20:9 (phone screens horizontal) |
| mobile-portrait | < 0.5 | 9:19.5, 9:20 (phone screens vertical) |

### model-specific behavior

| Model | Mode | Min | Max | Notes |
|-------|------|-----|-----|-------|
| Z-Image | Flexible | 256 | 4096 | Any valid resolution, DyPE hints |
| Qwen-Image-Layered | Fixed | 640 | 1024 | Only 640x640 or 1024x1024 allowed |
| Qwen-Image T2I | Flexible | 256 | 2048 | Default 1328, no DyPE |

In fixed mode (Qwen-Image-Layered), the width/height inputs are disabled and only preset chips are selectable.

### javascript api

```javascript
// Initialize (called by app.js on page load)
await ResolutionSelector.init();

// Load constraints when switching models
await ResolutionSelector.loadConstraints('zimage');

// Get current resolution
const { width, height } = ResolutionSelector.getResolution();

// Set resolution programmatically
ResolutionSelector.setResolution(1920, 1080);

// Validate current resolution against constraints
const isValid = ResolutionSelector.validate();
```

## available presets (web ui)

The web UI provides categorized presets with filter tabs.

### square (1:1)

| Resolution | Notes |
|------------|-------|
| 512x512 | Fast preview |
| 768x768 | |
| 1024x1024 | Default (native) |
| 1280x1280 | DyPE recommended |
| 1536x1536 | DyPE recommended |
| 1920x1920 | DyPE recommended |
| 2048x2048 | 2K, DyPE recommended |

### landscape

| Ratio | Resolutions |
|-------|-------------|
| 16:9 | 1280x720 (720p), 1920x1088 (1080p), 2560x1440 (1440p) |
| 3:2 | 1536x1024, 1920x1280 |
| 4:3 | 1024x768, 1280x960, 1600x1200 |
| 21:9 | 1792x768 (Ultrawide), 2560x1088 (UW 1080) |

### portrait

| Ratio | Resolutions |
|-------|-------------|
| 9:16 | 720x1280 (720p), 1088x1920 (1080p), 1440x2560 (1440p) |
| 2:3 | 1024x1536, 1280x1920 |
| 3:4 | 768x1024, 960x1280, 1200x1600 |

### mobile (phone screens)

| Orientation | Ratio | Resolutions |
|-------------|-------|-------------|
| Landscape | 19.5:9 | 2340x1080 (Phone HD), 2796x1290 (iPhone Pro Max) |
| Portrait | 9:19.5 | 1080x2340 (Phone HD), 1290x2796 (iPhone Pro Max) |

These are common smartphone screen resolutions. Useful for wallpapers and mobile app mockups.

## dype recommendation

DyPE (Dynamic Position Extrapolation) is recommended when `max(width, height) > 1024`.

The web UI shows a "DyPE recommended" indicator below the resolution dropdown when a high-resolution preset is selected. Users must explicitly enable DyPE in the settings panel.

Suggested exponent values based on scale factor:

| Scale Factor | Resolution Range | Exponent | Description |
|--------------|------------------|----------|-------------|
| <= 1.0 | <= 1024px | N/A | DyPE not needed |
| 1.0 - 1.5 | 1024-1536px | 0.5 | Gentle extrapolation |
| 1.5 - 3.0 | 1536-3072px | 1.0 | Standard |
| >= 3.0 | >= 3072px | 2.0 | Aggressive (4K+) |

## cli validation

Resolution validation happens server-side. When using `scripts/gen.py`, pass `--width` and `--height` and the server validates:

```bash
uv run scripts/gen.py zimage --prompt "A landscape" --width 1920 --height 1088
```

The server automatically:
1. Validates dimensions are divisible by 16
2. Snaps invalid values to nearest valid resolution with a warning
3. Warns if resolution is below minimum or above maximum

> **Note:** The deprecated `scripts/generate.py` performed the same validation client-side.

## api endpoint

Resolution presets are served by `/api/resolution-config`:

```json
{
  "vae_multiple": 16,
  "vae_scale_factor": 8,
  "min_resolution": 256,
  "max_resolution": 4096,
  "default_resolution": 1024,
  "dype_base_resolution": 1024,
  "categories": ["square", "landscape", "portrait"],
  "presets": [
    {
      "value": "1024x1024",
      "label": "1024",
      "width": 1024,
      "height": 1024,
      "category": "square",
      "ratio": "1:1",
      "aspect_category": "square",
      "default": true,
      "dype": {"recommended": false, "exponent": null}
    },
    {
      "value": "1080x2340",
      "label": "Phone HD",
      "width": 1080,
      "height": 2340,
      "category": "portrait",
      "ratio": "19.5:9",
      "aspect_category": "mobile-portrait",
      "default": false,
      "dype": {"recommended": true, "exponent": 2.0}
    },
    ...
  ]
}
```

The `aspect_category` field is used by the frontend resolution selector for filtering. Categories are: `square`, `landscape`, `portrait`, `mobile-landscape`, `mobile-portrait`.

## helper functions

Available in `llm_dit.constants` for programmatic use:

```python
from llm_dit.constants import (
    VAE_MULTIPLE,
    snap_to_multiple,
    validate_resolution,
    calculate_latent_size,
)

# Snap to nearest valid resolution
snap_to_multiple(1000)  # -> 1008 (nearest multiple of 16)

# Validate resolution
is_valid, error = validate_resolution(1024, 768)  # -> (True, "")

# Calculate latent dimensions
latent_w, latent_h = calculate_latent_size(1024, 1024)  # -> (128, 128)
```
