# resolution constraints

*last updated: 2025-12-23*

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

## available presets (web ui)

The web UI provides categorized presets with filter tabs (All / Square / Landscape / Portrait).

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

## dype auto-detection

DyPE (Dynamic Position Extrapolation) is automatically recommended when `max(width, height) > 1024`.

The web UI:
- Shows a "DyPE recommended" indicator below the resolution dropdown
- Auto-enables DyPE checkbox when high-res is selected
- Auto-sets optimal exponent based on scale factor:

| Scale Factor | Resolution Range | Exponent | Description |
|--------------|------------------|----------|-------------|
| <= 1.0 | <= 1024px | N/A | DyPE not needed |
| 1.0 - 1.5 | 1024-1536px | 0.5 | Gentle extrapolation |
| 1.5 - 3.0 | 1536-3072px | 1.0 | Standard |
| >= 3.0 | >= 3072px | 2.0 | Aggressive (4K+) |

## cli validation

The CLI (`scripts/generate.py`) automatically:
1. Validates dimensions are divisible by 16
2. Snaps invalid values to nearest valid resolution with a warning
3. Warns if resolution is below minimum or above maximum

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
      "default": true,
      "dype": {"recommended": false, "exponent": null}
    },
    ...
  ]
}
```

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
