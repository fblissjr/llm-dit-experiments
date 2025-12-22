# resolution constraints

*last updated: 2025-12-22*

Z-Image requires image dimensions divisible by 16 (VAE constraint). All preset resolutions are pre-validated.

## constants

| Constant | Value | Description |
|----------|-------|-------------|
| `VAE_SCALE_FACTOR` | 8 | Latent to pixel ratio (`latent_dim = image_dim / 8`) |
| `VAE_MULTIPLE` | 16 | Required divisibility for image dimensions |
| `MIN_RESOLUTION` | 256 | Minimum recommended resolution |
| `MAX_RESOLUTION` | 4096 | Maximum recommended resolution |
| `DEFAULT_RESOLUTION` | 1024 | Default width/height |

## available presets (web ui)

| Category | Resolutions |
|----------|-------------|
| Square | 512, 768, 1024, 1280, 1536, 1920 |
| Landscape (16:9) | 1280x720 (HD), 1920x1088 (Full HD) |
| Portrait (9:16) | 720x1280 (HD), 1088x1920 (Full HD) |
| Mobile Landscape | 1024x576, 1280x576 |
| Mobile Portrait | 576x1024, 576x1280 |
| Classic (4:3) | 1024x768, 768x1024, 1280x960, 960x1280 |

## cli validation

The CLI (`scripts/generate.py`) automatically:
1. Validates dimensions are divisible by 16
2. Snaps invalid values to nearest valid resolution with a warning
3. Warns if resolution is below minimum or above maximum

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
