last updated: 2025-12-19

# Qwen-Image-Layered

Documentation for Qwen-Image-Layered image decomposition model support.

## Overview

Qwen-Image-Layered is an image decomposition model that takes an input image and text prompt, then generates multiple RGBA layers representing different elements or aspects of the scene. Unlike Z-Image which generates images from text, this model decomposes existing images into layered components.

## Model Architecture

| Component | Details |
|-----------|---------|
| Text encoder | Qwen2.5-VL-7B-Instruct (3584 hidden dim) |
| Vision encoder | Integrated in Qwen2.5-VL |
| DiT | 60-layer dual-stream transformer |
| VAE | 3D causal VAE (for multi-layer output) |
| Input | Image (RGB) + text prompt |
| Output | 1-7 RGBA layers (configurable) |

## Key Differences from Z-Image

| Aspect | Z-Image | Qwen-Image-Layered |
|--------|---------|-------------------|
| Task | Text-to-image generation | Image decomposition |
| Input | Text prompt | Image + text prompt |
| Output | Single RGB image | Multiple RGBA layers |
| Text encoder | Qwen3-4B (2560 dim) | Qwen2.5-VL-7B (3584 dim) |
| DiT layers | 28 + 2 refiner | 60 dual-stream |
| CFG scale | 0.0 (baked in) | 4.0 (required) |
| Steps | 8-9 (distilled) | 50 (non-distilled) |
| Resolution | Flexible (16 multiple) | Fixed (640/1024) |
| LoRA support | Yes | No |

## Resolution Constraints

Qwen-Image-Layered supports only two fixed resolutions:
- 640x640
- 1024x1024 (default)

Input images are automatically resized and center-cropped to match the target resolution.

## CFG Scale

Unlike Z-Image (which has CFG baked in via Decoupled-DMD), Qwen-Image-Layered requires explicit CFG for quality:
- Default: 4.0
- Range: 1.0 to 10.0
- Lower values may produce lower quality results

## Layer Count

The model can output 1-7 RGBA layers:
- Default: 7 layers
- Each layer represents a different scene element
- Layers use alpha channel for transparency
- Ordering is model-determined (not controllable)

### Latent Packing Strategy

Since the DiT expects a single latent tensor, multiple layers are packed:

| Layers | Packing Strategy |
|--------|-----------------|
| 1-4 | Direct 2x2 grid packing |
| 5-7 | Recursive: Pack first 4, then pack remaining with padding |

This is handled automatically by `pack_multi_layer_latents()` and `unpack_multi_layer_latents()`.

## Usage

### CLI

```bash
# Basic decomposition (7 layers, 1024x1024)
uv run scripts/generate.py \
  --model-type qwenimage \
  --qwen-image-model-path ~/Storage/Qwen_Qwen-Image-Layered \
  --img2img input.jpg \
  "A cheerful child waving under a blue sky"

# Custom layer count and resolution
uv run scripts/generate.py \
  --model-type qwenimage \
  --qwen-image-model-path ~/Storage/Qwen_Qwen-Image-Layered \
  --qwen-image-layers 4 \
  --qwen-image-resolution 640 \
  --img2img input.jpg \
  "Extract main elements"

# Custom CFG scale
uv run scripts/generate.py \
  --model-type qwenimage \
  --qwen-image-model-path ~/Storage/Qwen_Qwen-Image-Layered \
  --qwen-image-cfg-scale 6.0 \
  --img2img input.jpg \
  "Decompose into layers"
```

### Web API

```bash
# Decompose image
curl -X POST http://localhost:8000/api/qwen-image/decompose \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,...",
    "prompt": "A cheerful child waving",
    "num_layers": 7,
    "cfg_scale": 4.0,
    "resolution": 1024
  }'

# Get configuration
curl http://localhost:8000/api/qwen-image/config
```

Response format:
```json
{
  "layers": [
    {"layer": 0, "image": "data:image/png;base64,..."},
    {"layer": 1, "image": "data:image/png;base64,..."},
    ...
  ]
}
```

### Python API

```python
from llm_dit.pipelines.qwen_image import QwenImagePipeline
from PIL import Image

# Load pipeline
pipe = QwenImagePipeline.from_pretrained(
    "/path/to/Qwen-Image-Layered",
    device="cuda"
)

# Load input image
input_image = Image.open("photo.jpg")

# Decompose
result = pipe.decompose(
    image=input_image,
    prompt="A cheerful child waving under a blue sky",
    num_layers=7,
    cfg_scale=4.0,
    resolution=1024,
    num_inference_steps=50,
    seed=42
)

# Access layers (list of PIL Images with alpha channel)
for i, layer in enumerate(result.layers):
    layer.save(f"layer_{i}.png")
```

## Configuration

Add to `config.toml`:

```toml
[default.qwen_image]
model_path = "/path/to/Qwen-Image-Layered"
num_layers = 7
cfg_scale = 4.0
resolution = 1024

[default.qwen_image.generation]
steps = 50
seed = 42
```

## Technical Implementation

### Text Encoding

Uses Qwen2.5-VL-7B-Instruct's text model (not vision encoder):
1. Format prompt with Qwen2.5-VL chat template
2. Extract hidden states from layer -2 (3584 dim)
3. Pass to DiT as text conditioning

Note: Vision features from input image are handled separately by the DiT.

### VAE Architecture

3D causal VAE with:
- 3D convolutions for multi-layer processing
- Causal masking (doesn't look ahead in layer dimension)
- Encoder: Image -> latent
- Decoder: Latent -> RGBA layers

### DiT Architecture

60-layer dual-stream transformer:
- Text stream: Processes text embeddings
- Visual stream: Processes image features
- Cross-attention between streams
- Timestep conditioning via adaptive layer norm

## Limitations

1. Fixed resolutions only (640 or 1024)
2. Layer ordering not controllable
3. No LoRA support
4. Slower than Z-Image (50 steps vs 9)
5. Requires more VRAM than Z-Image

## Memory Requirements

Approximate VRAM usage (1024x1024, 7 layers):
- Text encoder: 14GB (Qwen2.5-VL-7B)
- DiT: 8GB (60 layers)
- VAE: 2GB (3D architecture)
- Total: 24GB+ recommended

Device placement recommendations:
- Text encoder: CPU (slow but saves VRAM)
- DiT: CUDA (main computation)
- VAE: CUDA (relatively small)

## Future Work

- Web UI support for layer visualization
- Layer editing and re-composition tools
- Support for layer-to-image generation
- Quantization support (4-bit/8-bit)
- Optimizations for faster inference
- Ablation studies on layer count impact

## Related Files

| File | Description |
|------|-------------|
| `src/llm_dit/backends/qwen_image.py` | Text encoder backend |
| `src/llm_dit/models/qwen_image_vae.py` | VAE wrapper |
| `src/llm_dit/models/qwen_image_dit.py` | DiT wrapper |
| `src/llm_dit/pipelines/qwen_image.py` | Main pipeline |
| `src/llm_dit/utils/latent_packing.py` | Latent packing utilities |
| `scripts/generate.py` | CLI script with Qwen-Image support |
| `web/server.py` | Web API endpoints |
