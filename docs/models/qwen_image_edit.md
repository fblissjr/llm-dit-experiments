last updated: 2025-12-22

# qwen-image-edit-2509

Documentation for Qwen-Image-Edit-2509, the instruction-based image editing model used alongside Qwen-Image-Layered.

## overview

Qwen-Image-Edit-2509 is an instruction-based image editing model from Alibaba/Qwen that modifies images based on natural language instructions. In this project, it is primarily used to edit individual RGBA layers produced by the Qwen-Image-Layered decomposition pipeline.

### key capabilities

- Edit images using natural language instructions
- Modify colors, textures, and visual attributes
- Add or remove elements
- Style transfer and artistic effects
- Preserves alpha channel when editing RGBA layers

## integration with qwen-image-layered

The edit model works as part of the layer editing workflow:

```
Input Image -> Qwen-Image-Layered (decompose) -> RGBA Layers
                                                      |
                                                      v
                               Qwen-Image-Edit-2509 (edit) -> Edited Layer
                                                      |
                                                      v
                                          Composite (optional)
```

### typical workflow

1. **Decompose** an image into layers using Qwen-Image-Layered
2. **Edit** individual layers with text instructions
3. **Composite** layers back together (optional)

## model details

| Component | Details |
|-----------|---------|
| Model ID | `Qwen/Qwen-Image-Edit-2509` |
| Architecture | DiT-based image editor |
| Input | RGB image + text instruction |
| Output | Edited RGB image |
| Default CFG | 4.0 |
| Default steps | 50 |

**Note:** The edit model works with RGB images. When editing RGBA layers, the alpha channel is preserved by extracting it before editing and reapplying it afterward.

## installation

The edit model auto-downloads from HuggingFace on first use:

```bash
# Auto-download (recommended)
# The model downloads when first calling edit_layer()

# Pre-download (optional)
huggingface-cli download Qwen/Qwen-Image-Edit-2509 --local-dir ~/Storage/Qwen_Qwen-Image-Edit-2509
```

## configuration

### config.toml

```toml
[default.qwen_image]
model_path = "/path/to/Qwen_Qwen-Image-Layered"
edit_model_path = ""  # empty = auto-download from HuggingFace
cpu_offload = true    # recommended for RTX 4090

# Or specify local path
edit_model_path = "/path/to/Qwen_Qwen-Image-Edit-2509"
```

### cli

```bash
# Use auto-download (default)
uv run scripts/generate.py \
  --model-type qwenimage \
  --qwen-image-model-path ~/Storage/Qwen_Qwen-Image-Layered \
  --img2img input.jpg \
  "Decomposition prompt"

# Specify local edit model
uv run scripts/generate.py \
  --model-type qwenimage \
  --qwen-image-model-path ~/Storage/Qwen_Qwen-Image-Layered \
  --qwen-image-edit-model-path ~/Storage/Qwen_Qwen-Image-Edit-2509 \
  --img2img input.jpg \
  "Decomposition prompt"
```

## usage

### python api

```python
from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline
from PIL import Image

# Load pipeline (edit model loads lazily on first use)
pipe = QwenImageDiffusersPipeline.from_pretrained(
    "/path/to/Qwen_Qwen-Image-Layered",
    cpu_offload=True,
)

# Decompose image into layers
input_image = Image.open("scene.png")
layers = pipe.decompose(
    image=input_image,
    prompt="A cheerful scene with a house and garden",
    layer_num=4,
)

# Edit a specific layer
edited_layer = pipe.edit_layer(
    layer_image=layers[1],  # Layer 0 is composite
    instruction="Change the house color to blue",
    num_inference_steps=50,
    cfg_scale=4.0,
)
edited_layer.save("edited_layer.png")

# Edit with seed for reproducibility
edited_layer = pipe.edit_layer(
    layer_image=layers[2],
    instruction="Add a sunset glow effect",
    seed=42,
)
```

### preloading the edit model

By default, the edit model loads lazily on first `edit_layer()` call. To preload:

```python
# Eager loading at pipeline initialization
pipe = QwenImageDiffusersPipeline.from_pretrained(
    "/path/to/Qwen_Qwen-Image-Layered",
    load_edit_model=True,  # Load edit model immediately
    cpu_offload=True,
)

# Or load later explicitly
pipe = QwenImageDiffusersPipeline.from_pretrained(...)
pipe.load_edit_model()  # Downloads and loads if needed
```

### web ui

The web UI provides an edit modal for each decomposed layer:

1. Generate a decomposition via the Qwen-Image tab
2. Click the edit icon on any layer
3. Enter an instruction (e.g., "Make it brighter")
4. Click "Apply Edit"
5. The edited layer replaces the original

### web api

```bash
# Edit a layer
curl -X POST http://localhost:8000/api/qwen-image/edit-layer \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/png;base64,...",
    "instruction": "Change the color to blue",
    "steps": 50,
    "cfg_scale": 4.0,
    "seed": 42
  }'
```

## parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `instruction` | (required) | Natural language editing instruction |
| `num_inference_steps` | 50 | Diffusion steps |
| `cfg_scale` | 4.0 | Classifier-free guidance scale |
| `seed` | None | Random seed for reproducibility |

## instruction examples

| Instruction | Effect |
|-------------|--------|
| "Change the color to blue" | Recolors the subject |
| "Make it more vibrant" | Increases saturation |
| "Add a warm glow" | Adds warm lighting effect |
| "Remove the background" | Attempts to isolate subject |
| "Make it look vintage" | Applies vintage color grading |
| "Add shadows" | Enhances shadow definition |
| "Simplify the details" | Reduces visual complexity |
| "Make it photorealistic" | Enhances realism |

## alpha channel handling

When editing RGBA layers from decomposition:

1. Alpha channel is extracted before editing
2. RGB channels are processed by the edit model
3. Original alpha is reapplied to the result
4. If resolution changes, alpha is resized to match

This ensures transparency information is preserved through the editing process.

```python
# Internal alpha handling (automatic)
# Original: RGBA layer
# Process: RGB -> Edit model -> RGB
# Result: RGB + original alpha = RGBA
```

## memory requirements

| Component | VRAM |
|-----------|------|
| Edit model (DiT) | ~10 GB |
| With CPU offload | ~5 GB peak |

**Recommendation:** Enable CPU offload when using both decompose and edit models together:

```python
pipe = QwenImageDiffusersPipeline.from_pretrained(
    model_path,
    cpu_offload=True,  # Sequential CPU offload
)
```

## limitations

1. **RGB input only:** RGBA is converted to RGB for editing (alpha preserved separately)
2. **No mask support:** Edits apply to entire image, not regions
3. **Instruction quality:** Vague instructions may produce inconsistent results
4. **Resolution preservation:** Large changes may require more steps
5. **VRAM usage:** Requires 10+ GB without CPU offload

## error handling

```python
# Check if edit model is loaded
if not pipe.has_edit_model:
    pipe.load_edit_model()  # Loads on demand

# Handle download failures
try:
    edited = pipe.edit_layer(layer, "instruction")
except Exception as e:
    print(f"Edit failed: {e}")
    # Common issues:
    # - Network timeout during download
    # - Insufficient disk space
    # - VRAM exhaustion
```

## related files

| File | Description |
|------|-------------|
| `src/llm_dit/pipelines/qwen_image_diffusers.py` | Pipeline wrapper |
| `web/server.py` | Web API endpoint (`/api/qwen-image/edit-layer`) |
| `docs/models/qwen_image_layered.md` | Decomposition model docs |
| `docs/qwen_image_guide.md` | Complete Qwen-Image guide |

## see also

- [Qwen-Image-Layered](qwen_image_layered.md) - Decomposition model
- [Qwen-Image Guide](../qwen_image_guide.md) - Complete workflow guide
- [HuggingFace Model](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) - Official model page
