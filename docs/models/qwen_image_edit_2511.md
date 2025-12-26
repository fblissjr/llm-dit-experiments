last updated: 2025-12-26

# qwen-image-edit-2511

Documentation for Qwen-Image-Edit-2511, the instruction-based image editing model used alongside Qwen-Image-Layered.

## overview

Qwen-Image-Edit-2511 is an enhanced version of the instruction-based image editing model from Alibaba/Qwen. Key improvements over the previous 2509 version:

- Mitigated image drift
- Improved character consistency
- Multi-person consistency for group photos
- Built-in LoRA capabilities (lighting enhancement, new viewpoints)
- Enhanced industrial design generation
- Strengthened geometric reasoning

### key capabilities

- Edit single images using natural language instructions
- Combine multiple images into coherent compositions (new in 2511)
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
                               Qwen-Image-Edit-2511 (edit) -> Edited Layer
                                                      |
                                                      v
                                          Composite (optional)
```

### typical workflow

1. **Decompose** an image into layers using Qwen-Image-Layered
2. **Edit** individual layers with text instructions
3. **Composite** layers back together (optional)

### multi-image workflow (new in 2511)

```
Image 1 + Image 2 + ... -> Qwen-Image-Edit-2511 (edit_multi) -> Combined Image
```

Use cases:
- Combine multiple person photos into a group shot
- Merge subject from one image with background from another
- Create creative compositions from multiple source images

## model details

| Component | Details |
|-----------|---------|
| Model ID | `Qwen/Qwen-Image-Edit-2511` |
| Architecture | DiT-based image editor |
| Input | RGB image(s) + text instruction |
| Output | Edited RGB image |
| Default CFG | 4.0 |
| Default steps | 40 (was 50 for 2509) |

**Note:** The edit model works with RGB images. When editing RGBA layers, the alpha channel is preserved by extracting it before editing and reapplying it afterward.

## installation

The edit model auto-downloads from HuggingFace on first use:

```bash
# Auto-download (recommended)
# The model downloads when first calling edit_layer() or edit_multi()

# Pre-download (optional)
huggingface-cli download Qwen/Qwen-Image-Edit-2511 --local-dir ~/Storage/Qwen_Qwen-Image-Edit-2511
```

## configuration

### config.toml

```toml
[default.qwen_image]
model_path = "/path/to/Qwen_Qwen-Image-Layered"
edit_model_path = ""  # empty = auto-download from HuggingFace
cpu_offload = true    # recommended for RTX 4090

# Or specify local path
edit_model_path = "/path/to/Qwen_Qwen-Image-Edit-2511"
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
  --qwen-image-edit-model-path ~/Storage/Qwen_Qwen-Image-Edit-2511 \
  --img2img input.jpg \
  "Decomposition prompt"
```

## usage

### python api - single image editing

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
    num_inference_steps=40,
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

### python api - multi-image editing (new in 2511)

```python
from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline
from PIL import Image

pipe = QwenImageDiffusersPipeline.from_pretrained(
    "/path/to/Qwen_Qwen-Image-Layered",
    cpu_offload=True,
)

# Load multiple images
person1 = Image.open("person1.jpg")
person2 = Image.open("person2.jpg")

# Combine into group photo
combined = pipe.edit_multi(
    images=[person1, person2],
    instruction="Both people standing together in a park, natural lighting",
    num_inference_steps=40,
    cfg_scale=4.0,
    seed=42,
)
combined.save("group_photo.png")

# Multiple images with creative instruction
landscape = Image.open("landscape.jpg")
subject = Image.open("subject.jpg")
style = Image.open("style_reference.jpg")

result = pipe.edit_multi(
    images=[landscape, subject, style],
    instruction="Place the subject in the landscape with the artistic style",
)
```

### preloading the edit model

By default, the edit model loads lazily on first `edit_layer()` or `edit_multi()` call. To preload:

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

### web api - single image

```bash
# Edit a layer
curl -X POST http://localhost:8000/api/qwen-image/edit-layer \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/png;base64,...",
    "instruction": "Change the color to blue",
    "steps": 40,
    "cfg_scale": 4.0,
    "seed": 42
  }'
```

### web api - multi-image (new in 2511)

```bash
# Combine multiple images
curl -X POST http://localhost:8000/api/qwen-image/edit-multi \
  -H "Content-Type: application/json" \
  -d '{
    "images": [
      "data:image/png;base64,...",
      "data:image/png;base64,..."
    ],
    "instruction": "Both people standing together at a beach",
    "steps": 40,
    "cfg_scale": 4.0,
    "seed": 42
  }'
```

## parameters

### edit_layer() parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `layer_image` | (required) | PIL Image to edit |
| `instruction` | (required) | Natural language editing instruction |
| `num_inference_steps` | 40 | Diffusion steps |
| `cfg_scale` | 4.0 | Classifier-free guidance scale |
| `seed` | None | Random seed for reproducibility |

### edit_multi() parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `images` | (required) | List of 2+ PIL Images to combine |
| `instruction` | (required) | How to combine the images |
| `num_inference_steps` | 40 | Diffusion steps |
| `cfg_scale` | 4.0 | Classifier-free guidance scale |
| `seed` | None | Random seed for reproducibility |

## instruction examples

### single image editing

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

### multi-image combining (2511)

| Instruction | Use Case |
|-------------|----------|
| "Both people standing together in a park" | Group photo |
| "The person in the first image sitting in the room from the second" | Subject + background |
| "All items arranged on the table" | Product composition |
| "The magician bear and alchemist bear facing each other in the square" | Character interaction |

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

## multi-image handling

When using edit_multi():

1. All images are converted to RGB (RGBA composited onto white)
2. Images are passed as a list to the pipeline
3. The model combines them based on the instruction
4. Returns a single RGB output image

```python
# Input: [RGB or RGBA images]
# Process: Convert to RGB list -> Edit model -> Combined RGB
# Result: Single combined image
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

## differences from 2509

| Feature | 2509 | 2511 |
|---------|------|------|
| Default steps | 50 | 40 |
| Multi-image support | No | Yes (`edit_multi()`) |
| Character consistency | Good | Improved |
| Multi-person consistency | Limited | Improved |
| Built-in LoRAs | No | Yes (lighting, viewpoints) |
| Image drift | Some | Mitigated |

## limitations

1. **RGB input only:** RGBA is converted to RGB for editing (alpha preserved separately)
2. **No mask support:** Edits apply to entire image, not regions
3. **Instruction quality:** Vague instructions may produce inconsistent results
4. **Resolution preservation:** Large changes may require more steps
5. **VRAM usage:** Requires 10+ GB without CPU offload
6. **Multi-image limit:** Practical limit of 2-4 images for best results

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

# Handle multi-image validation
try:
    combined = pipe.edit_multi([single_image], "instruction")
except ValueError as e:
    print(f"Validation error: {e}")
    # Requires at least 2 images
```

## related files

| File | Description |
|------|-------------|
| `src/llm_dit/pipelines/qwen_image_diffusers.py` | Pipeline wrapper |
| `web/server.py` | Web API endpoints |
| `docs/models/qwen_image_layered.md` | Decomposition model docs |
| `docs/qwen_image_guide.md` | Complete Qwen-Image guide |

## see also

- [Testing Guide](qwen_image_edit_2511_testing.md) - Complete testing documentation
- [Qwen-Image-Layered](qwen_image_layered.md) - Decomposition model
- [Qwen-Image Guide](../qwen_image_guide.md) - Complete workflow guide
- [HuggingFace Model](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) - Official model page
- [Official Blog](https://qwenlm.github.io/blog/qwen-image-edit-2511/) - Qwen team blog post
