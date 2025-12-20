# qwen-image-layered user guide

last updated: 2025-12-20

## overview

Qwen-Image-Layered is a diffusion transformer model that decomposes images into semantically disentangled RGBA layers. Unlike traditional segmentation, it uses a learned latent-space approach to extract layers that can be individually edited and recombined.

### capabilities

| Task | Description |
|------|-------------|
| Image Decomposition | Split an image into 2-10 RGBA layers |
| Layer Editing | Modify individual layers with text instructions |
| Layer Recombination | Composite layers back into a final image |

### key specifications

| Parameter | Value |
|-----------|-------|
| Text Encoder | Qwen2.5-VL-7B-Instruct (3584 hidden dim) |
| DiT Architecture | 60-layer dual-stream MMDiT |
| Resolution | 640x640 (recommended) or 1024x1024 |
| Layers | 2-10 (default: 4) |
| CFG Scale | 4.0 (required for quality) |
| Steps | 50 (default) |
| VRAM | ~5 GB with CPU offload |

---

## installation

### 1. download model weights

Download Qwen-Image-Layered from HuggingFace:

```bash
# Using huggingface-cli
huggingface-cli download Qwen/Qwen-Image-Layered --local-dir ~/Storage/Qwen_Qwen-Image-Layered

# Or using git lfs
git lfs install
git clone https://huggingface.co/Qwen/Qwen-Image-Layered ~/Storage/Qwen_Qwen-Image-Layered
```

The edit model (Qwen-Image-Edit-2509) is downloaded automatically on first use, or you can pre-download:

```bash
huggingface-cli download Qwen/Qwen-Image-Edit-2509 --local-dir ~/Storage/Qwen_Qwen-Image-Edit-2509
```

### 2. configure the project

Add to your `config.toml`:

```toml
[default.qwen_image]
model_path = "/path/to/Qwen_Qwen-Image-Layered"
edit_model_path = ""  # Leave empty for auto-download from HuggingFace
cpu_offload = true    # Recommended for RTX 4090
layer_num = 4
num_inference_steps = 50
cfg_scale = 4.0
resolution = 640      # 640 produces better quality
```

---

## quick start

### web ui

1. Start the server with Qwen-Image mode:

```bash
uv run web/server.py \
  --model-type qwenimage \
  --qwen-image-model-path /path/to/Qwen_Qwen-Image-Layered
```

2. Open http://localhost:7860 in your browser

3. Select "Qwen-Image" mode (top right toggle)

4. Upload an image and enter a scene description

5. Click "Decompose Image"

6. Hover over any layer and click "Edit" to modify it

### python api

```python
from llm_dit.pipelines import QwenImageDiffusersPipeline
from PIL import Image

# Load pipeline
pipe = QwenImageDiffusersPipeline.from_pretrained(
    "/path/to/Qwen_Qwen-Image-Layered",
    cpu_offload=True,  # Recommended for RTX 4090
)

# Decompose an image
input_image = Image.open("scene.png")
layers = pipe.decompose(
    image=input_image,
    prompt="A cheerful child waving under a blue sky",
    layer_num=4,
    resolution=640,
    num_inference_steps=50,
    cfg_scale=4.0,
)

# Save layers
for i, layer in enumerate(layers):
    layer.save(f"layer_{i}.png")

# Edit a specific layer
edited = pipe.edit_layer(
    layer_image=layers[2],
    instruction="Change the shirt color to blue",
    num_inference_steps=50,
    cfg_scale=4.0,
)
edited.save("edited_layer_2.png")
```

### cli

```bash
# Run the integration test
uv run python tests/integration/test_qwen_official.py \
  --model-path /path/to/Qwen_Qwen-Image-Layered \
  --input-image experiments/inputs/test_scene.png \
  --prompt "A scene with objects" \
  --num-layers 4 \
  --resolution 640 \
  --output-dir experiments/results/qwen_test
```

---

## decomposition workflow

### how it works

1. **Input Processing**: Your image is resized to the target resolution (640 or 1024)

2. **VAE Encoding**: Image is encoded to a 16-channel latent with 2x2 packing (64 packed channels)

3. **Layer Decomposition**: The DiT generates N latent representations, one per layer

4. **VAE Decoding**: Each latent is decoded to an RGBA image

### output format

The decomposition returns N+1 images:
- **Layer 0**: Composite (reconstructed input)
- **Layers 1-N**: Decomposed RGBA layers with alpha channels

### best practices

**Resolution**: Use 640x640 for best quality. The model was trained on 640 as the primary resolution.

**Layer count**:
- 2-4 layers: Works best for simple scenes
- 5-7 layers: Good for complex scenes with many objects
- 8-10 layers: May produce thin/incomplete layers

**Prompting**: Describe the scene, not the desired layers. The model decides how to decompose based on semantic understanding.

```
# Good prompt
"A cheerful child waving under a blue sky with clouds"

# Less effective
"Separate the child from the background"
```

---

## layer editing

### capabilities

The edit model can:
- Change colors ("Make the shirt blue")
- Add patterns ("Add polka dots to the dress")
- Modify textures ("Make it look metallic")
- Transform objects ("Turn the cat into a dog")

### limitations

- Works best on isolated layers (not composite)
- Large structural changes may fail
- Respects the alpha channel boundary

### example edits

```python
# Color change
edited = pipe.edit_layer(
    layer_image=layer,
    instruction="Change the color to vibrant red",
)

# Texture modification
edited = pipe.edit_layer(
    layer_image=layer,
    instruction="Add a wood grain texture",
)

# Style transfer
edited = pipe.edit_layer(
    layer_image=layer,
    instruction="Make it look like a watercolor painting",
)
```

---

## api reference

### rest endpoints

#### POST /api/qwen-image/decompose

Decompose an image into layers.

**Request:**
```json
{
  "image": "data:image/png;base64,...",
  "prompt": "A scene description",
  "layer_num": 4,
  "resolution": 640,
  "steps": 50,
  "cfg_scale": 4.0,
  "seed": null
}
```

**Response:**
```json
{
  "layers": [
    {"name": "Composite", "image": "data:image/png;base64,...", "index": 0},
    {"name": "Layer 1", "image": "data:image/png;base64,...", "index": 1},
    ...
  ],
  "zip_data": "base64-encoded-zip",
  "generation_time": 45.2,
  "layer_count": 5,
  "resolution": 640
}
```

#### POST /api/qwen-image/edit-layer

Edit a decomposed layer.

**Request:**
```json
{
  "layer_image": "data:image/png;base64,...",
  "instruction": "Change color to blue",
  "steps": 50,
  "cfg_scale": 4.0,
  "seed": null
}
```

**Response:** PNG image stream

#### GET /api/qwen-image/config

Get current Qwen-Image configuration.

#### GET /api/qwen-image/status

Check if pipeline is loaded.

#### GET /api/qwen-image/edit-status

Check if edit model is loaded.

---

## cli reference

| Flag | Description | Default |
|------|-------------|---------|
| `--model-type qwenimage` | Select Qwen-Image mode | zimage |
| `--qwen-image-model-path` | Path to model | (required) |
| `--qwen-image-edit-model-path` | Path to edit model | (auto-download) |
| `--qwen-image-cpu-offload` | Enable CPU offload | true |
| `--qwen-image-layers` | Number of layers | 4 |
| `--qwen-image-steps` | Diffusion steps | 50 |
| `--qwen-image-cfg-scale` | CFG scale | 4.0 |
| `--qwen-image-resolution` | Output resolution | 640 |

---

## memory optimization

### recommended settings for rtx 4090 (24gb)

```toml
[rtx4090.qwen_image]
cpu_offload = true     # Keeps VRAM usage at ~5 GB
resolution = 640       # Lower resolution uses less memory
layer_num = 4          # More layers = more decoding passes
```

### memory usage breakdown

| Component | VRAM (with offload) |
|-----------|---------------------|
| DiT inference | ~4 GB |
| VAE decode (per layer) | ~1 GB |
| Text encoder | CPU |
| **Total (4 layers)** | **~5 GB** |

Without CPU offload, the model requires ~20+ GB VRAM.

---

## troubleshooting

### "Pipeline does not support layer editing"

You're using the old `QwenImagePipeline` instead of `QwenImageDiffusersPipeline`. The diffusers wrapper is required for editing.

### "Failed to load Qwen-Image pipeline"

Check that:
1. The model path exists and contains all required files
2. You have enough disk space for the edit model (~20 GB)
3. The coderef/diffusers fork is properly set up

### "CUDA out of memory"

Enable CPU offload:
```python
pipe = QwenImageDiffusersPipeline.from_pretrained(
    model_path,
    cpu_offload=True,  # Required for most GPUs
)
```

Or use 640 resolution instead of 1024.

### layers are thin or incomplete

Try reducing the layer count. More layers means each layer contains less content.

### alpha channel is missing

Ensure the input image is RGBA. The pipeline converts RGB to RGBA automatically, but the alpha channel is generated by the model.

---

## advanced usage

### layer compositing

Recombine layers after editing:

```python
from PIL import Image

def composite_layers(layers):
    """Composite RGBA layers in order (back to front)."""
    result = Image.new('RGBA', layers[0].size, (0, 0, 0, 0))
    for layer in reversed(layers):
        result = Image.alpha_composite(result, layer)
    return result

# Edit layer 2, then recombine
layers[2] = edited_layer
final = composite_layers(layers[1:])  # Skip composite layer
```

### batch processing

```python
import os
from pathlib import Path

input_dir = Path("inputs")
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

for img_path in input_dir.glob("*.png"):
    image = Image.open(img_path)
    layers = pipe.decompose(image, prompt="A scene", layer_num=4)

    # Save layers
    for i, layer in enumerate(layers):
        layer.save(output_dir / f"{img_path.stem}_layer_{i}.png")
```

### custom prompting

The prompt influences layer semantics. Experiment with different descriptions:

```python
# Object-focused decomposition
layers = pipe.decompose(image, prompt="A dog sitting next to a red ball")

# Scene-focused decomposition
layers = pipe.decompose(image, prompt="A park scene with trees and grass")

# Abstract decomposition
layers = pipe.decompose(image, prompt="An artistic composition with shapes and colors")
```

---

## references

- [arxiv technical paper (2512.15603v1)](https://arxiv.org/html/2512.15603v1)
- [huggingface model: qwen/qwen-image-layered](https://huggingface.co/Qwen/Qwen-Image-Layered)
- [huggingface model: qwen/qwen-image-edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)
- [internal technical report](../internal/research/qwen_image_technical_report.md)
