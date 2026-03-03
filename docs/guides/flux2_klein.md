# flux.2 klein image generation guide

*last updated: 2026-03-03*

**Status: PRODUCTION READY** - Text-to-image and multi-reference image editing both work.

FLUX.2 Klein is Black Forest Labs' latest image generation model, using a Qwen3 text encoder with 4D RoPE for spatial-temporal attention. Available in 4B and 9B variants, with FP8 quantized versions for lower VRAM.

## model variants

| Variant | Parameters | VRAM (Block Offload) | Steps | CFG | Use Case |
|---------|-----------|---------------------|-------|-----|----------|
| klein-9b-fp8 | 9B | ~12GB | 4 | 1.0 | **Recommended** - Best quality/speed |
| klein-9b | 9B | ~20GB | 4 | 1.0 | Highest quality, needs 24GB+ |
| klein-4b-fp8 | 4B | ~8GB | 4 | 1.0 | Consumer GPUs (16GB) |
| klein-4b | 4B | ~14GB | 4 | 1.0 | Smaller model, BF16 |
| klein-base-9b-fp8 | 9B | ~12GB | 50 | 4.0 | Non-distilled, more control |
| klein-base-9b | 9B | ~20GB | 50 | 4.0 | Base model, full precision |
| klein-base-4b-fp8 | 4B | ~8GB | 50 | 4.0 | Smaller base model |
| klein-base-4b | 4B | ~14GB | 50 | 4.0 | Smallest base model |

**Distilled vs Base:**
- **Distilled** (klein-*): 4 steps, CFG=1.0 baked in, fast generation
- **Base** (klein-base-*): 50 steps, CFG=4.0, more control over generation

## cli usage

The CLI client `scripts/gen.py` requires the server to be running. Model paths, block offload, and other infrastructure settings are configured server-side in `config.toml`.

```bash
# Start the server (once)
uv run web/server.py --config config.toml
```

### basic text-to-image

```bash
uv run scripts/gen.py flux2 --prompt "A photo of a cat sitting on a windowsill" \
    --width 1024 --height 1024 --seed 42
```

### with model switching

```bash
# Use a specific model variant (must be available on server)
uv run scripts/gen.py flux2 --prompt "A sunset over mountains" \
    --model-name klein-9b-fp8
```

### image editing with references

```bash
# Single reference - transform existing image
uv run scripts/gen.py flux2 --prompt "Transform the cat into a black cat" \
    --reference-images cat.png

# Multiple references - combine elements
uv run scripts/gen.py flux2 --prompt "Place the cat from image 1 on the mountain from image 2" \
    --reference-images cat.png mountain.png
```

### streaming progress

```bash
uv run scripts/gen.py flux2 --prompt "A detailed landscape" --stream
```

## cli arguments (gen.py flux2)

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | (required) | Generation prompt |
| `--width` | (config) | Image width |
| `--height` | (config) | Image height |
| `--num-steps` | (config) | Override denoising steps |
| `--guidance` | (config) | Override CFG scale |
| `--seed` | (random) | Reproducibility seed |
| `--model-name` | (config) | Model variant (see table above) |
| `--loras` | None | LoRA specs (path:scale) |
| `--block-offload` | (config) | Enable block-by-block GPU offloading |
| `--reference-images` | None | Reference image path(s) for editing |
| `--upsample-prompt` | (config) | Enable prompt upsampling |
| `--stream` | False | Use SSE streaming for progress |

> **Note:** Model paths (`model_path`, `vae_path`) and other infrastructure settings are server-side config in `config.toml`, not gen.py flags.

## web ui usage

### starting the server

```bash
uv run web/server.py --port 8080
```

### text-to-image generation

1. Click the **FLUX.2** button in the model selector (orange accent)
2. Enter your prompt in the text area
3. Select model variant from dropdown
   - Distilled models auto-set 4 steps, CFG=1.0
   - Base models auto-set 50 steps, CFG=4.0
4. Choose resolution (default: 1024x1024)
5. Click **Generate Image**
6. Download result when complete

### image editing with references

1. Expand **"Image Editing (Reference Images)"** section
2. Drag & drop or click to upload 1-4 reference images
3. Previews appear with remove buttons on hover
4. Write a prompt describing how to combine/edit the images
5. Click **Generate Image**

**Example prompts for multi-reference:**
- "Combine the subject from image 1 with the background from image 2"
- "Place the cat next to the flowers with the mountain view behind"
- "Transform the person's outfit to match the style in image 2"

### advanced options

Expand **"Advanced Options"** for:
- **Block Offload**: Enable for GPUs < 24GB VRAM (slower but fits)
- **Custom Model Path**: Use local weights instead of HuggingFace
- **Custom VAE Path**: Use local VAE instead of HuggingFace

## memory management

### block offload mode

For GPUs with less than 24GB VRAM, enable block offload in `config.toml`:

```toml
[rtx4090.flux2]
block_offload = true
```

Or pass `--block-offload` to gen.py. This moves transformer blocks to/from GPU one at a time during inference:
- Peak VRAM: ~12-15GB (vs ~22GB without)
- Speed: ~2x slower due to CPU-GPU transfers
- Works on RTX 3080/4070/4080 (16GB+ recommended)

### vram unloading

In the web UI, FLUX.2 can be unloaded via the VRAM management panel to free memory for other models.

API endpoint: `POST /api/models/flux2/unload`

## technical details

### 4d rope position encoding

FLUX.2 uses 4D rotary position embeddings with coordinates `(t, h, w, l)`:

| Token Type | t | h | w | l |
|------------|---|---|---|---|
| Text | 0 | 0 | 0 | sequence_pos |
| Generated Image | 0 | row | col | 0 |
| Reference Image N | 10×N | row | col | 0 |

This temporal separation (`t_scale=10`) allows the model to distinguish multiple reference images in attention.

### fp8 quantization

FP8 models use per-tensor scale factors for proper dequantization:
```
actual_weight = fp8_value × scale_factor
```

The loader automatically applies scales when casting FP8 to BF16 for inference.

### architecture

- **Text Encoder**: Qwen3-8B (9B models) or Qwen3-4B (4B models)
- **Transformer**: 8 double-stream + 24 single-stream blocks (9B)
- **VAE**: 16x spatial compression (8x encoder + 2x patchify)
- **Latent**: 128 channels after patchify

## toml configuration

```toml
[rtx4090]
# Auto-load FLUX.2 at web server startup (optional)
default_pipeline = "flux2"  # Options: none, z-image, qwen-image, flux2, ltx2

[rtx4090.flux2]
model_path = "models/FLUX.2-klein/FLUX.2-klein-9b-fp8"
vae_path = "models/FLUX.2-klein/FLUX.2-klein-9B"
default_model = "klein-9b-fp8"
block_offload = true
default_steps = 4
default_guidance = 1.0
```

The `default_pipeline` setting in the profile's main section controls which model automatically loads when the web server starts. Set to "flux2" to preload FLUX.2, or "none" to start without loading any model (saves memory at startup).

See [config_management.md](config_management.md) for detailed TOML configuration documentation.

## troubleshooting

### out of memory (OOM)

1. Enable block offload: set `block_offload = true` in `config.toml` or pass `--block-offload` to gen.py
2. Use FP8 variant: `klein-9b-fp8` instead of `klein-9b`
3. Use smaller model: `klein-4b-fp8`
4. Reduce resolution: 768x768 instead of 1024x1024
5. Unload other models first (Z-Image, Qwen-Image, LTX-2)

### pure noise output

If generation produces static noise instead of images:
1. Verify using correct model files (not Diffusers format)
2. Check FP8 models have scale tensors (`.weight_scale` files)
3. Ensure block offload is properly configured

### slow generation

Block offload mode is inherently slower due to CPU-GPU transfers. For faster generation:
1. Use GPU with 24GB+ VRAM
2. Disable block offload
3. Use distilled model (4 steps vs 50)

## api reference

### status endpoint

```
GET /api/flux2/status
```

Response:
```json
{
  "available": true,
  "loaded": false,
  "supported_models": ["klein-9b", "klein-9b-fp8", ...]
}
```

### generate endpoint

```
POST /api/flux2/generate
Content-Type: application/json

{
  "prompt": "A photo of a cat",
  "model_name": "klein-9b-fp8",
  "width": 1024,
  "height": 1024,
  "num_steps": 4,
  "guidance": 1.0,
  "seed": 42,
  "block_offload": true,
  "reference_images": ["data:image/png;base64,..."]
}
```

Response: PNG image binary with headers:
- `X-Seed`: Generation seed used
- `X-Generation-Time`: Time in seconds
- `X-Model`: Model variant used

### unload endpoint

```
POST /api/models/flux2/unload
```

Frees VRAM by unloading the FLUX.2 pipeline.

## python api

```python
from llm_dit.pipelines.flux2_generate import Flux2GenerationConfig, generate_image
from PIL import Image

# Text-to-image
config = Flux2GenerationConfig(
    prompt="A beautiful sunset over mountains",
    width=1024,
    height=1024,
    num_steps=4,
    guidance=1.0,
    seed=42,
    block_offload=True,
)

image = generate_image(
    config,
    model_name="klein-9b-fp8",
    model_path="models/FLUX.2-klein/FLUX.2-klein-9b-fp8/",
    vae_path="models/FLUX.2-klein/FLUX.2-klein-9B/",
)
image.save("output.png")

# Image editing with references
ref1 = Image.open("cat.png")
ref2 = Image.open("background.png")

config = Flux2GenerationConfig(
    prompt="Place the cat on the beach background",
    width=1024,
    height=1024,
    reference_images=[ref1, ref2],
    block_offload=True,
)

result = generate_image(config, model_name="klein-9b-fp8")
result.save("edited.png")
```

## related guides

- [config_management.md](config_management.md) - TOML configuration
- [vl_conditioning.md](vl_conditioning.md) - Vision-language conditioning (different approach)
