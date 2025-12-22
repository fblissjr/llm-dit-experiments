last updated: 2025-12-22

# z-image

Documentation for Z-Image (Tongyi-MAI Z-Image-Turbo), the primary text-to-image model in this project.

## overview

Z-Image is a flow-matching Diffusion Transformer (DiT) developed by Alibaba/Tongyi that uses Qwen3-4B as its text encoder. It is turbo-distilled for fast 8-9 step generation with classifier-free guidance baked in via Decoupled-DMD distillation.

### key capabilities

- Text-to-image generation at arbitrary resolutions (divisible by 16)
- Image-to-image transformation with strength control
- High-resolution generation (2K-4K) via DyPE position extrapolation
- LoRA support with automatic weight fusion
- Skip Layer Guidance (SLG) for improved anatomy/structure
- Multi-pass generation for ultra-high resolutions
- Tiled VAE decoding for memory-efficient large image generation

## architecture

| Component | Details |
|-----------|---------|
| Text encoder | Qwen3-4B (2560 hidden dim, 36 layers) |
| Embedding layer | -2 (penultimate, configurable via `--hidden-layer`) |
| DiT | 30-layer transformer, 6B parameters |
| Latent channels | 16 |
| VAE | SD3-style AutoencoderKL (scaling_factor + shift_factor) |
| Scheduler | FlowMatchEulerDiscreteScheduler (shift=3.0) |

### embedding pipeline

```
Text Prompt -> Qwen3Formatter -> Qwen3-4B -> hidden_states[layer] -> DiT -> VAE -> Image
```

The text encoder extracts embeddings from a configurable hidden layer of Qwen3-4B (default: layer -2, penultimate). These embeddings condition the DiT through cross-attention during the denoising process.

## key parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Max tokens | 1504 | DiT RoPE limit - see [long_prompts.md](../reference/long_prompts.md) |
| CFG scale | 0.0 | Baked in via Decoupled-DMD distillation |
| Steps | 8-9 | Turbo distilled (non-distilled uses 28+) |
| Scheduler shift | 3.0 | Flow matching shift parameter |
| Resolution | 1024x1024 | Default, supports any multiple of 16 |

### resolution constraints

- Dimensions must be divisible by 16 (VAE scale factor * 2)
- Typical resolutions: 512, 768, 1024, 1536, 2048
- For 2K+ resolutions, enable DyPE and tiled VAE

## configuration

### config.toml

```toml
[default]
model_path = "/path/to/z-image"
templates_dir = "templates/z_image"

[default.encoder]
device = "cuda"          # or "cpu", "mps", "auto"
torch_dtype = "bfloat16"
hidden_layer = -2        # penultimate layer

[default.pipeline]
device = "cuda"
torch_dtype = "bfloat16"

[default.generation]
height = 1024
width = 1024
num_inference_steps = 9
guidance_scale = 0.0     # baked in, leave at 0.0

[default.scheduler]
shift = 3.0

[default.pytorch]
attention_backend = "auto"  # flash_attn_2, flash_attn_3, sage, xformers, sdpa
use_custom_scheduler = false
tiled_vae = false
tile_size = 512
tile_overlap = 64
long_prompt_mode = "interpolate"  # truncate, interpolate, pool, attention_pool
```

### cli overrides

CLI flags override config file values:

```bash
uv run scripts/generate.py \
  --config config.toml \
  --hidden-layer -3 \
  --steps 9 \
  --width 1024 --height 1024 \
  "Your prompt"
```

## usage

### cli

```bash
# Basic generation
uv run scripts/generate.py \
  --model-path /path/to/z-image \
  "A cat sleeping in sunlight"

# With seed for reproducibility
uv run scripts/generate.py \
  --model-path /path/to/z-image \
  --seed 42 \
  "A sunset over mountains"

# Custom resolution
uv run scripts/generate.py \
  --model-path /path/to/z-image \
  --width 1536 --height 1024 \
  "A panoramic landscape"

# Image-to-image
uv run scripts/generate.py \
  --model-path /path/to/z-image \
  --img2img input.jpg \
  --strength 0.6 \
  "A watercolor painting"
```

### web api

```bash
# Generate image
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat sleeping",
    "width": 1024,
    "height": 1024,
    "steps": 9,
    "seed": 42
  }'

# With template
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A sunset",
    "template": "photography"
  }'
```

### python api

```python
from llm_dit.pipelines.z_image import ZImagePipeline

# Load pipeline
pipe = ZImagePipeline.from_pretrained(
    "/path/to/z-image",
    templates_dir="templates/z_image",
    torch_dtype=torch.bfloat16,
    encoder_device="cuda",
    dit_device="cuda",
    vae_device="cuda",
)

# Basic generation
image = pipe(
    "A cat sleeping in sunlight",
    height=1024,
    width=1024,
    num_inference_steps=9,
)
image.save("output.png")

# With seed
import torch
image = pipe(
    "A sunset",
    generator=torch.Generator().manual_seed(42),
)

# Image-to-image
from PIL import Image
input_img = Image.open("photo.jpg")
image = pipe.img2img(
    "A watercolor painting",
    image=input_img,
    strength=0.6,
)
```

## advanced features

### dype (dynamic position extrapolation)

Enable high-resolution generation (2K-4K) by dynamically scaling RoPE position encodings:

```bash
# 2K generation
uv run scripts/generate.py \
  --model-path /path/to/z-image \
  --dype \
  --dype-scale 2.0 \
  --width 2048 --height 2048 \
  "A detailed landscape"

# 4K generation with tiled VAE
uv run scripts/generate.py \
  --model-path /path/to/z-image \
  --dype \
  --dype-scale 4.0 \
  --tiled-vae \
  --width 4096 --height 4096 \
  "An ultra-detailed cityscape"
```

Config:
```toml
[default.dype]
enabled = true
method = "vision_yarn"  # recommended for Z-Image
dype_scale = 2.0
dype_exponent = 2.0
```

See [dype.md](../reference/dype.md) for detailed documentation.

### skip layer guidance (slg)

Improve anatomy and structure coherence by skipping layers during part of the denoising process:

```bash
uv run scripts/generate.py \
  --model-path /path/to/z-image \
  --slg \
  --slg-scale 2.5 \
  --slg-layers 7,8,9,10,11,12 \
  --slg-start 0.05 \
  --slg-stop 0.5 \
  "A portrait of a person"
```

Config:
```toml
[default.slg]
enabled = true
scale = 2.5
layers = [7, 8, 9, 10, 11, 12]  # middle layers (Z-Image has 30)
start = 0.05                    # start at 5% of steps
stop = 0.5                      # stop at 50% of steps
```

**Notes:**
- SLG approximately doubles inference time (two forward passes per step where active)
- Z-Image DiT has 30 layers; middle layers (7-12) work best for structure
- Scale 2.5 is lower than SD3.5's default since turbo models have more steps affected

### lora

Load and fuse LoRA weights into the transformer:

```bash
# Single LoRA
uv run scripts/generate.py \
  --model-path /path/to/z-image \
  --lora style.safetensors:0.8 \
  "A prompt"

# Multiple LoRAs
uv run scripts/generate.py \
  --model-path /path/to/z-image \
  --lora style.safetensors:0.5 \
  --lora detail.safetensors:0.3 \
  "A prompt"
```

Config:
```toml
[default.lora]
paths = ["style.safetensors", "detail.safetensors"]
scales = [0.5, 0.3]
```

**Important:** LoRAs are fused (permanently merged) into weights. To remove a LoRA, reload the pipeline.

See [lora.md](../guides/lora.md) for more details.

### long prompt handling

Prompts exceeding 1504 tokens are automatically compressed:

| Mode | Flag | Description |
|------|------|-------------|
| truncate | `--long-prompt-mode truncate` | Cut off at 1504 tokens |
| interpolate | `--long-prompt-mode interpolate` | Linear resampling (default) |
| pool | `--long-prompt-mode pool` | Adaptive average pooling |
| attention_pool | `--long-prompt-mode attention_pool` | Cosine similarity weighting |

See [long_prompts.md](../reference/long_prompts.md) for details.

### multi-pass generation

Generate ultra-high resolution images through iterative refinement:

```python
image = pipe.generate_multipass(
    "A detailed portrait",
    final_width=4096,
    final_height=4096,
    passes=[
        {"scale": 0.5, "steps": 9},      # 2K first pass
        {"scale": 1.0, "steps": 9, "strength": 0.5},  # 4K refinement
    ],
)
```

### tiled vae decoding

Decode large images in tiles to save VRAM:

```bash
uv run scripts/generate.py \
  --model-path /path/to/z-image \
  --tiled-vae \
  --tile-size 512 \
  --tile-overlap 64 \
  --width 2048 --height 2048 \
  "A large image"
```

### distributed inference

Encode prompts on one machine (e.g., Mac with large RAM) and generate on another (e.g., CUDA server):

```python
# On Mac: encode and save
embeddings = encoder.encode("A prompt")
torch.save(embeddings, "embeddings.pt")

# On CUDA server: load and generate
pipe = ZImagePipeline.from_pretrained_generator_only("/path/to/z-image")
embeddings = torch.load("embeddings.pt")
image = pipe.generate_from_embeddings(embeddings)
```

See [distributed.md](../guides/distributed.md) for details.

## memory requirements

Approximate VRAM usage (1024x1024):

| Component | VRAM |
|-----------|------|
| Text encoder (Qwen3-4B) | 8 GB |
| DiT (30 layers) | 12 GB |
| VAE | 2 GB |
| **Total** | **~22 GB** |

### memory optimization strategies

1. **CPU offload encoder:** `--encoder-device cpu` (slower encoding, saves 8 GB)
2. **Sequential CPU offload:** `--cpu-offload` (moves components on/off GPU)
3. **Quantization:** `--quantization 8bit` or `--quantization 4bit` (encoder only)
4. **Tiled VAE:** `--tiled-vae` (for high-res, reduces VAE peak VRAM)

## limitations

1. **Max 1504 tokens:** DiT RoPE configuration limits text sequence length
2. **CFG baked in:** Cannot adjust guidance_scale (always 0.0)
3. **High VRAM:** Requires 22+ GB for full quality at 1024x1024
4. **No native inpainting:** Use img2img with masks externally
5. **Quality at 4K+:** May degrade despite DyPE; use multi-pass for best results

## related files

| File | Description |
|------|-------------|
| `src/llm_dit/pipelines/z_image.py` | Main pipeline implementation |
| `src/llm_dit/encoders/z_image_encoder.py` | Text encoder wrapper |
| `src/llm_dit/schedulers/flow_match.py` | Pure PyTorch scheduler |
| `src/llm_dit/utils/tiled_vae.py` | Tiled VAE decoder |
| `src/llm_dit/utils/dype.py` | DyPE implementation |
| `src/llm_dit/guidance/skip_layer.py` | SLG implementation |
| `src/llm_dit/utils/lora.py` | LoRA loading and fusion |
| `templates/z_image/` | 140+ prompt templates |

## see also

- [CLI Reference](../reference/cli_flags.md)
- [API Endpoints](../reference/api_endpoints.md)
- [Configuration Guide](../reference/configuration.md)
- [Resolution Guide](../reference/resolution.md)
- [DyPE Reference](../reference/dype.md)
- [Long Prompts](../reference/long_prompts.md)
