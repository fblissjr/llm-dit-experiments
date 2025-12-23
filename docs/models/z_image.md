last updated: 2025-12-23

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

### scheduler shift behavior

The shift parameter compresses the noise schedule via: `sigma' = shift * sigma / (1 + (shift - 1) * sigma)`

**Why shift changes appear imperceptible in Z-Image-Turbo:**

1. **Turbo distillation**: The model was trained with Decoupled-DMD using shift=3.0 specifically. The distillation process made the model robust to schedule variations within a reasonable range.

2. **Coarse step discretization**: With only 8-9 steps, each step covers ~11% of the trajectory. Small differences in where steps land on the noise curve are overshadowed by the large step size.

3. **Flow matching robustness**: Flow matching predicts velocity directly (not noise), making the velocity field smoother and more tolerant of schedule changes than DDPM-style models.

**Expected behavior at different shift values:**

| Shift Range | Expected Quality |
|-------------|------------------|
| < 1.5 | Incomplete denoising, noise artifacts |
| 2.5 - 4.0 | Sweet spot, minimal visible difference (by design) |
| > 6.0 | Over-compressed schedule, potential blocky artifacts |

**Bottom line:** Shift imperceptibility within the 2.5-4.0 range is a feature of turbo distillation, not a bug. The default shift=3.0 is optimal for 8-9 step generation.

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

## performance optimization

### attention backends

The attention backend has the largest impact on inference speed. Z-Image benefits significantly from optimized attention implementations, especially for the 30-layer DiT.

**Priority order (best to worst):**

1. **flash_attn_2** - Best for Ampere+ GPUs (RTX 3090, 4090, A100, etc.)
   - ~2x speedup over SDPA baseline
   - Requires manual installation: `pip install flash-attn --no-build-isolation`
   - Auto-detected when installed and `attention_backend = "auto"`
   - **Recommended for RTX 4090**

2. **flash_attn_3** - Hopper architecture only (H100, H200)
   - Further optimized for H100 tensor cores
   - Requires Hopper GPU and latest flash-attn package
   - Auto-detected on compatible hardware

3. **sage** - SageAttention INT8 kernel
   - 15% faster than SDPA (4.41s vs 5.08s per image, RTX 4090)
   - SSIM 0.98 vs SDPA reference (very close to identical)
   - Easier to install than Flash Attention
   - Good fallback when FA2 not available
   - Incompatible with torch.compile (use without compilation)

4. **xformers** - Flexible attention implementation
   - Cross-platform, good performance
   - Well-tested with diffusion models
   - Install: `pip install xformers`

5. **sdpa** - PyTorch built-in fallback
   - Always available (no installation needed)
   - Reasonable performance, used as baseline
   - Auto-selected when no optimized backend is available

**Configuration:**

```toml
[default.pytorch]
attention_backend = "auto"  # recommended - picks best available
```

**Check available backends:**

```bash
uv run scripts/profiler.py --show-info
```

**Run attention benchmark:**

```bash
uv run experiments/attention_backend_benchmark.py --config config.toml --profile rtx4090
```

This compares speed AND quality (SSIM, PSNR) for each backend against SDPA reference.

### rtx 4090 optimal settings

The RTX 4090 (Ada Lovelace, 24GB VRAM) is ideal for Z-Image with these settings:

```toml
[rtx4090]
# Attention - Flash Attention 2 for best speed
[rtx4090.pytorch]
attention_backend = "auto"      # selects flash_attn_2 if installed
compile = true                  # ~10% speedup after warmup
embedding_cache = true          # faster repeated prompts in web server

# Precision - RTX 4090 has native BF16 support
[rtx4090.encoder]
device = "cuda"
torch_dtype = "bfloat16"

[rtx4090.pipeline]
device = "cuda"
torch_dtype = "bfloat16"

# No offloading needed with 24GB VRAM
# All components (encoder, DiT, VAE) fit on GPU simultaneously
```

**Installation for maximum performance:**

```bash
# Install Flash Attention 2 (requires CUDA toolkit)
pip install flash-attn --no-build-isolation

# Verify it's detected
uv run scripts/profiler.py --show-info
# Should show: "flash_attn_2: available"
```

**Performance expectations (1024x1024, 9 steps):**

| Configuration | Encode Time | Generate Time | Total | Speedup |
|---------------|-------------|---------------|-------|---------|
| SDPA (baseline) | ~2.0s | ~5.1s | ~7.1s | 1.0x |
| SageAttention | ~2.0s | ~4.4s | ~6.4s | 1.15x |
| Flash Attention 2 | ~1.0s | ~3.0s | ~4.0s | ~1.8x |
| FA2 + compile | ~1.0s | ~2.7s | ~3.7s | ~1.9x |

*Benchmark results from RTX 4090, no torch.compile*

### device placement by vram budget

| VRAM | Encoder | DiT | VAE | Notes |
|------|---------|-----|-----|-------|
| **24GB+ (RTX 4090, A100)** | CUDA | CUDA | CUDA | Optimal, no offloading |
| **16GB (RTX 4080)** | CPU | CUDA | CUDA | Encode once, slower first inference |
| **12GB (RTX 3060)** | CPU | CUDA | CUDA | Add `--quantization 8bit` for encoder |
| **8GB** | CPU | CUDA | CPU | Requires sequential offload (`--cpu-offload`) |
| **<8GB** | Not recommended | | | Use distributed inference or cloud GPU |

**Example configurations:**

```toml
# 24GB - all on GPU (fastest)
[rtx4090.encoder]
device = "cuda"

[rtx4090.pipeline]
device = "cuda"

# 16GB - encoder on CPU
[rtx4080.encoder]
device = "cpu"

[rtx4080.pipeline]
device = "cuda"

# 12GB - encoder on CPU with quantization
[rtx3060.encoder]
device = "cpu"
quantization = "8bit"

[rtx3060.pipeline]
device = "cuda"

# 8GB - sequential CPU offload
[rtx3060ti]
# Use CLI flag: --cpu-offload
```

### torch.compile

PyTorch 2.0+ compilation can provide ~10% speedup but has tradeoffs:

**Benefits:**
- ~10% faster generation after warmup
- Good for web servers or batch processing
- Works best with static shapes (same resolution repeatedly)

**Drawbacks:**
- First inference is slow (compilation overhead)
- May increase memory usage slightly
- Can cause issues with dynamic shapes

**Configuration:**

```toml
[default.pytorch]
compile = true                  # enable torch.compile
compile_mode = "reduce-overhead"  # default mode
```

**When to use:**
- Running a web server with repeated generations
- Batch processing many images at same resolution
- After warmup cost is acceptable

**When to skip:**
- One-off generations or quick tests
- Constantly changing resolutions
- Limited VRAM (compilation uses extra memory)

### embedding cache

For web servers handling repeated prompts, enable embedding cache to avoid re-encoding:

```toml
[default.pytorch]
embedding_cache = true
```

This caches text embeddings by prompt hash. Second generation with same prompt skips encoding entirely.

**Benefits:**
- Near-instant encoding on cache hits
- Useful for template-based workflows
- Minimal memory overhead (caches embeddings only, not images)

**Not useful for:**
- CLI one-off generations
- Unique prompts every time
- Very limited RAM (cache is in-memory)

### benchmarking your setup

Use the profiler to find optimal settings for your hardware:

```bash
# Show available backends and current config
uv run scripts/profiler.py --show-info

# Test encoding and generation speed
uv run scripts/profiler.py \
  --model-path /path/to/z-image \
  --tests encode,generate

# Compare multiple configurations
uv run scripts/profiler.py \
  --model-path /path/to/z-image \
  --sweep

# Test specific backend
uv run scripts/profiler.py \
  --model-path /path/to/z-image \
  --attention-backend flash_attn_2 \
  --tests generate
```

The profiler will show:
- Available attention backends
- Encode time (text -> embeddings)
- Generate time (embeddings -> image)
- Peak VRAM usage
- Comparison across configurations

See [profiler.md](../guides/profiler.md) for detailed usage.

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
