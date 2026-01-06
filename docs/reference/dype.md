# dype (dynamic position extrapolation)

*last updated: 2026-01-06*

DyPE enables generation at resolutions beyond the model's training resolution (1024x1024) by dynamically scaling the RoPE position encodings. Essential for high-resolution generation (2K, 4K) without retraining.

## current status

DyPE now supports two high-resolution generation approaches:

1. **Multipass mode** (recommended, stable): Generate at lower resolution first, then refine via img2img
2. **Frequency modulation** (experimental): Dynamically scale RoPE frequencies based on diffusion timestep

**Recommended approach:** Use **multipass** mode for best results:
- Two-pass: 512px first pass, then img2img upscale to target
- Three-pass: 256px -> 512px -> target

Frequency modulation is available as an experimental option via `--dype-frequency-modulation`.

## why dype is needed

Z-Image uses multi-axis RoPE (Rotary Position Embedding) with three axes:
- Axis 0 (text/time): 1504 positions
- Axis 1 (height): 512 positions (maps to 4096 pixels at 8x VAE scaling)
- Axis 2 (width): 512 positions (maps to 4096 pixels at 8x VAE scaling)

When generating at 2048x2048 or higher, the image axes exceed their trained position range. Without DyPE, the model produces degraded quality or artifacts.

## available methods

| Method | Class | Use Case | Recommendation |
|--------|-------|----------|----------------|
| `vision_yarn` | `VisionYaRNDyPE` | Multi-axis RoPE extrapolation | **Recommended** for Z-Image |
| `yarn` | `YaRNDyPE` | Text-only RoPE extrapolation | Fallback for debugging |
| `ntk` | `NTKDyPE` | Alternative frequency scaling | Experimental comparison |

**Vision-YaRN** is recommended as it properly handles the multi-axis RoPE structure in Z-Image's DiT.

## configuration

**Via TOML config:**
```toml
[default.dype]
enabled = true
method = "vision_yarn"
scale = 2.0             # 2.0 for 2K, 4.0 for 4K
alpha = 1.0
beta = 32.0
```

**Via CLI:**
```bash
uv run scripts/generate.py \
  --dype \
  --dype-method vision_yarn \
  --dype-scale 2.0 \
  --width 2048 --height 2048 \
  "Your prompt"
```

## usage examples

**2K Generation (2048x2048):**
```bash
uv run scripts/generate.py \
  --model-path /path/to/z-image \
  --dype \
  --dype-scale 2.0 \
  --width 2048 --height 2048 \
  "A detailed mountain landscape"
```

**4K Generation (4096x4096):**
```bash
uv run scripts/generate.py \
  --model-path /path/to/z-image \
  --dype \
  --dype-scale 4.0 \
  --width 4096 --height 4096 \
  --tiled-vae \
  --tile-size 512 \
  "An ultra-detailed cityscape"
```

## implementation

DyPE is implemented in `src/llm_dit/utils/dype.py` (765+ lines) with three classes:

1. **VisionYaRNDyPE**: Multi-axis RoPE scaling for vision transformers
2. **YaRNDyPE**: Text-only RoPE extrapolation
3. **NTKDyPE**: Neural Tangent Kernel scaling

## quality considerations

**DyPE can only extrapolate what the model learned:**
- 2K (2048x2048): Usually works well with scale=2.0
- 4K (4096x4096): May show quality degradation, use with tiled VAE
- 8K+: Consider multi-pass rendering in addition to DyPE

## recommended settings

### rtx 4090 optimized config

```toml
[rtx4090.dype]
enabled = true
method = "vision_yarn"           # Best method for Z-Image
dype_scale = 2.0                 # Proven community value
dype_exponent = 2.0              # Quadratic decay
dype_start_sigma = 1.0           # From beginning
base_shift = 0.5                 # Base resolution shift
max_shift = 1.15                 # High-res shift
base_resolution = 1024           # Z-Image training res
anisotropic = false              # Enable for 21:9, 32:9
multipass = "twopass"            # Key for 2K quality
pass2_strength = 0.5             # 0.3-0.7 recommended
pass3_strength = 0.4             # For threepass only
frequency_modulation = false     # Keep off unless experimenting

[rtx4090.pytorch]
tiled_vae = true                 # Required for 2K+
tile_size = 512
tile_overlap = 64
```

### resolution presets

| Resolution | Multipass | Pass Strengths | tiled_vae | Notes |
|------------|-----------|----------------|-----------|-------|
| 1024x1024 | `single` | n/a | false | Base resolution, no DyPE needed |
| 1536x1536 | `single` | n/a | false | Works without multipass |
| 2048x2048 | `twopass` | 0.5 | true | Recommended 2K workflow |
| 2560x2560 | `twopass` | 0.4-0.5 | true | Lower strength preserves detail |
| 4096x4096 | `threepass` | 0.5/0.4 | true | Three passes for best quality |

### pass strength tuning

| Strength | Effect | Use Case |
|----------|--------|----------|
| 0.3-0.4 | Minimal change | Preserve structure from pass 1, subtle refinement |
| 0.5 | Balanced | Good default, moderate creative freedom |
| 0.6-0.7 | More change | Allow divergence, more detail regeneration |

**Tips:**
- Lower `pass2_strength` = more consistent with first pass
- Higher `pass2_strength` = more freedom to add detail (but may lose coherence)
- For portraits/faces: use 0.3-0.4 to preserve likeness
- For landscapes/abstract: 0.5-0.6 works well

### aspect ratio recommendations

| Aspect | Resolution | anisotropic | Notes |
|--------|------------|-------------|-------|
| 1:1 | 2048x2048 | false | Standard square |
| 16:9 | 1920x1088 | false | HD widescreen |
| 21:9 | 2560x1088 | true | Ultrawide panorama |
| 32:9 | 3840x1088 | true | Super ultrawide |
| 9:16 | 1088x1920 | false | Portrait/mobile |

Enable `anisotropic = true` for extreme aspect ratios (wider than 2:1 or taller than 1:2).

### quick reference cli

```bash
# 2K generation (recommended)
uv run scripts/generate.py --config config.toml --profile rtx4090 \
  --width 2048 --height 2048 "prompt"

# 2K with custom pass strength
uv run scripts/generate.py --dype --dype-multipass twopass \
  --dype-pass2-strength 0.4 --width 2048 --height 2048 "prompt"

# 4K generation
uv run scripts/generate.py --dype --dype-multipass threepass \
  --dype-pass2-strength 0.5 --dype-pass3-strength 0.4 \
  --tiled-vae --width 4096 --height 4096 "prompt"

# Ultrawide panorama
uv run scripts/generate.py --dype --dype-multipass twopass \
  --width 3840 --height 1088 "prompt"
```

## complementary techniques

DyPE works well with:
- **Tiled VAE** (`--tiled-vae`): Decode large latents in tiles to save VRAM
- **Multi-pass rendering**: Generate ultra-high-res in overlapping passes
- **CPU offload** (`--cpu-offload`): Save VRAM for large DiT models

## multipass generation (recommended)

For high-resolution generation, use multipass mode instead of single-pass DyPE:

```bash
# Two-pass 1080p generation
uv run scripts/generate.py \
  --model-path /path/to/z-image \
  --dype \
  --dype-multipass twopass \
  --dype-pass2-strength 0.5 \
  --width 1920 --height 1088 \
  "A detailed landscape"

# Three-pass 4K generation
uv run scripts/generate.py \
  --model-path /path/to/z-image \
  --dype \
  --dype-multipass threepass \
  --dype-pass2-strength 0.5 \
  --width 4096 --height 4096 \
  --tiled-vae \
  "An ultra-detailed cityscape"
```

**Multipass modes:**
- `single`: Direct generation at target resolution (use with `--dype-frequency-modulation` for best results)
- `twopass`: Half-res first pass, then img2img refinement (recommended)
- `threepass`: Quarter-res -> half-res -> full-res (for 4K+)

**Pass strength:**
- `pass2_strength` (default 0.5): Controls how much the second pass changes the image. Lower values preserve more detail from the first pass. Range: 0.3-0.8.
- `pass3_strength` (default 0.4): Controls the third pass refinement in threepass mode. Typically lower than pass2 for fine-tuning. Range: 0.2-0.7.

## frequency modulation (experimental)

Frequency modulation dynamically scales RoPE frequencies based on the diffusion timestep:
- Early steps (high sigma): Lower frequencies capture global structure
- Late steps (low sigma): Higher frequencies capture fine details

```bash
# Single-pass with frequency modulation
uv run scripts/generate.py \
  --model-path /path/to/z-image \
  --dype \
  --dype-frequency-modulation \
  --width 2048 --height 2048 \
  "A detailed landscape"
```

**Note:** Frequency modulation is experimental. If results are unsatisfactory, use multipass mode instead.

## dynamic shift (scheduler)

**Important:** Scheduler "dynamic shift" is separate from DyPE's `base_shift`/`max_shift` settings.

### what's the difference?

| Setting | Where | What It Affects |
|---------|-------|-----------------|
| DyPE `base_shift`/`max_shift` | DyPE config | Noise schedule weighting for high-res |
| Scheduler `dynamic_shift` | Scheduler config | FlowMatch sigma spacing |

**DyPE shift** (`base_shift`, `max_shift`):
- Part of DyPE configuration
- Affects the internal noise schedule weighting during RoPE extrapolation
- Linearly interpolates between `base_shift` (0.5) at 1024px and `max_shift` (1.15) at 2048px

**Scheduler dynamic shift** (`dynamic_shift`):
- Independent of DyPE
- Affects the FlowMatch scheduler's sigma schedule
- When enabled, calculates shift based on sequence length (resolution)
- Formula: `shift = seq_len * m + b` (clamped to [0.5, 1.15])

### using both together

You can combine DyPE with scheduler dynamic shift for optimal high-resolution generation:

```bash
# DyPE multipass with dynamic shift
uv run scripts/generate.py --config config.toml --profile rtx4090 \
  --dype --dype-multipass twopass \
  --dynamic-shift \
  --width 2048 --height 2048 "prompt"
```

```toml
[rtx4090.scheduler]
dynamic_shift = true   # Let scheduler calculate shift based on resolution

[rtx4090.dype]
enabled = true
base_shift = 0.5       # DyPE internal shift at base res
max_shift = 1.15       # DyPE internal shift at max res
```

### when to use dynamic shift

| Scenario | Use Dynamic Shift? | Notes |
|----------|-------------------|-------|
| Fixed resolution (1024x1024) | No | Use fixed `shift=3.0` |
| Variable resolution | Yes | Automatically adapts to resolution |
| High-res DyPE generation | Optional | May improve results at 2K+ |
| Experimentation | Try both | Compare fixed vs dynamic shift quality |

## web ui

The web UI exposes all DyPE parameters in a collapsible "DyPE (High-Resolution)" section:

| Control | Parameter | Range | Default | Notes |
|---------|-----------|-------|---------|-------|
| Enable DyPE | `enabled` | checkbox | false | Master toggle |
| Method | `method` | select | vision_yarn | vision_yarn, yarn, ntk |
| Multipass | `multipass` | select | twopass | single, twopass, threepass |
| DyPE Scale | `dype_scale` | 0.5-4.0 | 2.0 | Magnitude of effect |
| DyPE Exponent | `dype_exponent` | 1.0-4.0 | 2.0 | Decay speed |
| Base Shift | `base_shift` | 0.1-1.0 | 0.5 | Noise schedule at 1024px |
| Max Shift | `max_shift` | 0.5-2.0 | 1.15 | Noise schedule at max res |
| Pass 2 Strength | `pass2_strength` | 0.3-0.8 | 0.5 | Refinement pass intensity |
| Pass 3 Strength | `pass3_strength` | 0.2-0.7 | 0.4 | Third pass (threepass only) |
| Frequency Modulation | `frequency_modulation` | checkbox | false | Experimental timestep-based RoPE |

**Visibility notes:**
- Pass 3 Strength only visible when "threepass" mode selected
- Frequency Modulation marked as experimental

## python api

```python
from llm_dit.pipelines import ZImagePipeline

# Multipass generation (recommended)
image = pipeline.generate_multipass(
    prompt="Your prompt",
    final_width=2048,
    final_height=2048,
    passes=[
        {"scale": 0.5, "steps": 9},  # 1024x1024 first pass
        {"scale": 1.0, "steps": 9, "strength": 0.5},  # 2048x2048 refinement
    ],
)
```
