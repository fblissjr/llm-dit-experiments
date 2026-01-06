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
