# dype (dynamic position extrapolation)

*last updated: 2025-12-24*

DyPE enables generation at resolutions beyond the model's training resolution (1024x1024) by dynamically scaling the RoPE position encodings. Essential for high-resolution generation (2K, 4K) without retraining.

## current status

**DyPE frequency modulation is not yet implemented for Z-Image.** The diffusers `ZImageTransformer2DModel` uses complex64 RoPE embeddings (via `torch.polar`) which requires matching the exact output format. The current implementation delegates to the original embedder.

**Recommended approach for high-resolution generation:** Use **multipass** mode which works well without frequency modulation:
- Two-pass: 512px first pass, then img2img upscale to target
- Three-pass: 256px -> 512px -> target

The multipass approach produces excellent results and avoids the need for DyPE frequency extrapolation.

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
- `single`: Direct generation at target resolution (frequency modulation not yet working)
- `twopass`: Half-res first pass, then img2img refinement
- `threepass`: Quarter-res -> half-res -> full-res

**pass2_strength:** Controls how much the refinement passes change the image (0.3-0.7 recommended).

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
