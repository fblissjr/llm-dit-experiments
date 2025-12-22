# dype (dynamic position extrapolation)

*last updated: 2025-12-22*

DyPE enables generation at resolutions beyond the model's training resolution (1024x1024) by dynamically scaling the RoPE position encodings. Essential for high-resolution generation (2K, 4K) without retraining.

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

## python api

```python
from llm_dit.utils.dype import VisionYaRNDyPE

dype = VisionYaRNDyPE(
    model=text_encoder.model,
    scale=2.0,
    alpha=1.0,
    beta=32.0
)
dype.apply()

result = pipeline(
    prompt="Your prompt",
    width=2048,
    height=2048,
    num_inference_steps=9
)
```
