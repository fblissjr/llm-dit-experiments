last updated: 2025-12-19

# Z-Image Omni Research Findings

## Executive Summary

Z-Image Omni (diffusers PR 12857) adds image conditioning to Z-Image Turbo using SigLIP2 vision encoder. After extensive analysis, **the siglip_embedder and siglip_refiner components require trained weights that have NOT been released**. The architecture is understood and implemented, but omni mode produces noise without these weights.

## Table of Contents

1. [Background](#background)
2. [Architecture Analysis](#architecture-analysis)
3. [Implementation Work](#implementation-work)
4. [Initialization Experiments](#initialization-experiments)
5. [Key Findings](#key-findings)
6. [Open Questions](#open-questions)
7. [Files and Resources](#files-and-resources)

---

## Background

### What is Z-Image Omni?

Z-Image Omni extends Z-Image Turbo with **image conditioning** - the ability to use reference images to guide generation. This is implemented via a SigLIP2 vision encoder that extracts image embeddings which are then processed and concatenated with text embeddings in the transformer.

### Source Materials

- **Diffusers PR 12857**: https://github.com/huggingface/diffusers/pull/12857
- **RuoyiDu's Commit** (Z-Image dev): https://github.com/RuoyiDu/diffusers/commit/3435ba2a9f686e8adff8dbaaf5ed2746c2bd0327
- **Local PR Copy**: `./coderef/diffusers/src/diffusers/`

### Key Assertion

The user confirmed: **"its the same zimage turbo model"** - Z-Image Omni uses the identical Z-Image Turbo weights for the core DiT, with added siglip components.

---

## Architecture Analysis

See detailed breakdown: [diffusers_pr_12857_breakdown.md](./diffusers_pr_12857_breakdown.md)

### Dual Encoder System

```
Text Input                    Reference Image
    |                              |
Qwen3-4B                     SigLIP2 Vision Model
(~/Storage/Qwen3-4B)         (google/siglip2-so400m-patch14-384)
    |                              |
2560-dim embeddings          1152-dim embeddings
    |                              |
cap_embedder (trained)       siglip_embedder (NEW - needs training)
    |                              |
    +-------> 3840-dim <-----------+
                |
        Z-Image Turbo DiT
                |
          Generated Image
```

### New Transformer Components

From `transformer_z_image.py` in PR 12857:

```python
# Config parameter
siglip_feat_dim = 1152  # Default fallback (line 855)

# New components (require trained weights)
siglip_embedder = nn.Sequential(
    RMSNorm(siglip_feat_dim, eps=1e-5),
    nn.Linear(siglip_feat_dim, 3840, bias=True)
)
siglip_refiner = nn.ModuleList([
    ZImageTransformerBlock(2000 + layer_id, ...)
    for layer_id in range(2)
])
siglip_pad_token = nn.Parameter(torch.empty((1, 3840)))
```

### Processing Flow

1. **Dual Timestep Embeddings**: `t_noisy` for target, `t_clean=1.0` for conditioning
2. **Noise Mask**: Per-token mask distinguishing noisy (target) vs clean (condition) tokens
3. **Unified Sequence**: `[cap_feats] + [x_latents] + [siglip_feats]`
4. **Main Layers**: 30 transformer layers with noise-mask-based modulation

### SigLIP2 Model Specification

| Property | Value |
|----------|-------|
| Model | `google/siglip2-so400m-patch14-384` |
| Hidden Dimension | **1152** |
| Input Size | 384x384 |
| Patch Size | 14x14 |
| Patches | 27x27 = 729 tokens |

The 1152 dimension comes from PR line 855:
```python
sig_pad_dim = self.config.siglip_feat_dim or 1152
```

---

## Implementation Work

### SigLIP Encoder

Created `siglip_encoder.py` - a wrapper class for SigLIP2 vision model:

```python
class SigLIPEncoder:
    def __init__(self, model_path=None, device="cpu", dtype=torch.float32):
        if model_path is None:
            model_path = str(Path.home() / "Storage/google_siglip2-so400m-patch14-384")

        full_model = AutoModel.from_pretrained(model_path)
        self.model = full_model.vision_model.to(device)
        self.hidden_size = self.model.config.hidden_size  # 1152

    def encode(self, images):
        # Returns (H_patches, W_patches, hidden_size) spatial grid
        ...
```

**Key Fix**: The model type is "siglip" (not "siglip2"), requiring `AutoModel` then `.vision_model` extraction.

### Test Files

| File | Purpose | Status |
|------|---------|--------|
| `test_siglip_encoding.py` | SigLIP embedding extraction | Working |
| `test_omni_pipeline.py` | Minimal omni forward pass | Working |
| `test_full_omni_generation.py` | End-to-end generation | Runs, but omni produces noise |

### Tensor Format Fixes

The PR uses unconventional tensor formats:
- Latents: `(C, F, H, W)` not `(B, C, H, W)`
- F=1 for single-frame images

Fixed VAE decode:
```python
# latents is (C, F, H, W) = (16, 1, 32, 32)
latents_for_decode = latents.squeeze(1).unsqueeze(0)  # -> (1, 16, 32, 32)
```

---

## Initialization Experiments

Since trained siglip weights are not available, we tried multiple initialization strategies:

### 1. Random Initialization

```python
for name, param in transformer.named_parameters():
    if "siglip" in name:
        if param.dim() >= 2:
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.zeros_(param)
```

**Result**: Pure noise output

### 2. Statistics-Matched Initialization

Analyzed cap_embedder output statistics and initialized siglip_embedder to produce similar distributions:

```python
# cap_embedder output stats: mean≈0, std≈2.7
# Initialize to match
scale = 2.7 / (1152 ** 0.5)  # Account for input dim
nn.init.normal_(linear.weight, mean=0, std=scale)
```

**Result**: Still noise output

### 3. Cap_embedder Subset Initialization

Used first 1152 dimensions of cap_embedder's weight matrix:

```python
# siglip_embedder Linear: (1152 -> 3840)
# cap_embedder Linear: (2560 -> 3840)
siglip_linear.weight.data = cap_linear.weight[:, :1152].clone()
siglip_linear.bias.data = cap_linear.bias.clone()
```

**Result**: Still noise output

### Why Initialization Fails

The siglip components are **trained end-to-end** with the model. They learn:
1. How to project 1152-dim SigLIP features to 3840-dim transformer space
2. How to refine these features in the siglip_refiner (2 layers)
3. The proper siglip_pad_token for variable-length sequences

Without training, the projections produce embeddings that don't align with the text embedding space, causing the DiT to produce noise.

---

## Key Findings

### 1. Architecture is Understood

The Z-Image Omni architecture is fully documented in [diffusers_pr_12857_breakdown.md](./diffusers_pr_12857_breakdown.md). Key insights:
- Same Z-Image Turbo DiT (521 parameters, NO siglip weights)
- Dual encoder: Qwen3-4B (text) + SigLIP2-so400m (vision)
- Dual timestep embeddings for clean/noisy token distinction
- Per-token noise mask for conditioning

### 2. Trained Weights Required

Z-Image Turbo checkpoint analysis (521 keys):
```
cap_embedder.0.weight, cap_embedder.1.weight, cap_embedder.1.bias
context_refiner (2 layers)
noise_refiner (2 layers)
cap_pad_token, x_pad_token

NO siglip_embedder
NO siglip_refiner
NO siglip_pad_token
```

The siglip components require trained weights that **have not been released**.

### 3. Basic Mode Works

When omni mode is disabled (no reference images), the pipeline produces correct output using existing Z-Image Turbo weights. This confirms our implementation is correct.

### 4. SigLIP2 Encoder Works

Our SigLIP encoder correctly extracts 1152-dim embeddings:
```
Embedding shape: torch.Size([27, 27, 1152])
Hidden size: 1152
Mean: -0.0234
Std: 0.4892
```

### 5. Default Dimension is 1152

From PR line 855:
```python
sig_pad_dim = self.config.siglip_feat_dim or 1152
```

This indicates they use SigLIP2-so400m-patch14-384 (1152 dim), not the base model (768 dim).

---

## Open Questions

### 1. Different SigLIP2 Variant?

Could there be a fine-tuned SigLIP2 model we're missing? The PR uses stock `google/siglip2-so400m-patch14-384` but perhaps:
- Custom fine-tuned weights exist
- A different model checkpoint is used

**Available SigLIP2 so400m Models** (from HuggingFace):

| Model | Hidden Dim | Resolution | Notes |
|-------|-----------|------------|-------|
| [google/siglip2-so400m-patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384) | 1152 | 384x384 | **Likely candidate** |
| [google/siglip2-so400m-patch14-224](https://huggingface.co/google/siglip2-so400m-patch14-224) | 1152 | 224x224 | Lower resolution |
| [google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) | 1152 | 384x384 | SigLIP v1 (not v2) |
| google/siglip2-so400m-patch16-naflex | 1152 | Variable | NaFlex variant |

All so400m models have 1152 hidden dimensions, matching the PR's default. The PR likely uses the stock `siglip2-so400m-patch14-384` model.

**Evidence against custom model**: The PR code shows standard model loading with no special weights.

### 2. When Will Weights Release?

The diffusers PR discussion doesn't mention a timeline for omni weight release. The PR focuses on code architecture, suggesting weights may come later.

### 3. Training Data Requirements?

To train siglip components ourselves, we would need:
- Paired (image, text, target) training data
- The Z-Image training recipe
- Significant compute resources

---

## Files and Resources

### Created Files

| File | Description |
|------|-------------|
| `experiments/siglip/README.md` | Overview and architecture summary |
| `experiments/siglip/diffusers_pr_12857_breakdown.md` | **Detailed PR analysis** |
| `experiments/siglip/siglip_encoder.py` | SigLIP2 encoder wrapper |
| `experiments/siglip/test_siglip_encoding.py` | Encoder unit tests |
| `experiments/siglip/test_omni_pipeline.py` | Minimal pipeline test |
| `experiments/siglip/test_full_omni_generation.py` | End-to-end generation test |
| `experiments/siglip/findings.md` | This document |

### External Resources

| Resource | URL |
|----------|-----|
| Diffusers PR 12857 | https://github.com/huggingface/diffusers/pull/12857 |
| RuoyiDu's Commit | https://github.com/RuoyiDu/diffusers/commit/3435ba2a9f686e8adff8dbaaf5ed2746c2bd0327 |
| SigLIP2 Model | https://huggingface.co/google/siglip2-so400m-patch14-384 |

### Local Model Paths

| Model | Path |
|-------|------|
| Z-Image Turbo | `~/Storage/Tongyi-MAI_Z-Image-Turbo` |
| SigLIP2-so400m | `~/Storage/google_siglip2-so400m-patch14-384` |
| Qwen3-4B | `~/Storage/Qwen3-4B` |

---

## Conclusions

1. **Architecture**: Z-Image Omni's architecture is fully understood and documented
2. **Implementation**: Pipeline runs end-to-end, basic mode works correctly
3. **Blocker**: siglip_embedder/siglip_refiner require trained weights not yet released
4. **No Workaround**: Random/heuristic initialization produces noise - proper training required
5. **Next Steps**: Wait for official weight release, or attempt training if resources available

---

## Appendix: Z-Image Turbo Checkpoint Keys

All 521 keys from the transformer checkpoint:

```
cap_embedder.0.weight
cap_embedder.1.weight, cap_embedder.1.bias
cap_pad_token
context_refiner.0.* (attention, ffn, norms)
context_refiner.1.* (attention, ffn, norms)
final_layer.adaLN_modulation.1.weight, .bias
final_layer.linear.weight, .bias
layers.0-29.* (30 main transformer layers)
noise_refiner.0.* (attention, ffn, norms, adaln)
noise_refiner.1.* (attention, ffn, norms, adaln)
t_embedder.mlp.0.weight, .bias
t_embedder.mlp.2.weight, .bias
x_embedder.proj.weight, .bias
x_pad_token

NO siglip_* keys
```
