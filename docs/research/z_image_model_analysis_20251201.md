# Z-Image Model Architecture Analysis

**Date:** 2025-12-01
**Purpose:** Comprehensive analysis of Z-Image transformer config, training methodology, and implementation considerations

---

## Executive Summary

This document consolidates our research into the Z-Image model architecture, based on analysis of:
- Official diffusers transformer config (`transformer/config.json`)
- DiffSynth-Studio reference implementation
- Decoupled-DMD training paper (arXiv:2511.22677)
- Qwen3-4B text encoder config

**Key findings:**
1. The 512 token limit in reference implementations appears to be a choice, not a hard architectural constraint
2. The DiT uses 3D RoPE where text and image patches occupy different coordinate axes
3. CFG is "baked in" during distillation - use CFG=0 at inference
4. The context refiner operates without timestep modulation (text conditioning stays stable)

---

## 1. Transformer Configuration

From the official `transformer/config.json`:

```json
{
  "_class_name": "ZImageTransformer2DModel",
  "in_channels": 16,
  "dim": 3840,
  "n_layers": 30,
  "n_refiner_layers": 2,
  "n_heads": 30,
  "n_kv_heads": 30,
  "cap_feat_dim": 2560,
  "axes_dims": [32, 48, 48],
  "axes_lens": [1536, 512, 512],
  "rope_theta": 256.0,
  "all_patch_size": [2],
  "all_f_patch_size": [1],
  "t_scale": 1000.0,
  "qk_norm": true,
  "norm_eps": 1e-05
}
```

### 1.1 Architecture Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `in_channels` | 16 | VAE latent channels (16-channel Wan VAE) |
| `dim` | 3840 | Hidden dimension of the transformer |
| `n_layers` | 30 | Main transformer blocks (with timestep modulation) |
| `n_refiner_layers` | 2 | Context refiner (no modulation) + noise refiner |
| `n_heads` | 30 | Attention heads |
| `n_kv_heads` | 30 | KV heads (equals n_heads = full attention, not GQA) |
| `cap_feat_dim` | 2560 | Caption feature dimension (matches Qwen3-4B hidden size) |

### 1.2 RoPE Configuration

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `axes_dims` | [32, 48, 48] | RoPE dimensions per axis (sum = 128 = head_dim) |
| `axes_lens` | [1536, 512, 512] | Max positions per axis |
| `rope_theta` | 256.0 | Frequency base (much lower than LLM's 1M) |

### 1.3 Patching

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `all_patch_size` | [2] | Spatial patch size (2x2 in latent space) |
| `all_f_patch_size` | [1] | Temporal patch size (1 = no temporal compression, image model) |

### 1.4 Normalization & Scaling

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `qk_norm` | true | Apply RMSNorm to Q and K before attention (stability) |
| `norm_eps` | 1e-05 | Epsilon for layer normalization |
| `t_scale` | 1000.0 | Timestep scaling factor |

---

## 2. The 3D RoPE Coordinate System

This is the most important architectural detail for understanding token limits.

### 2.1 How It Works

The DiT uses a 3D coordinate system for position encoding:

| Axis | axes_lens | axes_dims | Purpose |
|------|-----------|-----------|---------|
| 0 (sequence) | 1536 | 32 | Sequential position for text tokens |
| 1 (height) | 512 | 48 | Spatial Y position for image patches |
| 2 (width) | 512 | 48 | Spatial X position for image patches |

### 2.2 Position Assignment

Based on DiffSynth's `patchify_and_embed`:

```python
# Text tokens: sequential along axis 0, fixed at (h=0, w=0)
cap_pos_ids = create_coordinate_grid(
    size=(cap_len, 1, 1),
    start=(1, 0, 0),  # Starts at position 1
)
# Result: [(1,0,0), (2,0,0), (3,0,0), ...]

# Image patches: continue sequence axis, use h/w for spatial
image_pos_ids = create_coordinate_grid(
    size=(1, H_tokens, W_tokens),  # F=1 for images
    start=(cap_len + 1, 0, 0),
)
# Result: All patches share ONE sequence position, spatial info in axes 1,2
```

### 2.3 Why This Matters

Text and image patches don't compete for positions:

- **Text tokens**: Use axis 0 sequentially (1, 2, 3, ... up to cap_len)
- **Image patches**: Share ONE position on axis 0 (cap_len + 1), spread across axes 1 and 2

A 1024x1024 image (64x64 patches after 8x VAE + 2x2 patching) uses:
- 1 position on axis 0
- 64 positions on axis 1
- 64 positions on axis 2

So 4096 image patches + 512 text tokens don't require 4608 sequential positions - they coexist in the 3D coordinate space.

---

## 3. The 512 Token Limit Question

### 3.1 What We Know

DiffSynth and diffusers both enforce 512 tokens:

```python
text_inputs = pipe.tokenizer(
    prompt,
    padding="max_length",
    max_length=max_sequence_length,  # 512
    truncation=True,  # Silently truncates
    return_tensors="pt",
)
```

### 3.2 What We Don't Know

We haven't found documentation confirming:
- What token lengths were used during training
- Whether 512 is a training constraint or just a conservative inference choice
- How the model behaves with longer sequences

### 3.3 Architectural Analysis

The config allows up to 1536 positions on axis 0. So 512 is **not a hard architectural limit**.

However, the low `rope_theta=256.0` is significant:

**What rope_theta controls:**

RoPE uses sine/cosine waves to encode position:
```
freqs = 1.0 / (theta ** (arange(0, d, 2) / d))
```

- **High theta (1M in Qwen3-4B)**: Waves cycle slowly. Position 1 and position 1000 look somewhat similar. Good for generalizing across long sequences.
- **Low theta (256 in DiT)**: Waves cycle fast. Position 1 and position 100 look very different. Precise local discrimination, but positions far from training data look "alien."

**Analogy:**
- High theta = ruler with inch marks - can estimate between marks
- Low theta = ruler with millimeter marks - precise nearby, but past the edge there's no reference

### 3.4 Practical Implications

Going beyond 512 tokens means:
1. The model sees position numbers it may not have seen during training
2. With theta=256, those unfamiliar positions may produce artifacts
3. It might work fine, or it might not - this is experimental territory

---

## 4. Why Full Attention in DiT vs GQA in Text Encoder

### 4.1 The Configs

**Qwen3-4B (text encoder):**
```json
"num_attention_heads": 32,
"num_key_value_heads": 8  // GQA: 4 heads share each KV pair
```

**DiT:**
```json
"n_heads": 30,
"n_kv_heads": 30  // Full attention
```

### 4.2 Why the Difference

| Aspect | Text Encoder (GQA) | DiT (Full Attention) |
|--------|-------------------|---------------------|
| Sequence length | Up to 40,960 | Typically <5,000 |
| Attention pattern | Causal (autoregressive) | Bidirectional |
| Memory concern | KV cache for long sequences | Batch processing |
| Goal | Efficient generation | Maximum expressiveness |

The text encoder is designed for potentially very long sequences with causal attention. GQA reduces memory 4x during autoregressive generation.

The DiT processes a fixed-length sequence (text + image patches) without autoregressive generation. Full attention provides maximum expressiveness for cross-modal alignment without memory pressure.

---

## 5. Connection to Decoupled-DMD Training

From the Decoupled-DMD paper (arXiv:2511.22677):

### 5.1 CFG Baking

The key insight: Z-Image Turbo has CFG "baked in" during distillation.

The training objective decomposes into:
- **CFG Augmentation (CA)** - "The Engine": Directly applies CFG signal as gradient
- **Distribution Matching (DM)** - "The Regularizer": Prevents collapse and artifacts

At inference, use `guidance_scale=0.0` because guidance is already embedded in the weights.

### 5.2 Why Context Refiner Has No Timestep Modulation

The context refiner processes text embeddings with `modulation=False`:

```python
self.context_refiner = nn.ModuleList([
    ZImageTransformerBlock(
        ...,
        modulation=False,  # No timestep conditioning
    )
    for layer_id in range(n_refiner_layers)
])
```

This makes sense given CFG baking:
- Text describes the target image, not noisy intermediates
- CFG patterns are embedded in weights during training
- Text conditioning should be stable across diffusion steps

Compare to `noise_refiner` which HAS modulation (processes timestep-dependent noisy latents).

### 5.3 8 NFEs Target

The model was distilled for 8 Number of Function Evaluations (8 steps). The official diffusers example uses `num_inference_steps=9` which produces 8 actual DiT forwards.

---

## 6. Attention Mask Filtering

DiffSynth filters padding tokens from embeddings before sending to the DiT:

```python
# From z_image.py lines 191-196
prompt_masks = text_inputs.attention_mask.to(device).bool()
prompt_embeds = pipe.text_encoder(
    input_ids=text_input_ids,
    attention_mask=prompt_masks,
    output_hidden_states=True,
).hidden_states[-2]

# Filter embeddings by attention mask - removes padding
embeddings_list = []
for i in range(len(prompt_embeds)):
    embeddings_list.append(prompt_embeds[i][prompt_masks[i]])
```

Our implementation (`filter_padding=True` default) matches this behavior.

**Why filter if context_refiner handles padding?**

The context_refiner has a learned `cap_pad_token` for padding positions. But DiffSynth filters first anyway. Possible reasons:
1. Reduces sequence length (memory/compute savings)
2. Removes potential artifacts from long padding sequences
3. May have been the pattern during training

---

## 7. Key Configuration Relationships

```
head_dim = dim / n_heads = 3840 / 30 = 128
sum(axes_dims) = 32 + 48 + 48 = 128 = head_dim  // Must match

cap_feat_dim = 2560 = Qwen3-4B hidden_size
in_channels = 16 = Wan VAE latent channels

patch_embedder_input = patch_size^2 * in_channels = 2*2*16 = 64
patch_embedder_output = dim = 3840
```

---

## 8. Resolution Limits

From the config:
- `axes_lens[1,2]` = 512 each (spatial positions)
- `patch_size` = 2
- VAE compression = 8x

**Theoretical max resolution:**
512 * 2 * 8 = 8192px per dimension

In practice, VRAM limits this well before 8192px.

**Alignment:**
DiffSynth uses 16-pixel alignment:
```python
height_division_factor=16, width_division_factor=16
```

The `SEQ_MULTI_OF=32` in the DiT is for internal sequence padding, not image resolution.

---

## 9. Summary: What We Know vs What We Infer

### Confirmed (from configs/code):
- 3D RoPE with `axes_lens=[1536, 512, 512]`
- `rope_theta=256.0` (low, sharp position encoding)
- Full attention in DiT, GQA in text encoder
- Context refiner has no timestep modulation
- Reference implementations truncate at 512 tokens
- Attention mask filtering in DiffSynth

### Inferred (reasonable but not confirmed):
- 512 token limit may match training data (or may just be conservative)
- Low rope_theta means position extrapolation is risky
- CFG=0 is required due to DMD distillation

### Unknown:
- Actual training data token length distribution
- How the model behaves with >512 tokens in practice
- Whether artifacts beyond 512 are common or rare

---

## 10. Practical Recommendations

| Setting | Value | Confidence |
|---------|-------|------------|
| CFG | 0 | High (documented in official examples) |
| Steps | 8-9 | High (distillation target) |
| Token limit | 512 (soft) | Medium (reference implementations use it) |
| Attention mask filtering | True | Medium (matches DiffSynth) |
| Resolution alignment | 16px | High (from DiffSynth) |

**For users exceeding 512 tokens:**
This is experimental. It might work fine, or it might cause artifacts. If you see issues with very long prompts (system + thinking + multi-turn), try simplifying.

---

## 11. References

- Official model: `Tongyi-MAI/Z-Image-Turbo` on HuggingFace
- Decoupled-DMD paper: arXiv:2511.22677
- Z-Image Technical Report: arXiv:2511.22699
- DiffSynth-Studio: `diffsynth/pipelines/z_image.py`
- Our analysis docs:
  - `internal/z_image_context_refiner_deep_dive.md`
  - `internal/z_image_diffsynth_analysis_20251201.md`
  - `internal/z_image_paper_analysis/decoupled_dmd_training_report.md`
  - `internal/z_image_paper_alignment/`
