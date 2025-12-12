# Model Configuration Comparison: Qwen3-4B vs Qwen3-VL

> **Last Updated:** 2025-12-12

This document details the architectural differences between Qwen3-4B (Z-Image's text encoder) and Qwen3-VL-4B, and their implications for vision conditioning.

## Summary

Despite sharing the same base architecture (`hidden_size=2560`, 36 layers), the models have critical differences in positional encoding that affect embedding compatibility.

## Configuration Comparison

### Core Architecture (Identical)

| Parameter | Qwen3-4B | Qwen3-VL text_config |
|-----------|----------|---------------------|
| hidden_size | 2560 | 2560 |
| num_hidden_layers | 36 | 36 |
| num_attention_heads | 32 | 32 |
| head_dim | 128 | 128 |
| num_key_value_heads | 8 | 8 |
| intermediate_size | 9728 | 9728 |
| hidden_act | silu | silu |
| vocab_size | 151936 | 151936 |

### RoPE Configuration (DIFFERENT)

| Parameter | Qwen3-4B | Qwen3-VL |
|-----------|----------|----------|
| **rope_theta** | **1,000,000** | **5,000,000** |
| **rope_scaling** | **None** | **MRoPE** |
| mrope_interleaved | N/A | True |
| mrope_section | N/A | [24, 20, 20] |

### Impact Analysis

The RoPE difference means:
- **Different base frequency**: 5x higher theta in VL = slower rotation rate
- **Multi-axis RoPE (MRoPE)**: VL uses 3 axes for (temporal, height, width) positioning
- **Section division**: [24, 20, 20] = 64 total dimension pairs split across 3 axes

For text-only tokens, all 3 axes use the same 1D position. For image tokens, axes encode (1, h, w) grid positions.

## Empirical Distribution Analysis

We analyzed hidden states from layer -2 across both models.

### Global Statistics

| Source | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Qwen3-4B text | 0.31 | 61.1 | -4544 | 13504 |
| VL text tokens | 0.33 | 47.8 | -3008 | 8960 |
| VL image tokens | 0.09 | 7.0 | -44 | 384 |

### Per-Dimension Correlation

| Comparison | Correlation | Median Ratio | Mean |ratio-1| |
|------------|-------------|--------------|-------------------|
| VL text vs Qwen3-4B | **0.999** | 1.11x | 0.27 |
| VL image vs Qwen3-4B | 0.737 | 1.55x | 0.91 |

### Worst Outlier Dimensions

**For image tokens (vs Qwen3-4B):**

| Dimension | Qwen3-4B std | VL image std | Ratio |
|-----------|--------------|--------------|-------|
| 396 | 927.6 | 1.5 | **617.9x** |
| 4 | 2689.6 | 64.1 | **42.0x** |
| 100 | 75.5 | 7.8 | 9.7x |
| 329 | 23.0 | 3.3 | 6.9x |
| 9 | 37.5 | 5.7 | 6.6x |

**For text tokens (vs Qwen3-4B):**

| Dimension | Qwen3-4B std | VL text std | Ratio |
|-----------|--------------|-------------|-------|
| 1710 | 9.3 | 2.7 | 3.4x |
| 1441 | 11.0 | 3.8 | 2.9x |
| 2401 | 10.8 | 3.8 | 2.9x |

## Key Insights

### 1. Text Tokens Are Highly Compatible

Despite the RoPE mismatch, VL text token positions produce hidden states with **0.999 correlation** to Qwen3-4B. This means:
- The RoPE difference doesn't catastrophically affect text position embeddings
- `text_tokens_only=True` is validated as the correct strategy
- Simple global scaling is sufficient for text tokens

### 2. Image Tokens Are Problematic

Image token positions have:
- Only 0.737 per-dimension correlation
- Extreme outliers (up to 617x std ratio in specific dimensions)
- Fundamentally different activation patterns due to MRoPE spatial encoding

### 3. Per-Dimension Normalization Is Critical for Image Tokens

Global std scaling cannot fix per-dimension outliers. A 617x mismatch in one dimension will corrupt the entire embedding. Per-dimension normalization is required:

```python
# Z-score using input per-dim stats, rescale to Qwen3-4B per-dim stats
z_scored = (embeddings - input_mean) / input_std
normalized = z_scored * ref_std + ref_mean
```

## Recommendations

### For Best Quality
Use **text tokens only** with global scaling:
```python
result = extractor.extract(
    image,
    text="description",
    text_tokens_only=True,
    normalization_mode="global",
)
```

### For Image Token Experiments
Use **per-dimension normalization**:
```python
result = extractor.extract(
    image,
    text="description",
    image_tokens_only=True,
    normalization_mode="per_dim",
)
```

## Vision Encoder Configuration

Qwen3-VL's vision encoder is completely incompatible (different embedding space):

| Parameter | Value |
|-----------|-------|
| Vision hidden_size | 1024 (internal) |
| out_hidden_size | 2560 (projected) |
| patch_size | **16** (not 14!) |
| spatial_merge_size | 2 |
| depth | 24 layers |
| deepstack_visual_indexes | [5, 11, 17] |

**Token calculation:** For an image of size (W, H):
```python
tokens = (W // 32) * (H // 32)  # patch_size * merge_size = 32
```

## Files

- Reference statistics: `src/llm_dit/vl/qwen3_4b_stats.npz`
- Per-dimension normalization: `src/llm_dit/vl/blending.py`
- Extraction code: `src/llm_dit/vl/qwen3_vl.py`
