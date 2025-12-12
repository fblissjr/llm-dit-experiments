# Qwen3-VL Conditioning Guide: Parameters and Tradeoffs

> **Last Updated:** 2025-12-12 (added outlier masking)

**IMPORTANT**: This is an EXPERIMENTAL approach that produces visible artifacts even with optimal settings. This guide documents parameter choices and their effects, but users should expect quality degradation compared to pure text generation or trained methods like IP-Adapter.

This guide explains how to control Qwen3-VL vision embedding influence on Z-Image generation and the quality tradeoffs involved:

## Overview of Control Parameters

```
                    CONTROL PARAMETERS
                    ==================

REFERENCE IMAGE -----> [Qwen3-VL] -----> VL Embeddings
                           |
                           |  Knobs:
                           |  - hidden_layer (-1 to -36)
                           |  - image_tokens_only (bool)
                           |  - text_tokens_only (bool)
                           |  - text_with_image (str)
                           |  - normalization_mode (global/per_dim/hybrid)
                           |  - outlier_masking (none/zero/clamp/scale)
                           |  - outlier_threshold (default: 10.0)
                           |
                           v
                    +-------------+
                    | VL Embeds   |
                    +-------------+
                           |
                           |  Knob: alpha (0.0 - 1.0)
                           |
                           v
TEXT PROMPT -------> [Blend] <---------+
                        |
                        v
                  +-------------+
                  | Blended     |
                  | Embeddings  |
                  +-------------+
                        |
                        v
                  [Z-Image DiT]
                        |
                        v
                  Generated Image
```

## 1. Alpha (Interpolation Ratio)

**What it controls**: How much the reference image vs text prompt influences the output.

| Alpha | VL Influence | Text Influence | Use Case |
|-------|--------------|----------------|----------|
| 0.0 | 0% | 100% | Pure text (baseline) |
| 0.1-0.2 | 10-20% | 80-90% | Subtle style hints |
| **0.3** | **30%** | **70%** | **Recommended default** |
| 0.4-0.5 | 40-50% | 50-60% | Strong style transfer |
| 0.6-0.7 | 60-70% | 30-40% | Preserve reference composition |
| 0.8-0.9 | 80-90% | 10-20% | Near-reproduction |
| 1.0 | 100% | 0% | Pure VL (experimental) |

**Key insight**: At alpha > 0.5 with unrelated content, VL dominates and text semantics get lost.

```python
# Style transfer (keep text content, add VL style)
blended = 0.2 * vl_emb + 0.8 * text_emb

# Balanced blend
blended = 0.3 * vl_emb + 0.7 * text_emb

# Reference-heavy (reproduce similar to reference)
blended = 0.6 * vl_emb + 0.4 * text_emb
```

## 2. Hidden Layer Selection

**What it controls**: Which layer of the VL model to extract embeddings from.

| Layer | Artifact Severity | Notes |
|-------|-----------------|----------|
| -1 (last) | Heavy | Most abstract, task-specific |
| -2 (penultimate) | Heavy | Default for Qwen3-4B text, but bad for VL. Has 617x outlier in dim 396. |
| -4 | Moderate | Better than -2, artifacts still visible |
| **-6** | **Least (but still visible)** | **Best layer for VL - crisp images, NO outliers** |
| -8 | Moderate | Previously recommended, -6 is better |
| -16 | Moderate-Heavy | Good center, edge artifacts |
| -24 | Severe | Heavy border noise |

**Important (2025-12-12):** Layer -6 is the best choice for VL conditioning - produces crisper images than -2 or -8 and has NO outlier dimensions. The 617x outlier in dimension 396 is specific to layer -2.

```python
# Recommended for VL conditioning (2025-12-12)
outputs.hidden_states[-6]  # Best: crisp images, no outliers

# Alternative (previously recommended)
outputs.hidden_states[-8]  # Good, but -6 is better

# Default for text encoding (avoid for VL)
outputs.hidden_states[-2]  # Has 617x outlier in dim 396
```

## 3. Image Tokens Only vs Text Tokens vs Full Sequence

**What it controls**: Whether to use only the image token embeddings, text tokens, or both.

**CRITICAL FINDING (2025-12-12):** Per-dimension analysis shows:
- **Text tokens**: 0.999 correlation with Qwen3-4B (nearly identical!)
- **Image tokens**: 0.737 correlation with extreme outliers (up to 617x std ratio)

| Mode | Tokens Included | Per-dim Correlation | Recommended Use |
|------|-----------------|---------------------|-----------------|
| **Text tokens only** | Just text positions | **0.999** | **Recommended - best quality** |
| Full sequence | Image + text + system | Mixed | Compromise |
| Image only | Just image tokens | 0.737 | Use with per_dim normalization |

**When to use text-only (PRODUCES FEWER ARTIFACTS)**:
- Default choice - fewer artifacts than image tokens
- Text tokens carry both prompt AND image context via self-attention
- Can use alpha=1.0 (pure VL) but expect quality loss vs pure text

**When to use image-only (EXPERIMENTAL)**:
- Experimental style transfer attempts
- MUST use `normalization_mode="per_dim"` to reduce outlier effects
- Expect more severe artifacts even with per_dim normalization

```python
# Recommended: text tokens only
result = extractor.extract(
    image=img,
    text=prompt,
    text_tokens_only=True,           # NEW recommended default
    normalization_mode="global",     # Simple scaling is sufficient
)

# Experimental: image tokens with per-dim normalization
result = extractor.extract(
    image=img,
    text=prompt,
    image_tokens_only=True,
    normalization_mode="per_dim",    # CRITICAL for image tokens
)
```

## 4. Text Description with Image

**What it controls**: Whether to include a text description when processing the reference image through Qwen3-VL.

| Mode | Example | Effect |
|------|---------|--------|
| No text | Just image | Pure visual features |
| Generic description | "Describe this image" | VL interprets the image |
| **Matching description** | "A house with red roof..." | **Anchors visual features** |
| Style description | "In the style of..." | Emphasizes style aspects |

**Key insight**: Including a description that matches the reference image content significantly improves quality because it anchors the VL embeddings in semantic space.

```python
# No text (pure visual)
content = [{"type": "image", "image": image}]

# With matching description (recommended)
content = [
    {"type": "image", "image": image},
    {"type": "text", "text": "A simple cartoon house with red roof, green grass, blue sky"},
]
```

## 5. Embedding Scaling and Normalization

**What it controls**: Normalizing VL embeddings to match text embedding statistics.

**NEW: Normalization Modes (2025-12-12)**

| Mode | Use Case | Formula |
|------|----------|---------|
| `global` | Text tokens (recommended) | `emb * (target_std / current_std)` |
| `per_dim` | Image tokens (required) | `(emb - mean) / std * ref_std + ref_mean` per dimension |
| `hybrid` | Experimental | 50/50 blend of global and per_dim |

**Global statistics:**

| VL Source | Original std | After Scaling | Notes |
|-----------|--------------|---------------|-------|
| Vision encoder | ~0.57 | N/A | Incompatible (different space) |
| VL text tokens | ~47.8 | ~70.0 | 0.999 correlation - excellent |
| VL image tokens | ~7.0 | ~70.0 | 0.737 correlation - use per_dim |
| VL full sequence | ~13 | ~70.0 | Mixed |

**Why per_dim normalization is critical for image tokens:**

Image tokens have extreme per-dimension outliers:
- Dimension 396: 617x std ratio
- Dimension 4: 42x std ratio
- Global scaling amplifies these outliers catastrophically

**Usage:**
```python
# Text tokens: simple global scaling
result = extractor.extract(
    ...,
    text_tokens_only=True,
    normalization_mode="global",     # Default
    target_std=70.0,
)

# Image tokens: per-dimension normalization
result = extractor.extract(
    ...,
    image_tokens_only=True,
    normalization_mode="per_dim",    # Required to fix outliers
    target_std=70.0,
)
```

## 6. Outlier Dimension Masking

**What it controls**: Handling of dimensions with extreme std ratios vs Qwen3-4B reference.

**Background (Layer -2 Only):** Image tokens at layer -2 have severe per-dimension outliers:
- Dimension 396: 617x std ratio (vs Qwen3-4B reference)
- Dimension 4: 42x std ratio
- **Layer -6 has NO outliers** above 10x threshold

**Recommendation (2025-12-12):** Use layer -6 instead of masking at layer -2.

| Mode | Behavior | Use Case |
|------|----------|----------|
| `none` | No masking (default) | Baseline comparison |
| `zero` | Zero out outlier dimensions | Test if specific dims cause artifacts |
| `clamp` | Scale outlier dims to threshold | Preserve some signal while limiting extremes |
| `scale` | Proportionally reduce outlier dims | Gradual reduction based on ratio |

**Threshold:** Default 10.0 (mask dims with >10x std ratio vs reference)

**Usage:**
```python
# Via VLEmbeddingExtractor
result = extractor.extract(
    image=img,
    text=prompt,
    outlier_masking="zero",      # "none", "zero", "clamp", "scale"
    outlier_threshold=10.0,
)

# Check results
print(f"Masked: {result.masked_dimensions}")  # [396, 4]
print(f"Ratios: {result.masked_dim_ratios}")  # {396: 617.9, 4: 42.0}
```

**Standalone functions:**
```python
from llm_dit.vl import mask_outlier_dimensions, get_outlier_dimensions

# Analyze outliers
outliers = get_outlier_dimensions(embeddings, threshold=10.0)
# Returns: [(396, 617.9), (4, 42.0), ...]

# Apply masking
masked_emb, info = mask_outlier_dimensions(
    embeddings,
    threshold=10.0,
    mode="clamp",
)
```

**Recommended combinations:**
- **Best approach:** Use layer -6 (no outliers, no masking needed)
- Text tokens: Usually don't need masking (0.999 correlation)
- Image tokens at layer -2: Use `outlier_masking="zero"` or `"clamp"` with `normalization_mode="per_dim"`

## Use Case Recipes

### Recipe 1: Style Transfer (EXPERIMENTAL - EXPECT ARTIFACTS)

Attempt to transfer visual style from reference while generating different content.

```python
# Use text tokens at layer -6 for best results (but artifacts still present)
result = extractor.extract(
    image=style_image,
    text="flat colors, simple shapes, cartoon style",
    hidden_layer=-6,                # Best layer for VL (2025-12-12)
    text_tokens_only=True,          # Fewer artifacts than image tokens
    normalization_mode="global",    # Sufficient for text tokens
    target_std=70.0,
)

# Can use higher alpha with text tokens
blended = 0.5 * result.embeddings + 0.5 * text_emb

# Or alpha=1.0 for pure VL (expect visible quality loss)
# NOTE: Even optimal settings produce artifacts vs pure text
```

### Recipe 2: Image Variation

Generate variations of a reference image.

```python
# Higher alpha to stay closer to reference
alpha = 0.5

# Include accurate description
vl_content = [
    {"type": "image", "image": reference},
    {"type": "text", "text": "detailed description of the reference image"},
]

# Similar text prompt
text_prompt = "same description with minor variations"

# Blend
blended = 0.5 * vl_emb + 0.5 * text_emb
```

### Recipe 3: Composition Guidance (EXPERIMENTAL)

Use reference for layout/composition, text for content.

```python
# Use layer -6 for image tokens (no outliers)
result = extractor.extract(
    image=composition_ref,
    text=None,                           # No text for pure visual
    hidden_layer=-6,                     # Best layer, no outliers
    image_tokens_only=True,
    normalization_mode="per_dim",        # CRITICAL for image tokens
    target_std=70.0,
)

# Lower alpha to reduce artifacts
alpha = 0.2  # (artifacts reduced at layer -6)
blended = 0.2 * result.embeddings + 0.8 * text_emb
```

### Recipe 4: Color Palette Transfer

Transfer color scheme from reference.

```python
# Low alpha for subtle color influence
alpha = 0.15

# Include color-focused description
vl_content = [
    {"type": "image", "image": color_ref},
    {"type": "text", "text": "vibrant sunset colors, orange and purple palette"},
]

# Your content prompt
text_prompt = "A mountain landscape at dusk"

# Blend
blended = 0.15 * vl_emb + 0.85 * text_emb
```

## What VL Transfer Attempts Produce

### May Transfer (WITH ARTIFACTS):
- **Color palette influence**: Overall color scheme shifts (with corruption)
- **Visual style hints**: Cartoon vs realistic tendencies (degraded quality)
- **Mood/atmosphere**: Bright/dark/warm/cool (with visible artifacts)
- **Basic composition hints**: Spatial layout influence (unpredictable)

### Unreliable or Fails:
- **Clean style transfer**: Artifacts persist even at optimal settings
- **Specific subjects**: Often fails or corrupts text intent
- **Precise compositional control**: Use ControlNet for reliable results
- **Fine details**: Severe quality degradation
- **Consistent results**: Output quality varies unpredictably

### Does Not Work:
- **Artifact-free generation**: No parameter combination eliminates artifacts
- **Production-quality images**: Quality gap vs trained methods is significant
- **Text/writing preservation**: Rendering quality degrades

## Blending Mode Variations

### Style Delta (TESTED - FAILED)
```python
from llm_dit.vl import compute_style_delta, blend_with_style_delta

# Extract style by subtracting neutral from styled
style_delta = compute_style_delta(styled_vl_emb, neutral_vl_emb)
result = blend_with_style_delta(text_emb, style_delta, alpha=0.3)
```

**Result (2025-12-12):** Failed. Even at alpha 0.3, completely destroys content. The delta contains too much "content" information, not just style.

### AdaIN Blending (TESTED - PARTIAL SUCCESS)
```python
from llm_dit.vl import blend_adain, blend_adain_per_dim

# Per-token AdaIN (preserves content, weak style transfer)
result = blend_adain(text_emb, vl_emb, alpha=0.3)

# Per-dimension AdaIN (stronger style, corrupts content)
result = blend_adain_per_dim(text_emb, vl_emb, alpha=0.3)
```

**Results (2025-12-12):**
- `per_token`: Preserves content perfectly but NO visible style transfer
- `per_dim`: Transfers colors but corrupts subject identity
- Fundamental tradeoff: visible style transfer requires accepting content corruption

## Experimental Parameters to Explore

1. **Per-token alpha**: Different alpha for different token positions
2. **Multi-image blending**: Blend VL from multiple reference images
3. **Layer blending**: Combine hidden states from multiple layers
4. **Attention masking**: Mask certain VL tokens during attention
5. **Iterative refinement**: Use output as new reference, iterate

## Memory Considerations

| Component | VRAM | Notes |
|-----------|------|-------|
| Qwen3-VL-4B | ~8GB | Can offload to CPU |
| Z-Image DiT | ~12GB | Must be on GPU |
| Z-Image VAE | ~2GB | Can share with DiT |
| Qwen3-4B text encoder | ~8GB | Can offload to CPU |

**For RTX 4090 (24GB)**:
1. Load Qwen3-VL, extract embeddings, unload
2. Load Z-Image pipeline
3. Generate from pre-computed embeddings

Or use two-stage workflow:
```bash
# Stage 1: Extract (on any machine)
python scripts/extract_embeddings.py --image ref.png --output vl.pt

# Stage 2: Generate (on GPU machine)
python scripts/blend_and_generate.py --vl-embeddings vl.pt --prompt "..." --output result.png
```

---

## See Also

- [../research/findings.md](../research/findings.md) - Experimental results
- [../research/techniques.md](../research/techniques.md) - Deep dive into techniques
- [../research/related_work.md](../research/related_work.md) - Prior art from other domains that I'm aware of, though likely incomplete
