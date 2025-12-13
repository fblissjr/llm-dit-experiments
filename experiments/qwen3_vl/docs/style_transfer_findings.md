# VL + img2img Style Transfer Pipeline

> **Last Updated:** 2025-12-12

## Executive Summary

sorta got to zero-shot style transfer (via two-stage img2img denoising + qwen3-vl/qwen3-4b blends)
1. **Qwen3-VL embeddings** - Extract style/mood from reference image
2. **img2img** - Use reference image as structural starting point
3. **Text prompt** - Specify the subject to generate

**The key finding:** High img2img strength (0.9) + moderate VL alpha (0.3-0.5) allows the text prompt subject to appear **in the style** of the reference image.

---

## Models Involved

| Model | Role | Hidden Dim | Location |
|-------|------|------------|----------|
| **Qwen3-VL-4B-Instruct** | Vision-language encoder | 2560 | Extracts embeddings from image+text |
| **Qwen3-4B** (in Z-Image) | Text encoder | 2560 | Encodes text prompts |
| **Z-Image DiT** | Diffusion transformer | 3840 | Generates images from embeddings |
| **Z-Image VAE** | Image codec | 16 channels | Encodes/decodes images to/from latents |

### Why These Models Work Together

Qwen3-VL-4B and Qwen3-4B share the same hidden dimension (2560) and similar architecture. This means:
- VL embeddings can be blended directly with text embeddings
- The DiT receives compatible conditioning regardless of source
- No projection layer needed (though one could improve alignment)

---

## Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT STAGE                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   REFERENCE IMAGE                         TEXT PROMPT                        │
│   (style_anime.png)                       "Homer Simpson"                    │
│         │                                       │                            │
│         ▼                                       ▼                            │
│   ┌─────────────┐                        ┌─────────────┐                     │
│   │ Qwen3-VL    │                        │ Qwen3-4B    │                     │
│   │ Vision      │                        │ Tokenizer   │                     │
│   │ Encoder     │                        └─────────────┘                     │
│   └─────────────┘                              │                             │
│         │                                      ▼                             │
│         ▼                               [11 text tokens]                     │
│   [1026 image tokens]                          │                             │
│         │                                      ▼                             │
│         ▼                               ┌─────────────┐                      │
│   ┌─────────────┐                       │ Qwen3-4B    │                      │
│   │ Qwen3-VL    │◄── "Homer Simpson"    │ LLM Layers  │                      │
│   │ LLM Layers  │    (text prompt)      │ (36 layers) │                      │
│   │ (36 layers) │                       └─────────────┘                      │
│   └─────────────┘                              │                             │
│         │                                      │                             │
│         ▼                                      ▼                             │
│   Extract layer -6                       Extract layer -2                    │
│         │                                      │                             │
│         ▼                                      ▼                             │
│   VL EMBEDDINGS                          TEXT EMBEDDINGS                     │
│   (1041, 2560)                           (11, 2560)                          │
│   ~1026 image + ~15 text                 pure text tokens                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BLENDING STAGE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   VL EMBEDDINGS (1041, 2560)          TEXT EMBEDDINGS (11, 2560)            │
│         │                                      │                             │
│         │              scale_to_text=True      │                             │
│         ▼              (normalize std)         │                             │
│   VL std: 13 → 70                              │                             │
│         │                                      │                             │
│         └──────────────┬───────────────────────┘                             │
│                        │                                                     │
│                        ▼                                                     │
│              BLEND MODE (choose one):                                        │
│                                                                              │
│              [interpolate] (RECOMMENDED - default)                           │
│                Linear interpolation VL (1041) → (11) tokens                  │
│                Preserves all VL information via resampling                   │
│                                                                              │
│              [adain_per_dim] (best for style transfer)                       │
│                Transfer per-dimension VL statistics to text structure        │
│                Preserves text content, applies VL style distribution         │
│                                                                              │
│              [adain]                                                         │
│                Transfer global VL statistics (mean/std) to text              │
│                                                                              │
│              [linear] (WARNING: loses info)                                  │
│                TRUNCATES VL to first 11 tokens (loses 99% of image info)     │
│                                                                              │
│                        │                                                     │
│                        ▼                                                     │
│              blended = alpha * vl_processed + (1-alpha) * text               │
│              blended = 0.3 * vl + 0.7 * text                                 │
│                        │                                                     │
│                        ▼                                                     │
│              BLENDED EMBEDDINGS (11, 2560)                                   │
│              Contains: 70% text semantics + 30% VL style                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           IMG2IMG STAGE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   REFERENCE IMAGE                        BLENDED EMBEDDINGS                  │
│   (style_anime.png)                      (11, 2560)                          │
│         │                                      │                             │
│         ▼                                      │                             │
│   ┌─────────────┐                              │                             │
│   │ VAE Encoder │                              │                             │
│   └─────────────┘                              │                             │
│         │                                      │                             │
│         ▼                                      │                             │
│   LATENT (1, 16, 128, 128)                     │                             │
│         │                                      │                             │
│         ▼                                      │                             │
│   Add noise based on strength                  │                             │
│   strength=0.9 → 90% noise added               │                             │
│   (start from step 1 of 9)                     │                             │
│         │                                      │                             │
│         ▼                                      │                             │
│   NOISY LATENT                                 │                             │
│         │                                      │                             │
│         └──────────────┬───────────────────────┘                             │
│                        │                                                     │
│                        ▼                                                     │
│              ┌─────────────────┐                                             │
│              │   Z-Image DiT   │                                             │
│              │   (denoising)   │                                             │
│              │                 │                                             │
│              │   9 steps,      │                                             │
│              │   starting at   │                                             │
│              │   step 1        │                                             │
│              │   (8 remaining) │                                             │
│              └─────────────────┘                                             │
│                        │                                                     │
│                        ▼                                                     │
│              DENOISED LATENT (1, 16, 128, 128)                               │
│                        │                                                     │
│                        ▼                                                     │
│              ┌─────────────────┐                                             │
│              │   VAE Decoder   │                                             │
│              └─────────────────┘                                             │
│                        │                                                     │
│                        ▼                                                     │
│              OUTPUT IMAGE (1024, 1024)                                       │
│              Homer Simpson in anime style!                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Parameter Reference

### alpha (VL Influence)

**What it is:** Controls the blend ratio between VL embeddings and text embeddings.

**Math:** Just a weighted average:
```python
blended = alpha * vl_emb + (1 - alpha) * text_emb

# Examples:
# alpha=0.3 → 0.3 * vl + 0.7 * text
# alpha=0.5 → 0.5 * vl + 0.5 * text
```

**Effects:**

| Alpha | VL Weight | Text Weight | Result |
|-------|-----------|-------------|--------|
| 0.0 | 0% | 100% | Pure text prompt, no VL influence |
| 0.3 | 30% | 70% | **Optimal** - text subject + VL style |
| 0.5 | 50% | 50% | Balanced - subject may start morphing |
| 0.7 | 70% | 30% | VL dominates - subject from reference |
| 1.0 | 100% | 0% | Pure VL, text prompt ignored |

**Why 0.3-0.5 works:** At this range, the text embedding provides enough signal for the DiT to generate "Homer Simpson" while the VL embedding adds style information without overwhelming the subject.

---

### strength (img2img Denoising Strength)

**What it is:** Controls how much the reference image structure is preserved vs. transformed.

**Key insight: img2img does NOT run the full denoising process.**

- **txt2img:** Starts from pure noise → runs all N steps
- **img2img:** Starts from partially noised image → runs fewer steps

**How it works:**
```python
# 1. VAE encodes reference image → latent
# 2. Add noise based on strength (where we "start" in schedule)
# 3. Denoise from that point forward (not from the beginning!)

# strength determines starting point in denoising schedule
actual_steps = int(num_inference_steps * strength)
start_step = num_inference_steps - actual_steps

# Example with 9 total steps:
# strength=0.9 → add 90% noise, start at step 1, run 8 steps
# strength=0.7 → add 70% noise, start at step 3, run 6 steps
# strength=0.5 → add 50% noise, start at step 5, run 4 steps
# strength=0.3 → add 30% noise, start at step 7, run 2 steps
```

**From actual logs:**
```
[img2img] strength=0.9, steps=9, actual_steps=8
[img2img] Starting denoising from step 1, 8 steps remaining

[img2img] strength=0.7, steps=9, actual_steps=6
[img2img] Starting denoising from step 3, 6 steps remaining
```

**Effects:**

| Strength | Noise Added | Steps Run | Reference Preserved | Transformation |
|----------|-------------|-----------|---------------------|----------------|
| 0.3 | 30% | 2-3 | ~70% structure | Minor color/detail changes |
| 0.5 | 50% | 4-5 | ~50% structure | Moderate transformation |
| 0.7 | 70% | 6-7 | ~30% structure | Significant change |
| **0.9** | 90% | 8 | ~10% structure | **Major transformation** - new subject can appear |

**Why 0.9 works for style transfer:**
- At step 1, the latent is ~90% noise, ~10% reference structure
- The DiT has 8 full steps to generate something new (almost like txt2img)
- But that 10% residual structure + VL embeddings carry the style information
- This gives enough freedom for "Homer Simpson" to emerge while retaining style

**Why 0.5 fails:**
- At step 5, the latent is ~50% reference structure
- Only 4 steps to denoise - not enough to generate a new subject
- Reference subject (anime girl) dominates because structure is too preserved
- Homer cannot emerge through the existing structure

---

### hidden_layer (VL Extraction Layer)

**What it is:** Which transformer layer to extract hidden states from in Qwen3-VL.

**Layer numbering:**
```
Layer -1  = final layer (layer 36)
Layer -2  = second to last (layer 35) ← DEFAULT for text, but BAD for VL
Layer -6  = sixth from end (layer 31) ← OPTIMAL for VL
Layer -36 = first layer
```

**Why layer -6 for VL:**

| Layer | Std Ratio (dim 396) | Quality | Notes |
|-------|---------------------|---------|-------|
| -2 | **617x** | Severe artifacts | Extreme outlier in dimension 396 |
| -4 | ~100x | Some artifacts | Still problematic |
| **-6** | ~10x | Clean | Optimal balance |
| -8 | ~5x | Clean | Also works well |

The outlier in dimension 396 at layer -2 causes grid/blocky artifacts in generated images. Layer -6 avoids this while still capturing semantic content.

---

### text_tokens_only

**What it is:** Whether to return only text token positions or include image tokens.

**Token breakdown in Qwen3-VL:**
```
Input: image + "Homer Simpson"

Tokenization:
- Image → ~1026 image tokens (from vision encoder)
- Text  → ~15 text tokens (varies by prompt length)
- Total → ~1041 tokens

With text_tokens_only=True:  returns ~15 tokens (NO image info!)
With text_tokens_only=False: returns ~1041 tokens (includes image)
```

**CRITICAL:** Must be `False` for VL to work. Setting it to `True` strips all image information, making VL useless.

---

### scale_to_text

**What it is:** Normalizes VL embedding statistics to match Qwen3-4B text embeddings.

**The problem:**
```
Qwen3-4B text embeddings: std ≈ 70-86
Qwen3-VL embeddings:      std ≈ 13

Without scaling: VL signal is ~5x weaker than text
With scaling:    VL scaled to match text std
```

**Math:**
```python
scale_factor = target_std / vl_std  # e.g., 70 / 13 ≈ 5.3
scaled_vl = vl_embeddings * scale_factor
```

**Why it matters:** Without scaling, even alpha=0.5 would be heavily text-dominated because the VL embeddings have much smaller magnitude.

---

## The Blending Process

### Step 1: Sequence Length Matching

VL and text embeddings have different sequence lengths:
```
VL:   (1041, 2560)  - ~1026 image + ~15 text tokens
Text: (11, 2560)    - pure text tokens
```

The `blend_embeddings()` function handles this via interpolation:
```python
def blend_embeddings(vl_emb, text_emb, alpha):
    # Interpolate longer sequence to match shorter
    if vl_emb.shape[0] != text_emb.shape[0]:
        # Use linear interpolation along sequence dimension
        vl_emb = F.interpolate(
            vl_emb.unsqueeze(0).transpose(1, 2),
            size=text_emb.shape[0],
            mode='linear'
        ).transpose(1, 2).squeeze(0)

    # Blend
    return alpha * vl_emb + (1 - alpha) * text_emb
```

This compresses 1041 VL tokens into 11 tokens, preserving overall style information.

### Step 2: Alpha Blending

Simple weighted average:
```python
blended = alpha * vl_scaled + (1 - alpha) * text
```

The resulting embeddings contain:
- **From VL:** Color palette, artistic style, mood, composition hints
- **From text:** Subject identity, semantic content

---

## What Each Component Contributes

| Component | Contributes | Controlled By |
|-----------|-------------|---------------|
| **VL embeddings** | Style, color palette, mood, artistic feel | `alpha`, `hidden_layer` |
| **Text embeddings** | Subject identity ("Homer Simpson") | `1 - alpha` |
| **img2img latent** | Structural layout, composition | `strength` |
| **VAE encoder** | Initial image structure → latent | Reference image choice |
| **DiT** | Final image synthesis | All of the above |

---

## Working Configuration

### Optimal Parameters

```python
# VL Extraction
vl_result = vl_extractor.extract(
    image=reference_image,
    text="Homer Simpson",       # Same prompt as generation
    hidden_layer=-6,            # Avoid layer -2 outliers
    text_tokens_only=False,     # MUST be False
    scale_to_text=True,         # Normalize to text std
)

# Blending (use blend_interpolate, NOT blend_embeddings)
from llm_dit.vl import blend_interpolate, blend_adain_per_dim

# Option 1: Interpolate (RECOMMENDED - compresses all VL tokens)
blended = blend_interpolate(vl_emb, text_emb, alpha=0.3)

# Option 2: AdaIN per-dim (best for pure style transfer)
blended = blend_adain_per_dim(text_emb, vl_emb, alpha=0.3)

# WARNING: blend_embeddings() TRUNCATES (loses 99% of VL info) - avoid!

# Generation
result = pipe.img2img(
    prompt_embeds=blended,
    image=reference_image,      # Same image as VL source
    strength=0.9,               # High strength for transformation
    num_inference_steps=9,
)
```

### Blend Mode Comparison (2025-12-12)

| Mode | How It Works | Best For | Std |
|------|--------------|----------|-----|
| `interpolate` | Compresses all 1041 VL tokens to 11 via linear interpolation | General use, preserves all info | 62 |
| `adain_per_dim` | Transfers VL per-dimension statistics to text structure | Style transfer | 193 |
| `adain` | Transfers global VL mean/std to text | Softer style influence | 150 |
| `linear` (old default) | TRUNCATES to first 11 tokens | **AVOID - loses 99% info** | 148 |

**Visual comparison grid:** `experiments/results/vl_blend_methods/grid_comparison.png`

### Parameter Combinations That Work

| Style Complexity | Alpha | Strength | Result |
|------------------|-------|----------|--------|
| Simple (flat cartoon) | 0.3 | 0.9 | Excellent style transfer |
| Medium (anime) | 0.3-0.5 | 0.9 | Good, some style bleeding |
| Complex (photorealistic) | 0.5 | 0.9 | Style transfers but fights with subject |

---

## Experimental Results

### Test: Flat Cartoon House → Homer

**Reference:** Simple red house, flat colors, blue sky, yellow sun

| Alpha | Strength | Result |
|-------|----------|--------|
| 0.3 | 0.7 | House preserved |
| 0.3 | 0.8 | House with slight changes |
| **0.3** | **0.9** | **Homer appears in flat cartoon style, red pants (color bleed from house)** |

### Test: Anime Girl → Homer

**Reference:** Detailed anime girl with pink hair, cherry blossoms

| Alpha | Strength | Result |
|-------|----------|--------|
| 0.0 | 0.9 | Anime girl transforms slightly |
| 0.3 | 0.9 | Homer appears with pink/anime background |
| 0.5 | 0.9 | Homer more prominent, anime style retained |

### Test: Noir Detective → Homer

**Reference:** B&W noir scene with detective in rain

| Alpha | Strength | Result |
|-------|----------|--------|
| 0.0 | 0.9 | Noir scene preserved |
| 0.3 | 0.9 | Homer appears in noir-tinted scene |
| 0.5 | 0.9 | Homer with B&W/grayscale influence |

### Test: Pixel Art Castle → Homer

**Reference:** 16-bit style castle scene

| Alpha | Strength | Result |
|-------|----------|--------|
| 0.3 | 0.9 | Pixel-style character begins appearing |
| 0.5 | 0.9 | Stylized pixel-art character with top hat |

---

## Why This Works

### The Three-Way Balance

1. **Text embeddings (70%)** - "Homer Simpson" provides strong subject signal
2. **VL embeddings (30%)** - Style/color information from reference
3. **img2img structure (10%)** - At strength=0.9, minimal but non-zero structure

### The Role of High Strength

At `strength=0.9`:
- The DiT has 8 of 9 steps to generate
- Initial latent is 90% noise, 10% reference structure
- This gives the text prompt's "Homer Simpson" room to emerge
- The style comes from VL embeddings, not latent structure

At `strength=0.5`:
- Only 4 steps to generate
- Initial latent is 50% reference structure
- Reference subject (anime girl) too strong to override
- Homer cannot emerge

---

## What Doesn't Work

### Low Strength (0.3-0.7)

Reference structure too dominant. Text prompt subject cannot emerge.

### High Alpha (0.7+)

VL embeddings overwhelm text. Reference subject replaces prompt subject.

### text_tokens_only=True

Strips all image information. VL has no effect.

### Layer -2 for VL

617x outlier in dimension 396 causes severe grid artifacts.

### Different Images for VL vs img2img

- VL from style image
- img2img from content image (Homer baseline)

**Result:** No visible style transfer. The img2img structure dominates, and low alpha means VL style doesn't show.

---

## Code Example

```python
from llm_dit.vl import VLEmbeddingExtractor, blend_embeddings
from llm_dit import ZImagePipeline

# 1. Load models
vl_extractor = VLEmbeddingExtractor.from_pretrained(
    "/path/to/Qwen3-VL-4B-Instruct",
    device="cpu",  # or "cuda" if enough VRAM
    torch_dtype=torch.bfloat16,
)

pipe = ZImagePipeline.from_pretrained(
    "/path/to/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    device="cuda",
)

# 2. Extract VL embeddings from style reference
reference = Image.open("style_anime.png")
vl_result = vl_extractor.extract(
    image=reference,
    text="Homer Simpson",
    hidden_layer=-6,
    text_tokens_only=False,
    scale_to_text=True,
)
vl_emb = vl_result.embeddings

# 3. Get text embeddings
text_emb = pipe.encode_prompt("Homer Simpson")

# 4. Blend (30% VL style, 70% text subject)
blended = blend_embeddings(vl_emb, text_emb, alpha=0.3)

# 5. Generate with img2img
result = pipe.img2img(
    prompt_embeds=blended,
    image=reference,
    strength=0.9,
    num_inference_steps=9,
    generator=torch.Generator().manual_seed(42),
)

# 6. Save
result.save("homer_anime_style.png")
```

---

## Command Line

```bash
# Test with any style reference
uv run python experiments/qwen3_vl/scripts/test_vl_img2img.py \
  -i experiments/inputs/style_anime_girl.png \
  -p "Homer Simpson" \
  --vl-alphas 0.0 0.3 0.5 \
  --strengths 0.7 0.8 0.9 \
  -o experiments/results/vl_img2img_anime \
  --seed 42
```

---

## Files

| File | Description |
|------|-------------|
| `test_vl_img2img.py` | Main test script for VL + img2img |
| `grid_utils.py` | Grid generation utility |
| `generate_style_refs.py` | Generate style reference images |
| `blending.py` | `blend_embeddings()` function |
| `qwen3_vl.py` | `VLEmbeddingExtractor` class |

---

## Summary

**Style transfer works when:**
- `alpha = 0.3-0.5` (enough text signal for subject)
- `strength = 0.9` (enough freedom to generate new subject)
- `hidden_layer = -6` (avoid outliers)
- `text_tokens_only = False` (include image info)
- `scale_to_text = True` (proper embedding magnitudes)

**The mechanism:**
1. VL extracts style information from reference image
2. Text embeddings provide subject identity
3. Blending creates conditioning with "subject in style"
4. High-strength img2img allows major transformation while retaining style influence
5. DiT generates the text subject rendered in the reference style
