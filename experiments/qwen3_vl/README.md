# Qwen3-VL Vision Conditioning for Z-Image

> **Last Updated:** 2025-12-12

**EXPERIMENTAL** exploration of using Qwen3-VL hidden states to condition Z-Image generation. This approach requires no training but produces **visible artifacts** even with optimal settings. Results show that embeddings transfer partial information but quality is significantly lower than pure text generation.

## TL;DR

We observed that extracting hidden states from Qwen3-VL's text model (after it processes an image) and blending them with text embeddings can influence Z-Image generation. This is possible because Qwen3-VL and Qwen3-4B (Z-Image's text encoder) share the same base architecture (`hidden_size=2560`). However, even with optimal settings, results show visible artifacts compared to pure text generation.

**Bottom line:** Exploratory technique useful for understanding embedding spaces and architectural compatibility. For practical vision conditioning, trained adapters (like IP-Adapter) produce significantly better results.

## Why This Works (And Why It's Limited)

### The Alignment Chain

```
Qwen3-VL training aligned:    vision tokens <---> VL text model
Z-Image training aligned:     Qwen3-4B instruct hidden states <---> DiT

Mostly aligned:               VL text tokens (0.999 correlation) <---> Qwen3-4B text
NOT aligned:                  VL image tokens (0.737 correlation) <---> Qwen3-4B text
```

**Observed (2025-12-12):** VL text token positions have 0.999 per-dimension correlation with Qwen3-4B, despite RoPE configuration differences (rope_theta: 5M vs 1M, MRoPE vs standard RoPE). This suggests `text_tokens_only=True` produces fewer artifacts than image tokens.

VL image token positions have only 0.737 correlation and extreme per-dimension outliers (up to 617x std ratio), which likely contributes to the artifacts observed even with per-dimension normalization.

### Why Interpolation is Required

At alpha=1.0 (pure VL), you get recognizable but heavily artifacted images. The DiT is receiving OOD embeddings and does its best to decode them. Interpolation with text embeddings brings the signal back toward the trained distribution:

```python
# alpha controls how much OOD signal the DiT can tolerate
blended = alpha * vl_embeddings + (1 - alpha) * text_embeddings
```

Think of it like a search/retrieval problem: "Find me something in the DiT's learned distribution that's closest to this blended embedding."

### Why Same-Family Models Matter

This only works because:
1. `hidden_size=2560` across both models
2. Same tokenizer
3. Same base architecture (Qwen3-4B before fine-tuning)

Attempts with other VLMs (different architectures) produced garbage even at low alpha values. The architectural compatibility is what makes zero-shot possible.

## Key Findings

### The "Content Override" Problem

When blending VL from an unrelated image with text describing different content, **VL dominates style while text loses semantic control**:

| Prompt | Reference Image | Result at alpha=0.3 |
|--------|-----------------|---------------------|
| "Homer Simpson eating spaghetti, sitting next to Abraham Lincoln in a diner" | Simple cartoon house (flat colors, blue/green) | Blue blob eating spaghetti in a house (not diner), no Lincoln |

The VL embeddings carried: flat color style, blue palette, simple shapes, house composition
The VL embeddings overrode: character identity, scene setting, second character

### Hidden Layer Effects

| Layer | Observation |
|-------|-------------|
| -1 (last) | Most abstract, task-specific |
| -2 (penultimate) | Default for Qwen3-4B text |
| -4 | Very good, clean output |
| **-8** | **Best results for VL** - cleaner than -2 |
| -16 | Good center, edge artifacts |
| -24 | Heavy border noise |
| -5 to -15 | Progressive "Asian bias" (Chinese training data influence) |
| -18 to -25 | "Semantic averaging" - outputs look like prototypes/category centroids |
| -30+ | Too abstract, loses all specificity |

**Important:** Layer -8 produces less artifacted results for VL than the -2 default used for Qwen3-4B text, though artifacts are still visible in generated images. The middle layer "Asian bias" likely reflects Qwen's Chinese training data distribution at the semantic prototype level.

### Embedding Statistics

**Global statistics:**

| Source | std | Notes |
|--------|-----|-------|
| Qwen3-4B text embeddings | ~61.1 | What Z-Image expects |
| VL text tokens (layer -2) | ~47.8 | **0.999 per-dim correlation** |
| VL image tokens (layer -2) | ~7.0 | 0.737 per-dim correlation |
| VL vision encoder output | 0.57 | Completely incompatible |

**Per-dimension analysis (critical discovery):**

| Token Type | Per-dim Correlation | Median Ratio | Worst Outlier |
|------------|---------------------|--------------|---------------|
| VL text tokens | **0.999** | 1.11x | 3.42x (dim 1710) |
| VL image tokens | 0.737 | 1.55x | **617x (dim 396)** |

**Observation:** VL text tokens have nearly identical per-dimension statistics to Qwen3-4B despite RoPE differences. Image tokens have extreme per-dimension outliers, but per-dimension normalization has not been sufficient to eliminate artifacts in our experiments.

**Outlier Dimension Masking:** To address image token outliers, we implemented dimension masking that zeros, clamps, or scales dimensions exceeding a threshold (default 10x std ratio). Key outliers: dim 396 (617x), dim 4 (42x). See "Outlier Dimension Masking" section below.

## Novelty Assessment

An arXiv literature search (2022-2025) suggests this specific approach has not been previously documented:

| Aspect | Our Approach | Prior Art |
|--------|--------------|-----------|
| Feature source | VLM text-model hidden states (post-vision) | Vision encoder outputs (CLIP, SigLIP) |
| Training | None (zero-shot) | Trained projection layers or adapters |
| Integration | Alpha blending in embedding space | Cross-attention injection, trained adapters |
| Key insight | Architectural compatibility (same hidden_size) | Learned alignment |

**Closest prior art:**
- **IP-Adapter**: Trained projection + decoupled cross-attention (higher quality, requires training)
- **UniFusion**: VLM as encoder but requires training LAP mechanism
- **MoMA**: MLLM adapter but requires two-stage pretraining
- **GILL**: LLM-to-diffusion mapping but requires trained projection network

None of these explored architectural compatibility between same-family models as a path to training-free conditioning.

## What This Exploration Demonstrates

1. **Architectural compatibility matters** - Same hidden dimensions enable embedding transfer
2. **Statistical alignment is measurable** - Per-dimension correlation reveals compatibility limits
3. **Quality tradeoffs are real** - Training-free approaches sacrifice quality for simplicity
4. **Research direction** - Shows what's possible without training and where the limits are

## What This Is NOT Suitable For

1. **Production image conditioning** - Use IP-Adapter or similar trained methods for practical use
2. **High-quality style transfer** - Even optimal settings produce visible artifacts
3. **Reliable content preservation** - VL influence is unpredictable; text semantics can be lost

## Outlier Dimension Masking

Image tokens have extreme per-dimension outliers compared to Qwen3-4B reference statistics. We implemented masking functions to handle these:

### Key Outlier Dimensions (Layer -2 Only)

| Dimension | Std Ratio | Impact |
|-----------|-----------|--------|
| **396** | **617x** | Most severe outlier at layer -2 |
| **4** | **42x** | Second worst outlier at layer -2 |
| 1710 | 3.4x | Minor outlier (text tokens worst) |

**Important:** These outliers are layer-specific. Layer -6 has NO outliers above 10x threshold.

### Masking Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `none` | No masking (default) | Baseline comparison |
| `zero` | Zero out outlier dimensions | Test if dim is the culprit |
| `clamp` | Scale outlier dims to threshold level | Preserve some signal |
| `scale` | Proportionally reduce outlier dims | Gradual reduction |

### Usage

**CLI (experiment runner):**
```bash
# Test all masking modes
uv run experiments/qwen3_vl/scripts/run_comparison.py \
    -i reference.png \
    -p "Your prompt" \
    --token-modes full \
    --outlier-masking none zero clamp scale \
    --outlier-threshold 10.0 \
    --alphas 1.0

# Use sweep preset
uv run experiments/qwen3_vl/scripts/run_comparison.py \
    -i reference.png \
    -p "Your prompt" \
    --sweep outlier
```

**Python API:**
```python
from llm_dit.vl import VLEmbeddingExtractor, mask_outlier_dimensions, get_outlier_dimensions

# Via extract method
result = extractor.extract(
    image=img,
    text=prompt,
    outlier_masking="zero",      # "none", "zero", "clamp", or "scale"
    outlier_threshold=10.0,      # Mask dims with >10x std ratio
)

# Check which dimensions were masked
print(f"Masked dims: {result.masked_dimensions}")
print(f"Ratios: {result.masked_dim_ratios}")

# Or use standalone functions
outliers = get_outlier_dimensions(embeddings, threshold=10.0)
# Returns: [(396, 617.9), (4, 42.0), ...]

masked, info = mask_outlier_dimensions(embeddings, threshold=10.0, mode="zero")
```

### Expected Results

- `none`: Baseline with full outlier effects
- `zero`: Tests hypothesis that specific dimensions cause artifacts
- `clamp`: Balanced approach preserving some outlier signal
- `scale`: Gradual proportional reduction

## VL + img2img Combined (CORRECTION - 2025-12-12)

**IMPORTANT:** Earlier tests showing "landscape composition" were misleading. The success was from **pure img2img**, not VL conditioning. With `text_tokens_only=True`, VL wasn't contributing anything.

### What Actually Happened

The original test used `text_tokens_only=True`, which strips out all image information:
- VL embeddings with `text_tokens_only=True` ≈ pure text embeddings
- The "success" was just img2img working well with landscapes
- vl=0.0 and vl=0.3 produced nearly identical results (proof VL wasn't doing anything)

### Corrected Understanding

When VL is configured correctly (`text_tokens_only=False`):

| Alpha | Result |
|-------|--------|
| 0.3 | Slight style influence from reference image |
| 0.5 | Strong style transfer (e.g., photorealistic from photo reference) |
| 0.7 | Reference image strongly influences output |
| 1.0 | Essentially reconstructs reference scene |

### Layer -2 vs -6 with Full Image Tokens

| Layer | Result |
|-------|--------|
| **-2** | Heavy glitch artifacts (617x outlier in dim 396) |
| **-6** | Clean results |

### Corrected Test Script

The `test_vl_img2img.py` script needs to be updated to use `text_tokens_only=False`:

```python
vl_result = vl_extractor.extract(
    image,
    text=args.prompt,
    hidden_layer=-6,           # Not -2
    text_tokens_only=False,    # CRITICAL: Must be False!
    scale_to_text=True
)
```

### What VL Actually Does (When Configured Correctly)

With `text_tokens_only=False` and layer -6:
- Transfers visual style from reference (photorealistic, cartoon, etc.)
- At high alpha, reconstructs the reference scene
- Works as actual vision conditioning, not just text embedding variation

## Recommended Next Steps

| Priority | Action | Why |
|----------|--------|-----|
| 1 | Test more landscape types | Confirm pattern holds for various backgrounds |
| 2 | Train minimal adapter (single linear layer) | Test if small amounts of training eliminate artifacts |
| 3 | Systematic layer sweep with metrics | Quantify quality vs layer depth |
| 4 | Compare with IP-Adapter baseline | Measure quality gap vs trained methods |
| 5 | Test on second model family | Determine if approach generalizes |

**CRITICAL CORRECTION (2025-12-12):**

Previous documentation was WRONG. `text_tokens_only=True` strips out ALL image information, making VL conditioning useless. The correct settings are:

```python
vl_result = vl_extractor.extract(
    image,
    text=prompt,
    hidden_layer=-6,          # NOT -2 (has 617x outlier causing artifacts)
    text_tokens_only=False,   # MUST be False to include image information!
    scale_to_text=True
)
```

**Why this matters:**
- Image information is encoded in **image token positions** (~1026 tokens for 1024x1024)
- `text_tokens_only=True` only keeps ~15-20 text tokens, discarding all visual info
- With `text_tokens_only=True`, VL embeddings are functionally identical to pure text embeddings

**Correct settings for VL conditioning:**
- `hidden_layer=-6` (layer -2 has 617x outlier causing glitch artifacts)
- `text_tokens_only=False` (REQUIRED to include image information)
- `scale_to_text=True` (normalize std to match Qwen3-4B)
- `alpha=0.3-0.7` (controls VL influence, 1.0 = reconstruct reference image)

## Parameters Tested (Complete List)

all variables experimented with during vl conditioning research:

the pipeline has 3 sources of input:
1. **qwen3-vl**: processes reference image + prompt -> vl embeddings (~1045 tokens: ~1026 image +  ~19 text tokens)
2. **qwen3-4b**: processes prompt only -> text embeddings ~15-20 tokens
3. **vae encoder**: processes reference image -> latent for img2img start

vl extraction (from qwen3-vl)
- `hidden_layer`: -1, -2, -4, -6, -8, -16, -24, -30 (winner -6)
- `text_tokens_only`:  true/false (winner: false - true strips all image info making vl useless)
- `image_tokens_only`: true/false
- `scale_to_text`: true/false (norms vl std to ~70 to match qwen3-4b)
- `target_std`: 70.0 (match qwen3-4b text embedding std)
- `normalization_mode`: none, global,per_dim
- `outlier_masking`: none, zero, clamp, scale
- `outlier_threshold`: 10.0 (std ratio cutoff)

blending (vl embeddings + text embeddings)
- `alpha`: 0.0, 0.1, 0.3, 0.5, 0.7, 1.0 (0=pure text, 1=pure vl)
- `blend_mode`: linear, adain, adain_per_dim, style_delta
- formula: `blended = alpha * vl_emb + (1-alpha) * text_emb`

img2img (vae encode first, then pass embeds)
- `strength`: 0.3, 0.5, 0.7, 0.8, 0.9 (0=preserve input,1=full regen)
- `prompt_embeds`: pass blended embeddings as conditioning
- uses same ref image encoded by vae as the starting latent

pipeline workflow across all: 9 steps. seed 42

what i found each component seems to do:
- vl embedding:  "vibe" from reference image
- text embeddings from qwen3-4b: semantic content from prompt
- img2img vae encode: structure/composition from reference image
- blended embeddings: combined conditioning for dit, best result

failed approaches:
- `text_tokens_only=true`: vl embeddings dont work (not enough image info)
- `layer -2 + text_tokens_only=false`: heavy glitch artifacts 
- `style_delta`: destroyed all concepts even at alpha=0.3
- `adain_per_token`: preserved content but no visible style transfer
- `adain_per_dim`: transferred color but corrupted subject identity (homer morphing into a house)

**working configuration (verified):**
```python
# vl extraction from qwen3-vl
vl_emb = extractor.extract(
    image=reference_image,        # image to extract style/scene from
    text=prompt,                  # same prompt used for text encoding
    hidden_layer=-6,              # not -2 (has glitch artifacts)
    text_tokens_only=False,       # MUST be false to include image tokens
    scale_to_text=True            # normalize std to match qwen3-4b
).embeddings

# text encoding from qwen3-4b
text_emb = pipe.encoder.encode(prompt).embeddings[0]

# blend vl + text
blended = alpha * vl_emb + (1 - alpha) * text_emb  # alpha=0.3-0.7

# generate with img2img (optional - adds structure preservation)
result = pipe.img2img(
    prompt_embeds=blended,        # blended conditioning
    image=reference_image,        # same image for vae latent start
    strength=0.7                  # how much to regenerate vs preserve
)
```

**what you get at different alpha values (with text_tokens_only=false):**
- alpha=0.0: pure text generation (baseline)
- alpha=0.3: slight style influence from reference
- alpha=0.5: strong style transfer (e.g., photorealistic from photo)
- alpha=0.7: reference strongly dominates
- alpha=1.0: reconstructs reference scene

## Documentation Structure

```
experiments/qwen3_vl/
  README.md                    # This file - start here
  __init__.py                  # Python module init
  scripts/                     # Executable scripts
    extract_embeddings.py      # Extract VL embeddings from images
    blend_and_generate.py      # Blend VL + text and generate
    run_comparison.py          # Systematic comparison sweeps
  docs/
    guides/                    # How-to guides
      parameters.md            # Parameter guide and use cases
      interpreting.md          # How to interpret experiment results
    research/                  # Research findings and analysis
      findings.md              # Main research findings (start here for research)
      quick_reference.md       # Quick lookup for specific questions
      techniques.md            # Deep dive on techniques from literature
      model_config_comparison.md  # Qwen3-4B vs Qwen3-VL comparison
      related_work.md          # Prior art from other domains
    experiments/               # Experiment-specific documentation
      token_position.md        # Token position experiments and findings
```

## Quick Start

```bash
# Extract VL embeddings from reference image
uv run experiments/qwen3_vl/scripts/extract_embeddings.py \
    --config config.toml --profile rtx4090 \
    --image reference.png \
    --output embeddings.pt

# Generate with VL conditioning
uv run experiments/qwen3_vl/scripts/blend_and_generate.py \
    --config config.toml \
    --vl-embeddings embeddings.pt \
    --prompt "Your text prompt" \
    --alpha 0.3 \
    --output result.png

# Run alpha sweep comparison
uv run experiments/qwen3_vl/scripts/run_comparison.py \
    --config config.toml \
    --image reference.png \
    --prompt "Your prompt" \
    --experiment alpha_sweep
```

## Architecture Diagram

```
REFERENCE IMAGE                          TEXT PROMPT
     |                                        |
     v                                        v
+------------+                         +------------+
| Qwen3-VL   |                         | Qwen3-4B   |
| ViT Vision |                         | Tokenizer  |
| Encoder    |                         +------------+
+------------+                                |
     |                                        v
     | vision features                 +------------+
     | (projected to 2560 dim)         | Qwen3-4B   |
     |                                 | Text Model |
     v                                 | (Z-Image)  |
+------------------+                   +------------+
| Qwen3-VL         |                          |
| Text Model       |                          |
| (same arch as    |                          |
| Qwen3-4B but     |                          |
| different fine-  |                          |
| tuning)          |                          |
+------------------+                          |
     |                                        |
     v                                        v
+------------------+                   +------------------+
| Hidden States    |                   | Hidden States    |
| (OOD - close     |                   | (in-distribution)|
| but not exact)   |                   |                  |
| std ~13          |                   | std ~58          |
+------------------+                   +------------------+
     |                                        |
     |    +---------------------------+       |
     +--->|     INTERPOLATION         |<------+
          | alpha * VL + (1-a) * text |
          | (finding OOD tolerance)   |
          +---------------------------+
                      |
                      v
              +---------------+
              | Z-Image DiT   |
              | (unchanged)   |
              +---------------+
                      |
                      v
              +---------------+
              | Generated     |
              | Image         |
              +---------------+
```

## Where to Go From Here

### For Users
- **Getting started:** Read the [Quick Start](#quick-start) section above
- **Parameter tuning:** See [docs/guides/parameters.md](docs/guides/parameters.md) for comprehensive parameter guide
- **Interpreting results:** See [docs/guides/interpreting.md](docs/guides/interpreting.md) to understand experiment outputs

### For Researchers
- **Research findings:** Start with [docs/research/findings.md](docs/research/findings.md) for comprehensive experimental results
- **Quick reference:** See [docs/research/quick_reference.md](docs/research/quick_reference.md) for specific questions
- **Techniques:** Deep dive into [docs/research/techniques.md](docs/research/techniques.md) for literature review
- **Related work:** See [docs/research/related_work.md](docs/research/related_work.md) for prior art analysis
- **Model comparison:** See [docs/research/model_config_comparison.md](docs/research/model_config_comparison.md) for RoPE and architecture details

### Experiments
- **Token position experiments:** See [docs/experiments/token_position.md](docs/experiments/token_position.md) for detailed experiment proposals

## References

### External
- [IP-Adapter](https://arxiv.org/abs/2308.06721) - Closest prior art (trained adapter approach)
- [UniFusion](https://arxiv.org/abs/2510.12789) - VLM as unified encoder (requires training)
- [MoMA](https://arxiv.org/abs/2404.05674) - MLLM adapter (requires training)
- [GILL](https://arxiv.org/abs/2305.17216) - LLM-to-diffusion mapping (requires training)
- [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) - Vision-Language Model
- [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) - DiT diffusion model
