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

## Recommended Next Steps

| Priority | Action | Why |
|----------|--------|-----|
| 1 | Characterize artifact patterns | Understand what causes corruption, document failure modes |
| 2 | Train minimal adapter (single linear layer) | Test if small amounts of training eliminate artifacts |
| 3 | Systematic layer sweep with metrics | Quantify quality vs layer depth |
| 4 | Compare with IP-Adapter baseline | Measure quality gap vs trained methods |
| 5 | Test on second model family | Determine if approach generalizes |

**Current least-artifacted settings (2025-12-12):**
- `hidden_layer=-6` (layer -6 produces crisper images than -2 or -8, no outliers found)
- `text_tokens_only=True` (image tokens have more severe artifacts)
- `normalization_mode="global"` for text tokens
- `normalization_mode="per_dim"` for image tokens (reduces but doesn't eliminate artifacts)
- `outlier_masking="zero"` or `"clamp"` for image tokens at layer -2 (layer -6 has no outliers)
- `alpha=1.0` possible with text tokens only (though quality loss vs pure text)

**Key Finding (2025-12-12):** Layer -6 is naturally cleaner than -2 or -8 for VL conditioning. The 617x outlier in dimension 396 only appears at layer -2.

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
