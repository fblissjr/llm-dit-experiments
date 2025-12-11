# Qwen3-VL Vision Conditioning for Z-Image

> **Last Updated:** 2025-12-11

Experiments using Qwen3-VL hidden states to condition Z-Image generation. This is a **zero-shot** approach that requires no training, but produces **out-of-distribution (OOD) embeddings** that must be interpolated with text to work.

## TL;DR

We can extract hidden states from Qwen3-VL's text model (after it processes an image) and blend them with text embeddings to influence Z-Image generation. This works because Qwen3-VL and Qwen3-4B (Z-Image's text encoder) share the same base architecture (`hidden_size=2560`). However, the VL embeddings are OOD relative to what Z-Image was trained on, so quality is limited without training an adapter.

**Bottom line:** Novel technique, useful for understanding embedding spaces, but a trained adapter (like IP-Adapter) would produce better results for practical use.

## Why This Works (And Why It's Limited)

### The Alignment Chain

```
Qwen3-VL training aligned:    vision tokens <---> VL text model
Z-Image training aligned:     Qwen3-4B instruct hidden states <---> DiT

NOT aligned:                  VL text model outputs <---> Qwen3-4B instruct outputs
```

The VL model aligned vision tokens to *its own* text model during training, but that doesn't mean VL outputs are identical to the base Qwen3-4B outputs that Z-Image expects. They're close (same architecture, same base weights) but not identical.

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
| **-2 (penultimate)** | **Best results** - matches Z-Image's extraction layer |
| -5 to -15 | Progressive "Asian bias" (Chinese training data influence) |
| -18 to -25 | "Semantic averaging" - outputs look like prototypes/category centroids |
| -30+ | Too abstract, loses all specificity |

The middle layer "Asian bias" likely reflects Qwen's Chinese training data distribution at the semantic prototype level.

### Embedding Statistics

| Source | std | Notes |
|--------|-----|-------|
| Qwen3-4B text embeddings | 58.75 | What Z-Image expects |
| VL text model hidden states | ~13 | 4-5x lower magnitude |
| VL vision encoder output | 0.57 | Completely incompatible |

Scaling VL embeddings to match text statistics helps but doesn't solve the fundamental OOD problem.

## Novelty Assessment

An arXiv literature search (2022-2025) confirmed this specific approach is **novel**:

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

None of these exploit zero-shot architectural compatibility between same-family models.

## What This Is Good For

1. **Understanding embedding spaces** - How VLM and text encoder representations relate
2. **Quick prototyping** - Test vision conditioning ideas without training
3. **Baseline for trained methods** - Compare "what zero-shot gets you" vs trained adapters
4. **Research finding** - Publishable as technical report on zero-shot cross-model transfer

## What This Is NOT Good For

1. **Production image conditioning** - Use IP-Adapter or similar trained methods
2. **Precise style transfer** - VL influence is unpredictable (some attributes transfer, others don't)
3. **Content preservation** - High alpha values override text semantics

## Recommended Next Steps

| Priority | Action | Why |
|----------|--------|-----|
| 1 | Write up as arXiv technical report | Establishes priority, documents the finding |
| 2 | Train minimal adapter (single linear layer) | Answers "how much training helps?" |
| 3 | Systematic layer sweep with metrics | Characterize what each layer contributes |
| 4 | Test on second model family | Validate generalization of the principle |
| - | Stop optimizing zero-shot blending tricks | Diminishing returns without training |

## Directory Structure

```
experiments/qwen3_vl/
  README.md                 # This file
  __init__.py               # Python module init
  scripts/                  # Executable scripts
    extract_embeddings.py   # Extract VL embeddings from images
    blend_and_generate.py   # Blend VL + text and generate
    run_comparison.py       # Systematic comparison sweeps
  docs/
    guides/
      conditioning_guide.md # Practical parameter guide
    research/
      research_findings.md  # Detailed experimental findings
      research_questions.md # Deep dive on techniques from literature
      research_index.md     # Quick reference
      related_work.md       # Prior art analysis
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

## References

### Our Research
- [docs/research/research_findings.md](docs/research/research_findings.md) - Detailed experimental results
- [docs/research/research_questions.md](docs/research/research_questions.md) - Literature review and technique analysis
- [docs/research/related_work.md](docs/research/related_work.md) - Prior art from other domains

### External
- [IP-Adapter](https://arxiv.org/abs/2308.06721) - Closest prior art (trained adapter approach)
- [UniFusion](https://arxiv.org/abs/2510.12789) - VLM as unified encoder (requires training)
- [MoMA](https://arxiv.org/abs/2404.05674) - MLLM adapter (requires training)
- [GILL](https://arxiv.org/abs/2305.17216) - LLM-to-diffusion mapping (requires training)
- [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) - Vision-Language Model
- [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) - DiT diffusion model
