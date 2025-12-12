# Qwen3-VL Experiments: Working Document

> **Last Updated:** 2025-12-12
> **Purpose:** Single source of truth for experiment planning, priorities, and open questions

This document tracks what experiments need to be run, modified, or created for the Qwen3-VL vision conditioning research. It serves as the working task list for Claude agents and human researchers.

---

## Current State Summary

**What we know:**
- VL text tokens have 0.999 correlation with Qwen3-4B per-dimension statistics
- VL image tokens have only 0.737 correlation with extreme outliers (dim 396: 617x, dim 4: 42x)
- Per-dimension normalization reduces but doesn't eliminate artifacts
- Text tokens produce fewer artifacts than image tokens
- Layer -8 appears to work better than layer -2 for VL embeddings

**What we don't know:**
- Why artifacts persist despite 0.999 correlation for text tokens
- Whether dim 396 (617x outlier) is the primary cause of image token artifacts
- Whether any zero-shot approach can match pure Qwen3-4B quality

---

## P0: Run Now (Infrastructure Ready)

These experiments can be run immediately with existing code.

### 1. Outlier Dimension Masking

**Hypothesis:** The 617x outlier in dimension 396 is the smoking gun for image token artifacts.

**Implementation:** Zero out or clamp dimensions with >10x std ratio before generation.

```bash
# Requires adding to blending.py:
# def mask_outlier_dimensions(vl_emb, qwen3_stats, threshold=10.0):
#     ratios = vl_emb.std(dim=0) / qwen3_stats['per_dim_std']
#     mask = ratios < threshold
#     return vl_emb * mask.float()

# Then run comparison with masked vs unmasked image tokens
uv run experiments/qwen3_vl/scripts/run_comparison.py \
  --image experiments/inputs/test_scene.png \
  --prompt "A simple cartoon house with a red roof" \
  --experiment outlier_masking \
  --output-dir experiments/results/outlier_masking
```

**Status:** Needs `mask_outlier_dimensions()` function added to `src/llm_dit/vl/blending.py`

**Expected outcome:** If dim 396 is the culprit, masking it should dramatically improve image token quality.

---

### 2. VL-Only vs Qwen3-4B Direct Comparison

**Hypothesis:** VL text tokens at alpha=1.0 might match Qwen3-4B quality for prompt adherence.

**Why this matters:** If true, we can eliminate Qwen3-4B from the pipeline entirely.

```bash
uv run experiments/qwen3_vl/scripts/run_comparison.py \
  --image experiments/inputs/test_scene.png \
  --prompt "Homer Simpson eating spaghetti" \
  --experiment vl_only_vs_qwen3 \
  --output-dir experiments/results/vl_vs_qwen3
```

**Configurations to compare:**
| Config | Alpha | Token Mode | Expected Result |
|--------|-------|------------|-----------------|
| Pure Qwen3-4B | 0.0 | N/A | Baseline quality |
| VL text tokens only | 1.0 | text_only | Should match baseline if 0.999 correlation holds |
| VL full sequence | 1.0 | full | Likely artifacts from image tokens |
| Traditional blend | 0.3 | full | Current "best" practice |

**Status:** Experiment type exists in `run_comparison.py`, ready to run

---

### 3. Layer-by-Token-Type Sweep

**Hypothesis:** Image tokens and text tokens might benefit from different hidden layers.

```bash
# Quick version (subset of layers)
uv run experiments/qwen3_vl/scripts/run_comparison.py \
  --image experiments/inputs/test_scene.png \
  --prompt "A simple cartoon house with a red roof" \
  --experiment vl_layer_by_token \
  --quick \
  --output-dir experiments/results/layer_by_token

# Full version (all layers)
uv run experiments/qwen3_vl/scripts/run_comparison.py \
  --image experiments/inputs/test_scene.png \
  --prompt "A simple cartoon house with a red roof" \
  --experiment vl_layer_by_token \
  --output-dir experiments/results/layer_by_token_full
```

**Status:** Ready to run

---

### 4. Normalization Mode Comparison (DONE)

**Status:** Completed on 2025-12-12

**Results:** `experiments/results/vl_normalization_test/`
- Global normalization: Oversaturated, strong style transfer
- Per-dim normalization: More photorealistic, lost some style
- Hybrid normalization: Balanced between the two

**Finding:** Per-dim normalization clips outliers (max 40240 -> 9182) but doesn't eliminate artifacts.

---

## P1: Modify Then Run

These experiments require code modifications before running.

### 5. Style Delta Arithmetic

**Hypothesis:** `result = text_emb + alpha * (VL_style - VL_neutral)` isolates style without content override.

**Required modifications to `src/llm_dit/vl/blending.py`:**

```python
def compute_style_delta(
    style_vl_embeddings: torch.Tensor,
    neutral_vl_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Extract style-only component by subtracting neutral baseline."""
    return style_vl_embeddings - neutral_vl_embeddings

def blend_with_style_delta(
    text_embeddings: torch.Tensor,
    style_delta: torch.Tensor,
    alpha: float = 0.3,
) -> torch.Tensor:
    """Add style delta to text embeddings."""
    # Handle sequence length mismatch
    if style_delta.shape[1] != text_embeddings.shape[1]:
        style_delta = interpolate_sequence(style_delta, text_embeddings.shape[1])
    return text_embeddings + alpha * style_delta
```

**Experiment workflow:**
1. Extract VL embeddings from neutral image (solid gray or white)
2. Extract VL embeddings from style image (same text prompt as neutral!)
3. Compute delta
4. Add scaled delta to text embeddings
5. Generate

**Status:** Needs implementation in blending.py and new experiment type in run_comparison.py

---

### 6. AdaIN-Style Blending

**Hypothesis:** Matching per-token mean/std (AdaIN) preserves content better than linear interpolation.

**Required modifications to `src/llm_dit/vl/blending.py`:**

```python
def blend_adain(
    text_embeddings: torch.Tensor,
    vl_embeddings: torch.Tensor,
    alpha: float = 0.3,
) -> torch.Tensor:
    """Apply AdaIN-style normalization: transfer VL statistics to text content."""
    # Normalize text embeddings
    t_mean = text_embeddings.mean(dim=-1, keepdim=True)
    t_std = text_embeddings.std(dim=-1, keepdim=True)
    normalized = (text_embeddings - t_mean) / (t_std + 1e-5)

    # Apply VL statistics
    v_mean = vl_embeddings.mean(dim=-1, keepdim=True)
    v_std = vl_embeddings.std(dim=-1, keepdim=True)
    restylized = normalized * v_std + v_mean

    # Blend
    return alpha * restylized + (1 - alpha) * text_embeddings
```

**Status:** Needs implementation and experiment integration

---

### 7. Intra-VL Token Blending

**Hypothesis:** Blend image tokens (style) with text tokens (content) from the SAME VL extraction.

**Required infrastructure:**
1. Extract image token subset from VL result
2. Extract text token subset from VL result
3. Blend them at specified ratio BEFORE passing to Z-Image

**New parameter needed:**
```python
vl_intra_blend_ratio: float  # 0.0 = all text tokens, 1.0 = all image tokens
```

**Status:** Needs new infrastructure in `VLEmbeddingExtractor`

---

## P2: New Experiment Variations

These are new experiment types to create.

### 8. Cross-Normalization Test

**Question:** What if we normalize VL embeddings using Qwen3-4B's per-dim stats, but then scale to a different target std?

**Variations:**
- Per-dim normalize, then scale to std=61 (Qwen3-4B default)
- Per-dim normalize, then scale to std=70 (higher energy)
- Per-dim normalize, then scale to std=50 (lower energy)

**Status:** Needs new experiment configuration

---

### 9. Timestep-Dependent Alpha

**Hypothesis:** Based on Prompt-to-Prompt research, early diffusion steps set structure, late steps set details.

**Approach:**
- High VL alpha at early timesteps (t=1.0 to 0.7) for structure
- Low VL alpha at late timesteps (t=0.3 to 0.0) for text-guided details

**Required modifications:** Changes to generation loop in pipeline, not just blending

**Status:** Requires pipeline-level changes

---

### 10. Multi-Reference Blending

**Question:** Can we blend VL embeddings from multiple reference images?

**Use case:** Combine style from image A with composition from image B

**Status:** Theoretical - needs design

---

## P3: Open Questions

### Root Cause Questions

1. **Why do artifacts persist with 0.999 correlation?**
   - The per-dimension correlation is nearly perfect, yet visible artifacts remain
   - Hypothesis: Correlation measures distribution shape, not absolute values
   - Hypothesis: Remaining 0.001 difference is in critical dimensions

2. **Is dimension 396 the smoking gun?**
   - It has 617x std ratio - by far the largest outlier
   - Zeroing just this dimension might dramatically improve results
   - Need to test masking experiments

3. **Are the artifacts from RoPE mismatch or embedding content?**
   - Qwen3-VL uses MRoPE with different theta (5M vs 1M)
   - Text tokens have high correlation despite RoPE difference
   - Image tokens have low correlation - is this RoPE or content?

### Approach Questions

4. **Is training required for quality results?**
   - Even with optimal normalization, artifacts persist
   - A small projection layer (1-2 linear layers) might bridge the gap
   - But that breaks "zero-shot" claim

5. **Should we focus on text tokens only?**
   - Text tokens have 0.999 correlation
   - Image tokens have severe outliers
   - Maybe the answer is: use VL text tokens for prompt adherence, ignore image tokens

6. **Is the VL model even necessary?**
   - VL text tokens at alpha=1.0 might match pure Qwen3-4B
   - If so, what's the point of vision conditioning?
   - Value might only be in extracting style from images, not general generation

### Infrastructure Questions

7. **How do we measure "artifact severity"?**
   - Visual inspection is subjective
   - Need quantitative metrics (grid pattern detection, color histogram analysis)
   - ImageReward/SigLIP scores might not capture artifact issues

8. **Should we save intermediate embeddings for analysis?**
   - Currently we generate images directly
   - Saving embeddings at each stage would help debug

---

## Priority Matrix

| Priority | Experiment | Effort | Impact | Status |
|----------|------------|--------|--------|--------|
| P0 | Outlier dimension masking | Low | High | Needs function |
| P0 | VL vs Qwen3-4B comparison | None | High | Ready |
| P0 | Layer-by-token sweep | None | Medium | Ready |
| P1 | Style delta arithmetic | Medium | High | Needs impl |
| P1 | AdaIN blending | Low | Medium | Needs impl |
| P1 | Intra-VL blending | High | High | Needs infra |
| P2 | Cross-normalization | Low | Low | Needs config |
| P2 | Timestep-dependent alpha | High | Medium | Needs pipeline |
| P3 | Multi-reference blend | Medium | Unknown | Theoretical |

---

## Reference Files

### Code
- `src/llm_dit/vl/qwen3_vl.py` - VLEmbeddingExtractor class
- `src/llm_dit/vl/blending.py` - Normalization and blending functions
- `src/llm_dit/vl/qwen3_4b_stats.npz` - Reference statistics for normalization
- `experiments/qwen3_vl/scripts/run_comparison.py` - Experiment runner

### Documentation
- `experiments/qwen3_vl/README.md` - Main entry point
- `experiments/qwen3_vl/docs/research/findings.md` - Research findings
- `experiments/qwen3_vl/docs/research/quick_reference.md` - Technique reference
- `experiments/qwen3_vl/docs/experiments/token_position.md` - Token position experiments

### Results
- `experiments/results/vl_normalization_test/` - Normalization comparison results
- `experiments/results/` - All experiment outputs

---

## Session Log

### 2025-12-12
- Completed normalization mode comparison (global vs per_dim vs hybrid)
- Per-dim normalization clips outliers but doesn't eliminate artifacts
- Created this working document
- Next: Run VL vs Qwen3-4B comparison, implement outlier masking

---

## Quick Commands

```bash
# Run VL vs Qwen3-4B comparison (P0, ready now)
uv run experiments/qwen3_vl/scripts/run_comparison.py \
  --image experiments/inputs/test_scene.png \
  --prompt "Homer Simpson eating spaghetti" \
  --experiment vl_only_vs_qwen3 \
  --output-dir experiments/results/vl_vs_qwen3

# Run layer sweep (P0, ready now)
uv run experiments/qwen3_vl/scripts/run_comparison.py \
  --image experiments/inputs/test_scene.png \
  --prompt "A simple cartoon house with a red roof" \
  --experiment vl_layer_by_token \
  --quick \
  --output-dir experiments/results/layer_by_token

# Check embedding statistics
uv run experiments/qwen3_vl/scripts/compare_embedding_stats.py \
  --image experiments/inputs/test_scene.png
```
