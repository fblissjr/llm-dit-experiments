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

### 1. Outlier Dimension Masking (DONE)

**Hypothesis:** The 617x outlier in dimension 396 is the smoking gun for image token artifacts.

**Implementation:** Three masking modes available via `mask_outlier_dimensions()`:
- `zero`: Zero out outlier dimensions entirely
- `clamp`: Scale outlier dimensions to threshold level
- `scale`: Proportionally reduce outlier dimension values

**Status:** DONE (2025-12-12)

**New CLI options:**
- `--sweep outlier` - Test all masking modes
- `--outlier-masking {none,zero,clamp,scale}` - Specify masking modes
- `--outlier-threshold` - Std ratio threshold (default: 10.0)

**New functions in `src/llm_dit/vl/blending.py`:**
- `mask_outlier_dimensions(embeddings, threshold, mode)` - Apply masking
- `get_outlier_dimensions(embeddings, threshold)` - Analysis helper

```bash
# Run outlier masking sweep
uv run python experiments/qwen3_vl/scripts/run_comparison.py \
  -i experiments/inputs/test_scene.png \
  -p "Homer Simpson eating spaghetti" \
  --sweep outlier \
  -o experiments/results/outlier_masking_test

# Test specific masking modes
uv run python experiments/qwen3_vl/scripts/run_comparison.py \
  -i experiments/inputs/test_scene.png \
  -p "Homer Simpson eating spaghetti" \
  --token-modes full \
  --outlier-masking zero clamp scale \
  -o experiments/results/outlier_modes_test
```

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
| P0 | Outlier dimension masking | Low | High | **DONE** |
| P0 | VL vs Qwen3-4B comparison | None | High | **DONE** |
| P0 | Layer-by-token sweep | None | Medium | **DONE** |
| P1 | VL format config (think_block, system_prompt) | Medium | High | **DONE** |
| P1 | Style delta arithmetic | Medium | High | **DONE - FAILED** |
| P1 | AdaIN blending | Low | Medium | **DONE - PARTIAL** |
| P1 | img2img comparison vs VL | Low | High | **DONE** |
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

### 2025-12-12 (Session 6 - img2img, Style Delta, AdaIN)
- **img2img Support Added to CLI**
  - New flags in `scripts/generate.py`: `--img2img`, `--strength`
  - Fixed generator device handling (CUDA for img2img, CPU for txt2img)
  - Fixed flow matching noise addition in `src/llm_dit/pipelines/z_image.py`
  - Formula: `latents = (1 - sigma) * init_latents + sigma * noise` (replaced incorrect `scheduler.add_noise()`)

- **Outlier Masking Fix Applied**
  - Fixed to apply masking to image tokens ONLY before combining with text tokens
  - Previously masked the combined embeddings, which was incorrect
  - Layer -6 has NO outliers above 10x threshold (617x outlier is layer -2 specific)

- **Style Delta Arithmetic Implemented**
  - Added `compute_style_delta()` to `src/llm_dit/vl/blending.py`
  - Added `blend_with_style_delta()` for applying deltas
  - Created test script: `experiments/qwen3_vl/scripts/test_style_delta.py`
  - **Result: FAILED** - Even at alpha 0.3, completely destroys content (Homer becomes woman on beach, cyan ball)
  - Delta contains too much "content" information, not just style

- **AdaIN Blending Implemented**
  - Added `blend_adain()` (per-token) to `src/llm_dit/vl/blending.py`
  - Added `blend_adain_per_dim()` (per-dimension) for stronger style transfer
  - Created test script: `experiments/qwen3_vl/scripts/test_adain.py`
  - **Result: PARTIAL SUCCESS**
    - `per_token`: Preserves Homer perfectly but no visible style transfer
    - `per_dim`: Transfers colors (orange shirt from hexagon!) but corrupts subject (Homer becomes Bart)
  - Fundamental tradeoff discovered: style transfer strong enough to be visible also corrupts content

- **VL vs img2img Comparison**
  - VL blending: Superimposition (Homer merged with house)
  - img2img low strength (0.3-0.7): Preserves input structure, no new subjects
  - img2img high strength (0.9+): New subject appears but transforms original
  - Neither achieves true spatial composition ("Homer next to house")

- **Layer -6 Discovery**
  - Layer -6 produces crisper images than -2 or -8
  - Preserves text prompt content better (Homer appears correctly)
  - NO outliers at layer -6 (617x outlier is specific to layer -2)
  - VL fine-tuning overwrote later layers for vision tasks

- **Core Research Finding:**
  - Embedding space doesn't cleanly separate "style" from "content"
  - VL influence strong enough to transfer style also corrupts semantic content
  - This is a fundamental limitation of zero-shot approaches without training

- **Files Modified:**
  - `src/llm_dit/vl/blending.py` - Added style delta and AdaIN functions
  - `src/llm_dit/pipelines/z_image.py` - Fixed img2img noise formula
  - `scripts/generate.py` - Added img2img CLI flags
  - `experiments/qwen3_vl/scripts/test_style_delta.py` - New test script
  - `experiments/qwen3_vl/scripts/test_adain.py` - New test script

### 2025-12-12 (Session 5)
- **Outlier Dimension Masking Implemented**
  - Added `mask_outlier_dimensions()` to `src/llm_dit/vl/blending.py`
  - Added `get_outlier_dimensions()` analysis helper
  - Exported from `src/llm_dit/vl/__init__.py`

- **Three masking modes available:**
  | Mode | Description |
  |------|-------------|
  | `zero` | Zero out outlier dimensions entirely |
  | `clamp` | Scale dimensions to threshold level |
  | `scale` | Proportionally reduce dimension values |

- **Integrated into VLEmbeddingExtractor:**
  - New parameters: `outlier_masking`, `outlier_threshold`
  - Results tracked in `VLExtractionResult`: `masked_dimensions`, `masked_dim_ratios`

- **Experiment Runner Updated:**
  - New CLI: `--outlier-masking`, `--outlier-threshold`
  - New sweep preset: `--sweep outlier`
  - Metadata output includes masked dimension info

- **Key Files Modified:**
  - `src/llm_dit/vl/blending.py` - Core masking functions
  - `src/llm_dit/vl/qwen3_vl.py` - VLEmbeddingExtractor integration
  - `src/llm_dit/vl/__init__.py` - Exports
  - `experiments/qwen3_vl/scripts/run_comparison.py` - CLI and experiment runner

### 2025-12-12 (Session 4)
- **VL vs Qwen3-4B Comparison Completed**
  - Prompt: "Homer Simpson eating spaghetti"
  - Reference image: test_scene.png (cartoon house)
  - Results in `experiments/results/vl_vs_qwen3_comparison/`

- **Key Finding: Text Tokens Match Pure Qwen3-4B Quality**
  - VL text tokens at alpha=100%: Clean Homer Simpson, excellent prompt adherence
  - VL all tokens at alpha=100%: Homer sitting ON the cartoon house roof!
  - This confirms: text tokens carry semantic content, image tokens carry visual content

- **Statistics Captured**
  | Mode | Tokens | Original Std | Scale Factor |
  |------|--------|--------------|--------------|
  | Text only | 13 | 63.7 | 0.96x (minimal) |
  | All tokens | 271 | 18.2 | **3.35x** (significant!) |

- **Conclusions:**
  1. **For prompt adherence:** Use `text_tokens_only=True` - nearly identical to Qwen3-4B
  2. **For image-guided generation:** Use `full` token mode - image content bleeds through
  3. **The 0.999 correlation is validated** - VL text tokens really do match Qwen3-4B semantically
  4. **3.35x scaling** for all tokens likely causes some artifacts due to amplified noise

- **Layer-by-Token Sweep Completed**
  - Results in `experiments/results/layer_by_token_sweep/`
  - Tested layers -2, -8, -16, -24 with both text_only and full token modes

- **Layer Statistics (Text Tokens - all good):**
  | Layer | Std | Scale | Quality |
  |-------|-----|-------|---------|
  | -2 | 52.6 | 1.16x | Clean |
  | -8 | 63.7 | 0.96x | Clean (best scaling) |
  | -16 | 63.6 | 0.96x | Clean |
  | -24 | 62.5 | 0.98x | Clean |

- **Layer Statistics (All Tokens - middle layers better):**
  | Layer | Std | Scale | Quality |
  |-------|-----|-------|---------|
  | -2 | 13.4 | **4.56x** | More artifacts |
  | -8 | 18.2 | 3.35x | Clean blending |
  | -16 | 18.1 | 3.38x | Clean blending |
  | -24 | 13.7 | 4.44x | More artifacts |

- **Optimal Layer Recommendation: -8**
  - Text tokens: minimal scaling (0.96x)
  - All tokens: lower scaling than edge layers (3.35x vs 4.56x)
  - Edge layers (-2, -24) require 35% more scaling, amplifying noise

- **Prompts File Support Added to run_comparison.py**
  - New flags: `--prompts-file`, `--prompt-ids`, `--prompt-category`, `--prompt-difficulty`
  - Can run multiple prompts from `experiments/prompts/standard_prompts.yaml`
  - Creates subdirectories per prompt for organization
  - Saves `prompts_summary.json` with all results

### 2025-12-12 (Session 3)
- **VL Format Configuration Implemented**: Added configurable format options to VL extractor
  - `force_think_block`: Toggle think block injection (default True for Qwen3-4B compatibility)
  - `system_prompt`: Optional system prompt support
  - Updated `VLExtractionResult` to track format used and system_prompt
- **Enhanced Metadata**: metadata.json now includes:
  - Full formatted prompt with all special tokens for both text encoder and VL
  - Model information (Qwen3-4B for text, Qwen3-VL-4B-Instruct for VL)
  - Chat template format used (with_think_block or no_think_block)
- **Code Verified**: All modified files compile correctly

### 2025-12-12 (Session 2)
- **Infrastructure Alignment**: Consolidated experiment scripts to import from core
  - `blend_and_generate.py` now imports `blend_embeddings` from `llm_dit.vl.blending`
  - `run_comparison.py` imports blending from core module
  - Added `force_think_block` and `system_prompt` parameters to experiments
- **Official Format Verified**: Checked diffusers, DiffSynth-Studio, and Z-Image repo
  - All use `enable_thinking=True` = NO think block by default
  - All use `hidden_states[-2]` (penultimate layer)
  - No system prompt in official implementations
- **Validation Passed**: Before/after comparison shows consistent results

### 2025-12-12 (Session 1)
- Completed normalization mode comparison (global vs per_dim vs hybrid)
- Per-dim normalization clips outliers but doesn't eliminate artifacts
- Created this working document
- Next: Run VL vs Qwen3-4B comparison, implement outlier masking

---

## Quick Commands

```bash
# VALIDATED: Alpha sweep with text tokens (default format, no think block)
uv run python experiments/qwen3_vl/scripts/run_comparison.py \
  -i experiments/inputs/test_scene.png \
  -p "A simple cartoon house with a red roof" \
  --sweep alpha \
  -o experiments/results/alpha_sweep

# VALIDATED: Layer sweep
uv run python experiments/qwen3_vl/scripts/run_comparison.py \
  -i experiments/inputs/test_scene.png \
  -p "A simple cartoon house with a red roof" \
  --sweep layer \
  -o experiments/results/layer_sweep

# VALIDATED: Token mode sweep (text_only, image_only, etc.)
uv run python experiments/qwen3_vl/scripts/run_comparison.py \
  -i experiments/inputs/test_scene.png \
  -p "A simple cartoon house with a red roof" \
  --sweep token \
  -o experiments/results/token_sweep

# VALIDATED: Outlier masking sweep (none, zero, clamp, scale)
uv run python experiments/qwen3_vl/scripts/run_comparison.py \
  -i experiments/inputs/test_scene.png \
  -p "Homer Simpson eating spaghetti" \
  --sweep outlier \
  -o experiments/results/outlier_sweep

# Custom outlier masking modes
uv run python experiments/qwen3_vl/scripts/run_comparison.py \
  -i experiments/inputs/test_scene.png \
  -p "Homer Simpson eating spaghetti" \
  --token-modes full \
  --outlier-masking zero clamp \
  --outlier-threshold 10.0 \
  -o experiments/results/outlier_custom

# VALIDATED: Custom alphas with specific layers
uv run python experiments/qwen3_vl/scripts/run_comparison.py \
  -i experiments/inputs/test_scene.png \
  -p "Homer Simpson eating spaghetti" \
  --alphas 0.0 0.3 1.0 \
  --layers -2 -8 \
  -o experiments/results/custom_sweep

# With think block (to test format impact)
uv run python experiments/qwen3_vl/scripts/run_comparison.py \
  -i experiments/inputs/test_scene.png \
  -p "A simple cartoon house with a red roof" \
  --alphas 0.0 0.3 1.0 \
  --force-think-block \
  -o experiments/results/with_think_block

# With system prompt (to test format impact)
uv run python experiments/qwen3_vl/scripts/run_comparison.py \
  -i experiments/inputs/test_scene.png \
  -p "A simple cartoon house with a red roof" \
  --alphas 0.0 0.3 1.0 \
  --system-prompt "You are an expert artist." \
  -o experiments/results/with_system_prompt

# Full format options (think block + system prompt)
uv run python experiments/qwen3_vl/scripts/run_comparison.py \
  -i experiments/inputs/test_scene.png \
  -p "A simple cartoon house with a red roof" \
  --alphas 0.0 0.3 1.0 \
  --force-think-block \
  --system-prompt "You are an expert artist." \
  -o experiments/results/full_format

# Validate infrastructure (after any changes)
uv run python experiments/qwen3_vl/scripts/validate_infrastructure.py

# ============================================================
# PROMPTS FILE SUPPORT (NEW)
# ============================================================

# Run specific prompts by ID
uv run python experiments/qwen3_vl/scripts/run_comparison.py \
  -i experiments/inputs/test_scene.png \
  --prompt-ids animal_001,simple_002 \
  --alphas 0.3 1.0 \
  -o experiments/results/multi_prompt_test

# Run all prompts in a category
uv run python experiments/qwen3_vl/scripts/run_comparison.py \
  -i experiments/inputs/test_scene.png \
  --prompt-category animals \
  --sweep alpha \
  -o experiments/results/animals_alpha_sweep

# Run prompts by difficulty
uv run python experiments/qwen3_vl/scripts/run_comparison.py \
  -i experiments/inputs/test_scene.png \
  --prompt-difficulty easy \
  --alphas 0.3 1.0 \
  -o experiments/results/easy_prompts

# Use custom prompts file
uv run python experiments/qwen3_vl/scripts/run_comparison.py \
  -i experiments/inputs/test_scene.png \
  --prompts-file my_custom_prompts.yaml \
  --prompt-ids custom_001 \
  -o experiments/results/custom_prompts
```
