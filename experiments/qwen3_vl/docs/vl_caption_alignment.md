# VL Caption Alignment: A Novel Approach to Vision Conditioning

> **Last Updated:** 2025-12-15 (Updated with sweep results)

## Executive Summary

This document describes a method for vision conditioning in Z-Image that uses **caption-aligned VL embeddings**. Unlike previous VL experiments that used mismatched text/image pairs (e.g., "Homer Simpson" text with anime girl image), this approach:

1. Generates a detailed caption of the input image using Qwen3-VL-4B-Thinking
2. Extracts VL embeddings with BOTH the generated caption AND the original image
3. Blends these "aligned" VL embeddings with text encoder embeddings of the same caption
4. Generates with Z-Image using the blended embeddings

The hypothesis is that aligning the text description with the actual image content produces richer, more coherent embeddings for image generation than mismatched text/image conditioning.

## Data Flow Architecture

```
                           INPUT IMAGE
                               |
                               v
        +----------------------+----------------------+
        |                      |                      |
        v                      v                      v
+---------------+    +-----------------+    +-----------------+
| Qwen3-VL-4B   |    | Qwen3-VL-4B     |    | Qwen3-VL-4B     |
| Thinking      |    | Thinking        |    | Thinking        |
| (CAPTIONING)  |    | (VL EXTRACTION) |    | (VL EXTRACTION) |
+---------------+    | Aligned         |    | Generic         |
        |            +-----------------+    +-----------------+
        |                   |                      |
        v                   v                      v
+---------------+    +-----------------+    +-----------------+
| Detailed      |    | VL Embeddings   |    | VL Embeddings   |
| Caption       |    | (caption+image) |    | (generic+image) |
| ~1024+ tokens |    | ~2700 tokens    |    | ~1039 tokens    |
+---------------+    +-----------------+    +-----------------+
        |                   |                      |
        |                   |                      |
        v                   v                      v
+---------------+    +----------------------------------------------+
| Qwen3-4B      |    |                BLENDING                       |
| Text Encoder  |    |  blended = alpha * VL + (1-alpha) * text     |
| (Z-Image)     |    |                                              |
+---------------+    +----------------------------------------------+
        |                   |
        v                   v
+---------------+    +-----------------+
| Text          |    | Blended         |
| Embeddings    |    | Embeddings      |
| ~1504 tokens  |--->| (interpolated   |
+---------------+    | to 1504 max)    |
                     +-----------------+
                            |
                            v
                     +-----------------+
                     | Z-Image DiT     |
                     | (Generation)    |
                     +-----------------+
                            |
                            v
                     +-----------------+
                     | Output Image    |
                     +-----------------+
```

### Token Composition Breakdown

The VL extraction produces embeddings with multiple token types:

```
VL ALIGNED EXTRACTION (~2732 tokens for 1696-token caption):
+------------------+------------------------+------------------+
| System/Template  | Image Tokens           | Caption Tokens   |
| Tokens (~10)     | (~1026 for 1024x1024)  | (~1696)          |
+------------------+------------------------+------------------+
                   |<----- Vision Info ---->|<--- Text Info -->|

VL GENERIC EXTRACTION (~1039 tokens):
+------------------+------------------------+------------------+
| System/Template  | Image Tokens           | Generic Text     |
| Tokens (~10)     | (~1026 for 1024x1024)  | (~13 tokens)     |
+------------------+------------------------+------------------+
```

**Key Insight**: The aligned extraction creates ~2732 tokens where:
- ~1026 tokens encode the image visually
- ~1696 tokens encode a detailed semantic description of the same image
- These tokens have been processed together through transformer self-attention

This contrasts with the generic extraction where the text tokens provide minimal semantic guidance ("Describe this image" = ~13 tokens).

## Hypothesis: Why Aligned Captions Should Work Better

### The Alignment Problem in Previous VL Experiments

Previous VL experiments failed because of **semantic mismatch**:

| Experiment | Text Input | Image Input | Result |
|------------|------------|-------------|--------|
| Style Transfer | "Homer Simpson" | Anime girl | Content destroyed, only partial style transfer |
| VL + img2img | "Your prompt" | Reference image | VL overpowered text semantics |

The fundamental issue: when text and image describe different content, the model receives conflicting signals. The VL embeddings encode visual features from one scene while text encodes semantics from another.

### How Caption Alignment Addresses This

With aligned captions:

1. **Consistent Semantics**: Both image tokens and text tokens describe the SAME content
2. **Mutual Reinforcement**: Image tokens provide visual grounding, caption tokens provide semantic structure
3. **Cross-Modal Attention**: During VL forward pass, image tokens attend to relevant caption tokens and vice versa
4. **Richer Representation**: The combined embedding should encode both visual detail AND linguistic structure

**Theoretical Basis**: In Qwen3-VL's architecture, the image tokens are projected into the same embedding space as text tokens via `PatchMerger.linear_fc2`. When both modalities describe the same content, the attention mechanism creates coherent cross-modal representations rather than conflicting ones.

### Expected Embedding Properties

| Property | VL Aligned | VL Generic | Pure Text |
|----------|------------|------------|-----------|
| Token count | High (~2700) | Medium (~1039) | Low-Medium |
| Visual detail | High (image tokens) | High (image tokens) | None |
| Semantic structure | High (long caption) | Low (generic prompt) | High (caption) |
| Cross-modal coherence | High (aligned) | Low (mismatched) | N/A |
| After interpolation | Preserves both | Loses text structure | N/A |

## Current Implementation

### Script: `vl_caption_embedding_test.py`

Location: `experiments/qwen3_vl/scripts/vl_caption_embedding_test.py`

**Workflow:**

```python
# Phase 1: Caption Generation
caption = vl_extractor.generate(
    image=input_image,
    system_prompt=detailed_captioning_prompt,
    max_new_tokens=2500,  # Target ~1024 tokens after thinking
    temperature=0.6,
    top_p=0.95,
    do_sample=True,
)

# Phase 2: VL Extraction (Aligned)
vl_result_aligned = vl_extractor.extract(
    image=input_image,
    text=caption,  # The generated caption
    hidden_layer=-6,
    text_tokens_only=False,  # Include image tokens
    scale_to_text=True,
    normalization_mode="global",
)

# Phase 3: VL Extraction (Generic - for comparison)
vl_result_generic = vl_extractor.extract(
    image=input_image,
    text="Describe this image",  # Generic prompt
    hidden_layer=-6,
    text_tokens_only=False,
    scale_to_text=True,
    normalization_mode="global",
)

# Phase 4: Text Encoding (Caption only)
text_result = text_encoder.encode(caption)

# Phase 5: Blending
blended_aligned = blend_interpolate(vl_aligned, text_emb, alpha=0.5)
blended_generic = blend_interpolate(vl_generic, text_emb, alpha=0.5)

# Phase 6: Generation (3 variants)
# 1. VL Aligned + Text (blended)
# 2. VL Generic + Text (blended)
# 3. Text Only (pure caption)
```

### Initial Results (2025-12-15)

From experiment `20251215_155328`:

| Condition | Token Count | Original Std | Scaled Std |
|-----------|-------------|--------------|------------|
| VL Aligned | 2732 | 9.125 | 61.0 |
| VL Generic | 1039 | 12.44 | 60.75 |
| Text Only | ~1504 | ~14.3 | ~14.3 |

**Observations:**

1. VL Aligned has 2.6x more tokens than VL Generic
2. Both VL extractions scale to ~61 std (matching Qwen3-4B target)
3. The aligned extraction has lower original std (9.125 vs 12.44), suggesting different embedding distribution

### Parameter Sweep Results (2025-12-15)

A comprehensive sweep was conducted testing:
- **Hidden Layers:** -2, -3, -4, -5, -6
- **Alphas:** 0.0, 0.25, 0.5, 0.75, 1.0
- **Methods:** Caption-aligned VL, Generic VL

Results saved to: `experiments/results/caption_alignment_sweep/20251215_163756/`

#### Key Findings

**1. Layer Selection**

| Layer | Quality at Low Alpha (0.0-0.25) | Quality at Mid Alpha (0.5) | Quality at High Alpha (0.75-1.0) |
|-------|--------------------------------|---------------------------|----------------------------------|
| -2 | Good | Good | Some artifacts, less stable |
| -3 | Good | Good | Good |
| -4 | Good | Good | Good, clean |
| -5 | Good | Good | Good, clean |
| -6 | Good | Good | **Most stable, cleanest** |

**Takeaway:** Layer -6 remains the best choice for VL extraction, consistent with previous research. Layer -2 shows occasional instability at high alphas.

**2. Alpha Blend Effects**

| Alpha | Effect (Caption Method) | Effect (Generic Method) |
|-------|------------------------|------------------------|
| 0.0 | Pure text - varied interpretations of caption | Pure text - same as caption |
| 0.25 | Slight VL influence, more grounded | Slight VL influence |
| 0.5 | **Balanced blend**, good quality | Balanced blend |
| 0.75 | Strong VL influence, converging to reference | Strong VL, more conservative |
| 1.0 | Near-reconstruction of input style | Near-reconstruction |

**Takeaway:** Alpha 0.5 provides the best balance. Higher alphas (0.75-1.0) cause both methods to converge toward similar outputs.

**3. Caption vs Generic Comparison**

| Aspect | Caption-Aligned | Generic |
|--------|-----------------|---------|
| Output diversity | **Higher** - more varied backgrounds, poses | Lower - more conservative |
| Color richness | **More vibrant** - varied palettes | More uniform |
| Style consistency | Varied across layers | More consistent |
| Quality at alpha=1.0 | Equivalent | Equivalent |

**Key Insight:** At low-to-mid alphas (0.0-0.5), caption-aligned produces more diverse and artistically varied results. At high alphas (0.75-1.0), both methods converge to similar VL-dominated outputs.

**4. Visual Quality Observations**

From the grid comparisons:

- **Best combinations:** Layer -5 or -6 with alpha 0.25-0.5 using caption-aligned method
- **Most stable:** Layer -6 across all alphas
- **Most diverse:** Alpha 0.25-0.5 with caption-aligned method
- **Reconstruction mode:** Alpha 1.0 at any layer (loses text variation)

**5. Embedding Statistics**

| Layer | Alpha | Std (Caption) | Std (Generic) | Notes |
|-------|-------|---------------|---------------|-------|
| -6 | 0.5 | ~33.0 | ~33.0 | Balanced blend |
| -6 | 1.0 | ~61.0 | ~60.75 | Pure VL |
| -2 | 0.5 | Similar | Similar | |

The standard deviation increases linearly with alpha, as expected from the interpolation formula.

#### Practical Recommendations

**For diverse creative outputs:**
```
--vl-hidden-layer -6
--vl-alpha 0.25 to 0.5
--method caption-aligned
```

**For faithful style transfer:**
```
--vl-hidden-layer -6
--vl-alpha 0.75 to 1.0
--method either (converge at high alpha)
```

**For maximum stability:**
```
--vl-hidden-layer -6
--vl-alpha 0.5
--method generic
```

### Next Steps

Based on sweep results, the following experiments are prioritized:

1. **Caption length study:** Does shorter caption (512 tokens) produce different results than long caption (1024+)?
2. **Cross-image validation:** Test on photorealistic images, not just anime
3. **Blend mode comparison:** Test adain/adain_per_dim blending with caption-aligned extraction
4. **Text-tokens-only ablation:** With aligned caption, do we need image tokens at all?
5. **Quality metrics:** Add SSIM/LPIPS for quantitative comparison

## Hyperparameter Space

### Primary Hyperparameters

| Parameter | Range | Current | Notes |
|-----------|-------|---------|-------|
| `vl_hidden_layer` | -2 to -8 | -6 | Layer -6 works best for VL (see previous research) |
| `vl_alpha` | 0.0 to 1.0 | 0.5 | Controls VL vs text blend |
| `caption_tokens` | 256 to 1504 | 1024 | Target caption length |
| `normalization_mode` | global/per_dim/hybrid | global | How to normalize VL embeddings |
| `text_tokens_only` | true/false | false | Whether to include image tokens |

### Hidden Layer Effects

| Layer | Expected Effect | Rationale |
|-------|-----------------|-----------|
| -2 | More abstract, task-specific | Default for text, but has outliers for VL image tokens |
| -3 | Slightly less abstract | |
| -4 | Good visual detail | |
| -5 | Transition zone | |
| -6 | **Recommended** | Cleaner for VL, no outliers, good semantic content |
| -8 | More concrete features | May lose some abstraction |

### VL Alpha Effects

| Alpha | Expected Effect |
|-------|-----------------|
| 0.0 | Pure text generation (baseline) |
| 0.2-0.3 | Slight VL influence (style hints) |
| 0.4-0.5 | Balanced blend (current default) |
| 0.6-0.7 | VL dominant (strong style/content from image) |
| 0.8-1.0 | Near-pure VL (reconstruction mode) |

### Caption Length Effects

| Length | Expected Effect |
|--------|-----------------|
| 256 tokens | Minimal description, may miss details |
| 512 tokens | Short but complete description |
| 1024 tokens | Detailed description (current default) |
| 1504 tokens | Maximum before interpolation needed |
| 2000+ tokens | Requires interpolation, compression artifacts |

### Token Mode Effects

| Mode | Token Count | Visual Info | Semantic Info |
|------|-------------|-------------|---------------|
| `text_tokens_only=True` | ~1696 | None | High |
| `text_tokens_only=False` | ~2732 | High (~1026 tokens) | High (~1696 tokens) |
| `image_tokens_only=True` | ~1026 | High | Low |

### Normalization Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `global` | Scale entire embedding by global std ratio | Text tokens, aligned extraction |
| `per_dim` | Per-dimension normalization to match Qwen3-4B stats | Image tokens with outliers |
| `hybrid` | 50/50 blend of global and per-dim | Mixed token types |

### Blend Modes

| Mode | Formula | Best For |
|------|---------|----------|
| `interpolate` | `alpha * VL + (1-alpha) * text` | Default, preserves both |
| `adain` | Transfer VL statistics to text content | Style transfer |
| `adain_per_dim` | Per-dimension statistic transfer | Stronger style, may corrupt |
| `style_delta` | `text + alpha * (VL_styled - VL_neutral)` | Isolated style (experimental) |

## Experiment Plan

### Phase 1: Baseline Characterization

**Goal:** Establish baseline quality for each condition.

```bash
# Run with current defaults
uv run experiments/qwen3_vl/scripts/vl_caption_embedding_test.py \
    -i experiments/inputs/test_photo.jpg \
    -o experiments/results/caption_baseline \
    --caption-tokens 1024 \
    --vl-alpha 0.5 \
    --vl-hidden-layer -6 \
    --steps 9 \
    --seed 42
```

**Metrics to collect:**
- SSIM between input and each output
- LPIPS perceptual similarity
- Manual quality assessment
- Generation time

### Phase 2: Alpha Sweep

**Goal:** Find optimal VL alpha for aligned captions.

| Alpha | Description |
|-------|-------------|
| 0.0 | Pure text baseline |
| 0.2 | Light VL influence |
| 0.3 | Previous best for style transfer |
| 0.4 | Balanced |
| 0.5 | Current default |
| 0.6 | VL dominant |
| 0.7 | Strong VL |
| 0.8 | Near-pure VL |
| 1.0 | Pure VL |

**Hypothesis:** Aligned captions may allow higher alpha values without content corruption because text and image are describing the same content.

### Phase 3: Hidden Layer Sweep

**Goal:** Verify layer -6 is optimal for aligned extraction.

| Layer | Priority |
|-------|----------|
| -2 | High (default for text) |
| -4 | Medium |
| -6 | High (current VL default) |
| -8 | Medium |

### Phase 4: Caption Length Study

**Goal:** Find optimal caption length.

| Target Length | Notes |
|---------------|-------|
| 256 | Minimal |
| 512 | Short |
| 768 | Medium |
| 1024 | Current default |
| 1280 | Long |
| 1504 | Maximum native |

### Phase 5: Token Mode Comparison

**Goal:** Compare text_tokens_only vs full extraction for aligned captions.

**Key question:** With aligned captions, is there benefit to including image tokens? The caption already describes the image semantically; do we need the visual tokens too?

| Mode | Hypothesis |
|------|------------|
| `text_tokens_only=True` | May be sufficient since caption describes image |
| `text_tokens_only=False` | May add visual detail not captured in caption |

### Phase 6: Normalization Mode Study

**Goal:** Determine if per-dim normalization helps aligned extraction.

Previous research showed layer -6 has no outliers, but aligned extraction may behave differently due to different token distribution.

### Phase 7: Cross-Image Validation

**Goal:** Verify method works across image types.

Test images:
- Photorealistic portrait
- Anime/illustration
- Landscape
- Abstract art
- Text-heavy image
- Complex scene with multiple subjects

## Comparison to Previous Approaches

### Previous: VL Style Transfer (Mismatched)

```
Input: "Homer Simpson"  +  Anime girl image
       (text prompt)       (VL reference)

Result: Style partially transferred, content destroyed
```

**Problem:** The VL embeddings encoded anime girl features, which conflicted with "Homer Simpson" semantics.

### Previous: VL + img2img

```
Input: User prompt  +  Reference image (img2img latent + VL embeddings)

Result: VL overpowered text, img2img provided structure
```

**Problem:** VL influence at any significant alpha disrupted text semantics.

### Current: Caption Alignment

```
Input: Image -> Caption -> VL(caption + image) + Text(caption)

Result: (To be measured)
```

**Expected Improvement:** No semantic conflict because caption describes the actual image.

## Technical Considerations

### Memory Management

The workflow requires sequential loading due to VRAM constraints:

```
1. Load Qwen3-VL -> Generate caption -> Extract VL embeddings
2. Unload Qwen3-VL
3. Load Z-Image pipeline -> Encode caption -> Blend -> Generate
```

This is implemented in the test script with explicit `vl_extractor.unload()` call.

### Interpolation for Long Sequences

VL aligned extraction produces ~2732 tokens, but Z-Image DiT has a 1504 token limit. The blending function uses linear interpolation:

```python
if blended_aligned.shape[0] > MAX_TOKENS:
    blended_aligned = torch.nn.functional.interpolate(
        blended_aligned.T.unsqueeze(0),
        size=MAX_TOKENS,
        mode="linear",
        align_corners=False,
    ).squeeze(0).T
```

**Concern:** Interpolation from 2732 to 1504 is ~1.8x compression. Quality impact needs measurement.

### Thinking Block Handling

Qwen3-VL-4B-Thinking generates responses with `<think>...</think>` blocks. The script parses these out:

```python
def parse_thinking_content(text: str) -> tuple[str, str | None]:
    match = re.search(r"<think>\s*(.*?)\s*</think>\s*", text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        caption = text[match.end():].strip()
        return caption, thinking
    return text.strip(), None
```

The thinking content is saved separately for analysis but not used in VL extraction.

## Future Directions

### Short-Term (No Training)

1. **Iterative refinement:** Generate -> caption -> regenerate -> caption (measure convergence)
2. **Multi-scale captions:** Blend short (style) and long (detail) caption extractions
3. **Attention analysis:** Visualize cross-attention between image and caption tokens

### Medium-Term (Light Training)

1. **Optimal caption format:** Fine-tune captioning prompt for Z-Image compatibility
2. **Projection layer:** Small MLP to bridge VL and text embedding spaces
3. **Compression-aware extraction:** Train to produce embeddings that compress well

### Long-Term (Research)

1. **Joint VL-DiT training:** Co-train VL projection with DiT
2. **Caption-free alignment:** Learn alignment without explicit captioning
3. **Architecture-agnostic method:** Generalize to other VL/diffusion pairs

## Usage

### Basic Usage

```bash
uv run experiments/qwen3_vl/scripts/vl_caption_embedding_test.py \
    -i experiments/inputs/your_image.png \
    -o experiments/results/caption_test \
    --caption-tokens 1024 \
    --steps 9
```

### Parameter Sweep

For comprehensive testing across layers and alphas:

```bash
uv run experiments/qwen3_vl/scripts/sweep_caption_alignment.py \
    -i experiments/inputs/your_image.png \
    --hidden-layers="-2,-3,-4,-5,-6" \
    --alphas 0.0 0.25 0.5 0.75 1.0 \
    --steps 9 \
    --seed 42 \
    -o experiments/results/caption_alignment_sweep
```

**Output grids:**
- `grid_method_caption.png` - All layers/alphas for caption-aligned method
- `grid_method_generic.png` - All layers/alphas for generic method
- `grid_alpha{X}.png` - Layer comparison at specific alpha
- `grid_layer{X}.png` - Alpha comparison at specific layer

### With Custom Parameters

```bash
uv run experiments/qwen3_vl/scripts/vl_caption_embedding_test.py \
    -i experiments/inputs/your_image.png \
    -o experiments/results/caption_test \
    --caption-tokens 1024 \
    --max-tokens 2500 \
    --vl-hidden-layer -6 \
    --vl-alpha 0.5 \
    --config config.toml \
    --profile default \
    --seed 42 \
    -v
```

### CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-i, --image` | Required | Input image path |
| `-o, --output-dir` | `experiments/results/caption_embedding_test` | Output directory |
| `--caption-tokens` | 1024 | Target caption length in tokens |
| `--max-tokens` | 2500 | Max tokens for generation (includes thinking) |
| `--steps` | 9 | Inference steps |
| `--seed` | 42 | Random seed |
| `--vl-model-path` | Auto-detect | Path to Qwen3-VL model |
| `--z-image-path` | From config | Path to Z-Image model |
| `--vl-hidden-layer` | -6 | Hidden layer for VL extraction |
| `--vl-alpha` | 0.5 | VL blend alpha |
| `--config` | `config.toml` | Config file path |
| `--profile` | `default` | Config profile |
| `-v, --verbose` | False | Enable verbose logging |

## Output Structure

```
caption_embedding_test/
└── YYYYMMDD_HHMMSS/
    ├── input.png                   # Input image
    ├── caption.txt                 # Generated caption (without thinking)
    ├── thinking.txt                # Thinking block content (if present)
    ├── raw_output.txt              # Full VL-Thinking output
    ├── result_vl_aligned.png       # VL aligned + text blended
    ├── result_vl_generic.png       # VL generic + text blended
    ├── result_text_only.png        # Pure text (caption) generation
    ├── comparison_grid.png         # Side-by-side comparison
    └── metadata.json               # Full experiment metadata
```

## References

- **VL Conditioning Research:** `experiments/qwen3_vl/README.md`
- **Research Findings:** `experiments/qwen3_vl/docs/research/findings.md`
- **Caption Studies:** `experiments/qwen3_vl/docs/caption_studies.md`
- **VL Blending:** `src/llm_dit/vl/blending.py`
- **VL Extractor:** `src/llm_dit/vl/qwen3_vl.py`

## Conclusion

The VL Caption Alignment approach represents a conceptual shift in how we think about vision conditioning. Rather than treating VL and text as competing signals, this method aligns them to reinforce each other.

### Summary of Findings (2025-12-15 Sweep)

The 50-image parameter sweep confirmed several hypotheses and revealed new insights:

**Confirmed:**
- Layer -6 remains optimal for VL extraction (clean, stable across all alphas)
- Alpha 0.5 provides good balanced blending
- Both caption-aligned and generic methods produce coherent results

**New Discoveries:**
- **Caption-aligned produces more diverse outputs** at low-to-mid alphas (0.0-0.5)
- **Methods converge at high alphas** (0.75-1.0) as VL dominates
- **Layer -2 shows instability** at high alphas, confirming it's suboptimal for VL
- **No catastrophic failures** across any parameter combination tested

### Key Takeaways

1. **Alignment benefits diversity, not just quality:** Caption-aligned VL doesn't just match generic VL - it produces more varied, creative outputs.

2. **Alpha controls convergence:** Low alpha = diverse text-driven outputs. High alpha = VL-dominated reconstruction. Choose based on use case.

3. **Layer -6 is robust:** Works well across all alphas and methods. Use as default.

4. **Caption vs Generic is a creative choice:** Use caption-aligned for artistic variation, generic for conservative/stable results.

### Recommended Defaults

```toml
[vl]
hidden_layer = -6       # Most stable
alpha = 0.5             # Balanced blend
method = "caption"      # More diverse outputs
caption_tokens = 1024   # Detailed description
```

### Open Questions

1. Does caption length (512 vs 1024 tokens) significantly affect output quality?
2. How does this method perform on non-anime images (photos, abstract art)?
3. Can we achieve pure style transfer (content from text, style from VL) with this approach?
4. With aligned caption, are image tokens actually necessary?

The key insight remains: **alignment matters more than raw VL influence**. A well-aligned caption-image pair produces more coherent and diverse results than a mismatched pair at any alpha.
