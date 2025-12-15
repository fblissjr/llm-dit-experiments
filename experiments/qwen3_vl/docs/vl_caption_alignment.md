# VL Caption Alignment: A Novel Approach to Vision Conditioning

> **Last Updated:** 2025-12-15

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

Location: `/home/fbliss/workspace/llm-dit-experiments/experiments/qwen3_vl/scripts/vl_caption_embedding_test.py`

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

- **VL Conditioning Research:** `/home/fbliss/workspace/llm-dit-experiments/experiments/qwen3_vl/README.md`
- **Research Findings:** `/home/fbliss/workspace/llm-dit-experiments/experiments/qwen3_vl/docs/research/findings.md`
- **Caption Studies:** `/home/fbliss/workspace/llm-dit-experiments/experiments/qwen3_vl/docs/caption_studies.md`
- **VL Blending:** `/home/fbliss/workspace/llm-dit-experiments/src/llm_dit/vl/blending.py`
- **VL Extractor:** `/home/fbliss/workspace/llm-dit-experiments/src/llm_dit/vl/qwen3_vl.py`

## Conclusion

The VL Caption Alignment approach represents a conceptual shift in how we think about vision conditioning. Rather than treating VL and text as competing signals, this method aligns them to reinforce each other. Early results show the approach produces coherent embeddings, but systematic experimentation is needed to:

1. Quantify quality improvement over previous methods
2. Find optimal hyperparameters
3. Understand failure modes
4. Establish best practices for practical use

The key insight is that **alignment matters more than raw VL influence**. A well-aligned caption-image pair may produce better results at high alpha than a mismatched pair at low alpha.
