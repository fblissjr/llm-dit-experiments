last updated: 2025-12-14

# Caption Length Study

This document explains the Caption Length Study experiment from a data flow perspective. The study tests how the Z-Image DiT responds to different embedding sequence lengths and fill strategies.

## Overview

The Caption Length Study answers the question: **Does the DiT care about exact embedding sequence length, or only the semantic content?**

We test this by:
1. Generating a source image from a simple prompt
2. Captioning the source image with Qwen3-VL (producing long descriptions)
3. Encoding the caption to embeddings and applying different fill strategies
4. Regenerating from the caption embeddings and comparing to the source

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CAPTION LENGTH STUDY PIPELINE                 │
└─────────────────────────────────────────────────────────────────┘

Phase 1: SOURCE GENERATION
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Simple Prompt│ -> │ Z-Image DiT  │ -> │ Source Image │
│ "A cat"      │    │ (seed=42)    │    │ 1024x1024    │
└──────────────┘    └──────────────┘    └──────────────┘
                                              |
                                              v
                                         [Saved to disk]

Phase 2: CAPTIONING
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Source Image │ -> │ Qwen3-VL     │ -> │ Detailed     │
│              │    │ (vision)     │    │ Caption Text │
└──────────────┘    └──────────────┘    └──────────────┘
                                         "A fluffy orange
                                          tabby cat with
                                          green eyes..."
                                         (500-2000 tokens)
                                              |
                                              v
                                         [Saved to disk]

Phase 3: EMBEDDING GENERATION
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Caption Text │ -> │ Qwen3-4B     │ -> │ Raw Embeddings│
│              │    │ Text Encoder │    │ [seq_len,2560]│
└──────────────┘    └──────────────┘    └──────────────┘
                                              |
                                              v
                                    [If seq_len > 1504:
                                     Apply compression]

Phase 4: FILL MODE APPLICATION
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Raw Embeddings│ -> │ Fill Mode    │ -> │Final Embeddings│
│ [seq_len,2560]│    │ (see below)  │    │[target,2560] │
└──────────────┘    └──────────────┘    └──────────────┘
                                              |
                                              v
                                     [Different lengths:
                                      128, 256, 512, etc.]

Phase 5: REGENERATION
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│Final Embeddings│ ->│ Z-Image DiT  │ -> │ Generated    │
│ (via prompt_  │   │ (seed=42)    │    │ Image        │
│  embeds param)│    └──────────────┘    └──────────────┘
└──────────────┘                              |
                                              v
                                         [Saved to disk]

Phase 6: ANALYSIS
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Source Image │    │ Generated    │    │              │
│      +       │ -> │   Image      │ -> │ SSIM Score   │
│ Generated    │    │ Comparison   │    │ (similarity) │
└──────────────┘    └──────────────┘    └──────────────┘
                                              |
                                              v
                                    [CSV/JSON results +
                                     comparison grids]
```

## Fill Modes Explained

Fill modes control how we transform raw embeddings to a target sequence length. This is the core experimental variable.

### Mode 1: `content_only` (Variable Length)

**Strategy:** Use only the actual caption content, truncate if needed.

```
Raw embeddings: [seq_len, 2560]
                 ↓
Target 512:     [min(seq_len, 512), 2560]

Example:
  Input:  [800, 2560]  → Output: [512, 2560]  (truncated)
  Input:  [200, 2560]  → Output: [200, 2560]  (unchanged)
```

**Purpose:** Test whether variable-length embeddings work as well as fixed-length.

**Characteristics:**
- Variable output length (capped at target)
- No padding/filler
- Maximum semantic content preservation
- Tests DiT's tolerance for variable sequence lengths

### Mode 2: `pad_end_zero` (Zero Padding)

**Strategy:** Fill to exact target length by appending zero vectors.

```
Raw embeddings: [seq_len, 2560]
                 ↓
                [seq_len, 2560] + [padding_len, 2560 zeros]
                 ↓
Target 512:     [512, 2560]

Example (seq_len=200, target=512):
  [200 content tokens] + [312 zero vectors] = [512 total]

Token layout:
  [0:200]   → Content embeddings
  [200:512] → All zeros
```

**Purpose:** Test whether zero-padding affects generation quality.

**Characteristics:**
- Exact target length
- Simple zero-fill strategy
- May signal "no content" to attention layers
- Common in sequence modeling

### Mode 3: `pad_end_mean` (Mean Padding)

**Strategy:** Fill to exact target length with mean embedding vector.

```
Raw embeddings: [seq_len, 2560]
                 ↓
                mean_vec = embeddings.mean(dim=0)  → [2560]
                 ↓
                [seq_len, 2560] + [padding_len, 2560] copies of mean_vec
                 ↓
Target 512:     [512, 2560]

Example (seq_len=200, target=512):
  mean_vec = [e1, e2, ..., e2560]  (average of all content tokens)

  [200 content tokens] + [312 copies of mean_vec] = [512 total]

Token layout:
  [0:200]   → Content embeddings
  [200:512] → Repeated mean vector (neutral semantic filler)
```

**Purpose:** Test whether "neutral" padding (average semantic content) performs better than zeros.

**Characteristics:**
- Exact target length
- Padding has same statistics as content (mean/std)
- May appear as "background" or neutral content to model
- Avoids zero-vector artifacts

### Mode 4: `pad_middle_zero` (Split Content with Middle Zeros)

**Strategy:** Split content in half, insert zeros in the middle, fill to target length.

```
Raw embeddings: [seq_len, 2560]
                 ↓
                Split at midpoint: [seq_len//2, 2560] + [seq_len//2, 2560]
                 ↓
                Insert zeros in middle to reach target
                 ↓
Target 512:     [512, 2560]

Example (seq_len=200, target=512):
  Content: [100 first half] + [312 zeros] + [100 second half] = [512 total]

Token layout:
  [0:100]   → First half of content
  [100:412] → Zero padding (middle gap)
  [412:512] → Second half of content
```

**Purpose:** Test position-dependent attention (does DiT weight early vs late tokens differently?).

**Characteristics:**
- Exact target length
- Preserves start and end of content
- Tests whether middle positions matter
- Unusual for natural text, but valid for testing

### Mode 5: `filler_repeat` (Cyclic Repeat)

**Strategy:** Repeat content tokens cyclically to fill target length.

```
Raw embeddings: [seq_len, 2560]
                 ↓
                Repeat content cyclically until target length
                 ↓
Target 512:     [512, 2560]

Example (seq_len=200, target=512):
  Content tokens: [A, B, C, ..., Z] (200 tokens)

  Output: [A,B,C,...,Z, A,B,C,...,Z, A,B,C,...,L] = [512 total]
          └─ cycle 1 ──┘ └─ cycle 2 ──┘ └─partial─┘

Token layout:
  [0:200]   → First copy of content
  [200:400] → Second copy of content
  [400:512] → Partial third copy (first 112 tokens)
```

**Purpose:** Test whether repeating semantic content is better/worse than padding.

**Characteristics:**
- Exact target length
- All positions contain semantic content
- May reinforce important features via repetition
- Tests attention's ability to handle duplicates

## Compression Modes (For Long Captions)

If the caption produces >1504 tokens (DiT RoPE limit), we apply compression BEFORE fill modes:

| Mode | Description | Quality Impact |
|------|-------------|----------------|
| `truncate` | Keep first 1504 tokens | Predictable, loses tail content |
| `interpolate` | Linear resampling to 1504 | Preserves all content, smooth |
| `pool` | Average pooling to 1504 | Experimental |
| `attention_pool` | Cosine similarity weighting | Experimental |

See `internal/research/long_prompt_research.md` for compression mode details.

## Experimental Variables

### Independent Variables

1. **Caption length** (controlled by captioning parameters)
   - Short captions: ~100-300 tokens
   - Medium captions: ~500-1000 tokens
   - Long captions: ~1500-2000 tokens (compressed)

2. **Fill mode** (5 modes listed above)
   - `content_only`, `pad_end_zero`, `pad_end_mean`, `pad_middle_zero`, `filler_repeat`

3. **Target sequence length** (multiple test points)
   - 128, 256, 512, 768, 1024, 1280, 1504

### Dependent Variables

1. **SSIM score** (Structural Similarity Index)
   - Measures how similar the regenerated image is to the source
   - Range: 0.0 (completely different) to 1.0 (identical)
   - Higher = caption+fill strategy preserves source image better

2. **Visual quality** (qualitative assessment)
   - Comparison grids for manual inspection

## Key Hypotheses

1. **Content matters more than length**
   - `content_only` (variable length) should perform well despite being shorter
   - Semantic content should drive generation quality

2. **Padding strategy affects quality**
   - `pad_end_mean` should outperform `pad_end_zero`
   - Zero-padding may create attention artifacts

3. **Position matters**
   - `pad_middle_zero` may show different results if DiT weights early/late tokens differently
   - Transformer attention typically prioritizes early tokens

4. **Repetition may help or hurt**
   - `filler_repeat` may reinforce important features (helpful)
   - OR may confuse attention with duplicates (harmful)

5. **Longer captions don't always help**
   - At some point, additional caption detail doesn't improve reconstruction
   - Compression artifacts may hurt long captions

## Output Structure

```
results/
└── caption_length_study_YYYYMMDD_HHMMSS/
    ├── source/
    │   ├── 0_source.png                  # Source image (from simple prompt)
    │   └── 0_source_caption.txt          # Generated caption
    │
    ├── regenerated/
    │   ├── content_only/
    │   │   ├── len_128_0.png
    │   │   ├── len_256_0.png
    │   │   └── ...
    │   ├── pad_end_zero/
    │   │   └── ...
    │   ├── pad_end_mean/
    │   │   └── ...
    │   ├── pad_middle_zero/
    │   │   └── ...
    │   └── filler_repeat/
    │       └── ...
    │
    ├── grids/
    │   ├── comparison_content_only.png   # Grid comparing all lengths
    │   ├── comparison_pad_end_zero.png
    │   └── ...
    │
    ├── results.csv                        # SSIM scores and metadata
    ├── results.json                       # Same data in JSON format
    └── config.json                        # Experiment configuration
```

## CLI Usage

### Basic Usage

```bash
# Run with default settings (all fill modes, lengths [128, 256, 512, 768, 1024])
uv run experiments/caption_length_study.py \
    --model-path /path/to/z-image-turbo \
    --vl-model-path /path/to/Qwen3-VL-4B-Instruct \
    --prompt "A cat sleeping in sunlight"
```

### Custom Configuration

```bash
# Test specific fill modes and lengths
uv run experiments/caption_length_study.py \
    --model-path /path/to/z-image-turbo \
    --vl-model-path /path/to/Qwen3-VL-4B-Instruct \
    --prompt "A mountain landscape at sunset" \
    --fill-modes content_only pad_end_mean \
    --target-lengths 256 512 1024 1504 \
    --seed 42
```

### Captioning Control

```bash
# Generate longer captions for more semantic content
uv run experiments/caption_length_study.py \
    --model-path /path/to/z-image-turbo \
    --vl-model-path /path/to/Qwen3-VL-4B-Instruct \
    --prompt "A futuristic cityscape" \
    --caption-max-tokens 2000 \
    --caption-temperature 0.7
```

### Device Placement

```bash
# Optimize for RTX 4090 (VL on CPU, DiT/VAE on CUDA)
uv run experiments/caption_length_study.py \
    --model-path /path/to/z-image-turbo \
    --vl-model-path /path/to/Qwen3-VL-4B-Instruct \
    --prompt "A cat" \
    --vae-device cuda \
    --dit-device cuda \
    --vl-device cpu \
    --text-encoder-device cpu
```

### Compression Mode Testing

```bash
# Test different compression modes for long captions
uv run experiments/caption_length_study.py \
    --model-path /path/to/z-image-turbo \
    --vl-model-path /path/to/Qwen3-VL-4B-Instruct \
    --prompt "A detailed scene..." \
    --long-prompt-mode interpolate \
    --caption-max-tokens 3000
```

### Output Control

```bash
# Custom output directory
uv run experiments/caption_length_study.py \
    --model-path /path/to/z-image-turbo \
    --vl-model-path /path/to/Qwen3-VL-4B-Instruct \
    --prompt "A cat" \
    --output experiments/results/my_caption_study
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--prompt` | Required | Simple prompt for source generation |
| `--fill-modes` | All 5 modes | Which fill strategies to test |
| `--target-lengths` | [128,256,512,768,1024] | Target embedding lengths |
| `--seed` | Random | Random seed for reproducibility |
| `--caption-max-tokens` | 512 | Max tokens for caption generation |
| `--caption-temperature` | 0.6 | Sampling temperature for captioning |
| `--long-prompt-mode` | `interpolate` | Compression for >1504 tokens |
| `--steps` | 9 | Inference steps for generation |
| `--width` | 1024 | Image width |
| `--height` | 1024 | Image height |

## Analysis and Interpretation

### Reading SSIM Scores

| SSIM Range | Interpretation |
|------------|----------------|
| 0.95 - 1.00 | Near-perfect reconstruction |
| 0.85 - 0.95 | Very good reconstruction (minor differences) |
| 0.70 - 0.85 | Good reconstruction (noticeable but similar) |
| 0.50 - 0.70 | Moderate similarity (some shared features) |
| < 0.50 | Poor reconstruction (significantly different) |

### Expected Patterns

1. **SSIM vs Target Length:**
   - Plot SSIM against target length for each fill mode
   - Look for plateau (length beyond which more doesn't help)

2. **Fill Mode Rankings:**
   - Which mode consistently scores highest?
   - Does ranking change at different lengths?

3. **Content vs Padding Trade-off:**
   - Compare `content_only` (short, all content) vs padded modes (long, mixed)
   - Find optimal balance point

### Comparison Grid Interpretation

```
Source | Length 128 | Length 256 | Length 512 | Length 1024
------------------------------------------------------------
[img]  |   [img]    |   [img]    |   [img]    |    [img]

Rows: Different fill modes
Cols: Different target lengths
```

Look for:
- Visual artifacts (especially with zero-padding or middle-padding)
- Color shifts (may indicate embedding distribution issues)
- Detail preservation (do longer sequences preserve more fine details?)
- Consistency (does the subject remain recognizable?)

## Implementation Details

### Caption Generation

Captioning uses Qwen3-VL's `generate()` method:

```python
from llm_dit.vl import VLEmbeddingExtractor

vl_extractor = VLEmbeddingExtractor.from_pretrained(
    vl_model_path,
    device="cpu",
)

caption = vl_extractor.generate(
    image=source_image,
    prompt="Describe this image in detail.",
    max_new_tokens=512,
    temperature=0.6,
    top_p=0.9,
)
```

### Fill Mode Application

Fill modes are applied in `experiments/caption_length_study.py`:

```python
def apply_fill_mode(embeddings, target_length, mode):
    """
    Apply fill mode to reach target sequence length.

    Args:
        embeddings: [seq_len, hidden_dim] tensor
        target_length: Desired sequence length
        mode: Fill strategy

    Returns:
        [target_length, hidden_dim] tensor (or [seq_len, hidden_dim] for content_only)
    """
    # Implementation varies by mode
    # See source code for details
```

### Direct Embedding Injection

The study bypasses normal text encoding by injecting embeddings directly:

```python
from llm_dit import ZImagePipeline

pipe = ZImagePipeline.from_pretrained(...)

# Normal generation (from text)
result = pipe(prompt="A cat", ...)

# Caption study (from embeddings)
result = pipe(
    prompt="",  # Empty prompt
    prompt_embeds=caption_embeddings,  # Direct injection
    num_inference_steps=9,
    ...
)
```

This ensures we're testing only the embedding fill strategy, not the text encoder.

## Related Research

- **Long Prompt Research:** `internal/research/long_prompt_research.md`
  - Compression modes for >1504 tokens
  - Quality vs compression ratio trade-offs

- **VL Conditioning:** `experiments/qwen3_vl/README.md`
  - Vision-language embedding extraction
  - Blending strategies (related to fill modes)

- **Hidden Layer Selection:** `internal/research/hidden_layer_selection.md`
  - Which layer to extract embeddings from
  - Affects caption embedding quality

## Future Directions

1. **Adaptive Fill Modes**
   - Automatically select fill mode based on content length
   - Hybrid modes (e.g., mean-pad up to 512, then content-only)

2. **Learned Padding**
   - Train a small MLP to generate optimal padding vectors
   - Condition on content statistics

3. **Position Encoding Analysis**
   - Investigate whether DiT's RoPE affects fill mode performance
   - Test non-uniform position assignments

4. **Cross-Modal Caption Study**
   - Use image captions from different VL models
   - Test whether caption style affects reconstruction

5. **Iterative Refinement**
   - Generate → caption → regenerate → caption (loop)
   - Measure convergence or divergence

## References

- **Z-Image DiT:** Text sequence length limit is 1504 tokens (RoPE constraint)
- **Qwen3-4B:** 2560 hidden dimensions, 36 layers
- **Qwen3-VL:** Vision-language model for captioning, same text encoder architecture
- **SSIM:** Structural Similarity Index (Wang et al., 2004)

## Conclusion

The Caption Length Study provides empirical data on how Z-Image DiT responds to embedding sequence length and fill strategies. Results inform best practices for:
- Handling variable-length prompts
- Padding strategies for batch processing
- Long prompt compression trade-offs
- Position-dependent attention patterns

By treating the DiT as a black box and varying only the input embeddings, we isolate the model's sensitivity to sequence length independent of semantic content.
