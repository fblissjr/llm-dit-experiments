# caption length study run guide

last updated: 2025-12-14

## overview

This experiment tests how Z-Image DiT responds to different embedding configurations. The core workflow:

1. Generate a "source" image from a simple prompt (e.g., "a cat")
2. Caption that source image with Qwen3-VL to get a detailed description (~500-2000 tokens)
3. Generate new images using variations of that caption's embeddings
4. Compare regenerated images to source using SSIM

**Key insight:** By generating source → captioning → regenerating, we can measure how well different embedding configurations preserve the original image's content.

---

## experiment workflow (what actually happens)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Source Generation                                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Simple Prompt: "a cat"                                                 │
│         │                                                                │
│         ▼                                                                │
│   ┌─────────────┐                                                        │
│   │ Z-Image DiT │ ──────► source_image.png                               │
│   │  (seed=42)  │                                                        │
│   └─────────────┘                                                        │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│ PHASE 2: Captioning                                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   source_image.png                                                       │
│         │                                                                │
│         ▼                                                                │
│   ┌─────────────┐                                                        │
│   │  Qwen3-VL   │ ──────► "A fluffy orange tabby cat with green eyes,   │
│   │  (caption)  │          lying on a soft beige cushion near a         │
│   └─────────────┘          sunlit window. The warm afternoon light..."  │
│                            (500-2000 tokens)                             │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│ PHASE 3: Regeneration with Embedding Variations                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Detailed Caption                                                       │
│         │                                                                │
│         ├──► Qwen3-4B Encode (hidden_layer=-2)                           │
│         │         │                                                      │
│         │         ▼                                                      │
│         │    embeddings [seq_len, 2560]                                  │
│         │         │                                                      │
│         │         ├──► apply_fill_mode(target=300, mode="content_only")  │
│         │         ├──► apply_fill_mode(target=300, mode="pad_end_zero")  │
│         │         ├──► apply_fill_mode(target=600, mode="content_only")  │
│         │         └──► ... more combinations ...                         │
│         │                    │                                           │
│         │                    ▼                                           │
│         │              ┌─────────────┐                                   │
│         │              │ Z-Image DiT │ ──────► regenerated_images/       │
│         │              │  (seed=42)  │                                   │
│         │              └─────────────┘                                   │
│         │                                                                │
│         └──► (OR) VL Extraction from source_image + caption              │
│                   │                                                      │
│                   ▼                                                      │
│              VL embeddings [seq_len, 2560]                               │
│                   │                                                      │
│                   └──► same fill_mode variations ...                     │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│ PHASE 4: Analysis                                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   source_image.png  vs  regenerated_image.png                            │
│         │                        │                                       │
│         └────────┬───────────────┘                                       │
│                  ▼                                                       │
│            SSIM Score (0.0 - 1.0)                                        │
│            Higher = more similar to source                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## variable 1: target_lengths

**What it tests:** Does embedding sequence length matter to the DiT?

**Default values:** `50, 150, 300, 600, 1000, 1504`

**What actually happens:**

```python
# The caption gets encoded to embeddings
embeddings = encoder.encode(caption)  # shape: [actual_len, 2560]

# Then we adjust to target length
if fill_mode == "content_only":
    # Just truncate/cap - keeps first N embeddings
    result = embeddings[:target_length]  # shape: [min(actual, target), 2560]
else:
    # Pad to exact target length (see fill_modes below)
    result = apply_fill_mode(embeddings, target_length, fill_mode)
```

**Real example:**

```
Caption: "A fluffy orange tabby cat..." (847 tokens when encoded)
Encoded embeddings shape: [847, 2560]

target_length=300, fill_mode=content_only:
  → Result: [300, 2560] (first 300 embeddings, truncated)

target_length=300, fill_mode=pad_end_zero:
  → Result: [300, 2560] (first 300 embeddings, truncated - no padding needed)

target_length=1000, fill_mode=content_only:
  → Result: [847, 2560] (all 847 embeddings, capped at actual length)

target_length=1000, fill_mode=pad_end_zero:
  → Result: [1000, 2560] (847 content + 153 zero padding)
```

**Research questions answered:**
- Does DiT produce better images with longer embedding sequences?
- Is there a "sweet spot" length where quality peaks?
- Does truncating detailed captions lose important information?

---

## variable 2: fill_modes

**What it tests:** How should we pad/fill embeddings when content is shorter than target?

**Default values:** `content_only, pad_end_zero, pad_end_mean, pad_middle_zero, filler_repeat`

### Mode: `content_only`

**What it does:** No padding. Just truncates if content exceeds target. Variable output length.

```python
def apply_fill_mode_content_only(embeddings, target_length):
    # Just cap at target length, don't pad
    return embeddings[:target_length]
    # Returns: [min(seq_len, target), 2560]
```

**Example:**
```
Input: [847, 2560], target=1000
Output: [847, 2560]  ← keeps all 847, doesn't pad to 1000

Input: [847, 2560], target=300
Output: [300, 2560]  ← truncates to 300
```

**Purpose:** Baseline - what happens with natural variable-length embeddings?

---

### Mode: `pad_end_zero`

**What it does:** Pads with zero vectors at the end to reach exact target length.

```python
def apply_fill_mode_pad_end_zero(embeddings, target_length):
    seq_len, hidden_dim = embeddings.shape  # [847, 2560]

    content = embeddings[:target_length]  # Truncate if needed
    if content.shape[0] >= target_length:
        return content

    # Create zero padding
    pad_length = target_length - content.shape[0]  # e.g., 1000 - 847 = 153
    padding = torch.zeros(pad_length, hidden_dim)  # [153, 2560] of zeros

    return torch.cat([content, padding], dim=0)  # [1000, 2560]
```

**Example:**
```
Input: [847, 2560], target=1000

content = embeddings[:1000] = [847, 2560]  (all of it)
pad_length = 1000 - 847 = 153
padding = zeros([153, 2560])

Output: [content | padding] = [1000, 2560]
         ↑                     ↑
         847 real embeddings   153 zero vectors

Visual:
[e0][e1][e2]...[e846][0][0][0]...[0]
 ←── 847 content ──→ ←── 153 zeros ──→
```

**Purpose:** Tests if DiT is sensitive to zero-padding at the end.

---

### Mode: `pad_end_mean`

**What it does:** Pads with the mean embedding vector repeated.

```python
def apply_fill_mode_pad_end_mean(embeddings, target_length):
    seq_len, hidden_dim = embeddings.shape

    content = embeddings[:target_length]
    if content.shape[0] >= target_length:
        return content

    pad_length = target_length - content.shape[0]

    # Calculate mean embedding across all content tokens
    mean_emb = content.mean(dim=0, keepdim=True)  # [1, 2560]

    # Repeat mean to fill padding
    padding = mean_emb.expand(pad_length, -1)  # [153, 2560] all same vector

    return torch.cat([content, padding], dim=0)
```

**Example:**
```
Input: [847, 2560], target=1000

mean_emb = embeddings.mean(dim=0)  # Average of all 847 embeddings → [2560]
padding = mean_emb repeated 153 times → [153, 2560]

Output: [content | mean repeated] = [1000, 2560]

Visual:
[e0][e1][e2]...[e846][μ][μ][μ]...[μ]
 ←── 847 content ──→ ←── 153 × mean ──→
```

**Purpose:** Tests if "realistic" padding (mean of existing embeddings) works better than zeros.

---

### Mode: `pad_middle_zero`

**What it does:** Splits content in half, inserts zeros in the middle.

```python
def apply_fill_mode_pad_middle_zero(embeddings, target_length):
    seq_len, hidden_dim = embeddings.shape

    content = embeddings[:target_length]
    if content.shape[0] >= target_length:
        return content

    pad_length = target_length - content.shape[0]

    # Split content in half
    half = content.shape[0] // 2
    first_half = content[:half]      # [423, 2560]
    second_half = content[half:]     # [424, 2560]

    # Create zero padding
    padding = torch.zeros(pad_length, hidden_dim)  # [153, 2560]

    # Concatenate: first_half + padding + second_half
    return torch.cat([first_half, padding, second_half], dim=0)
```

**Example:**
```
Input: [847, 2560], target=1000

first_half = embeddings[:423] = [423, 2560]
second_half = embeddings[423:] = [424, 2560]
padding = zeros([153, 2560])

Output: [first_half | padding | second_half] = [1000, 2560]

Visual:
[e0][e1]...[e422][0][0]...[0][e423][e424]...[e846]
 ←── 423 ──→     ←── 153 ──→ ←──── 424 ────→
```

**Purpose:** Tests if DiT attention can "skip over" zeros in the middle.

---

### Mode: `filler_repeat`

**What it does:** Repeats content tokens cyclically to fill target length.

```python
def apply_fill_mode_filler_repeat(embeddings, target_length):
    seq_len, hidden_dim = embeddings.shape

    content = embeddings[:target_length]
    if content.shape[0] >= target_length:
        return content

    # Calculate how many repeats needed
    repeats_needed = (target_length + content.shape[0] - 1) // content.shape[0]
    # e.g., (1000 + 847 - 1) // 847 = 2 repeats

    repeated = content.repeat(repeats_needed, 1)  # [1694, 2560]
    return repeated[:target_length]  # [1000, 2560]
```

**Example:**
```
Input: [847, 2560], target=1000

repeats_needed = ceil(1000 / 847) = 2
repeated = content × 2 = [1694, 2560]
result = repeated[:1000] = [1000, 2560]

Visual:
[e0][e1]...[e846][e0][e1]...[e152]
 ←── 847 original ──→←── 153 repeated ──→
```

**Purpose:** Tests if DiT detects/is harmed by repetitive embedding patterns.

---

## variable 3: hidden_layers

**What it tests:** Which Qwen3-4B transformer layer produces the best embeddings for image generation?

**Default:** `-2` (penultimate layer, layer 35 of 36)

**Other common values:** `-6, -8, -12, -16, -21`

**What actually happens:**

```python
# Qwen3-4B has 36 transformer layers (0-35)
# hidden_layer=-2 means layer 35 (penultimate)
# hidden_layer=-6 means layer 31

# When encoding:
outputs = model(input_ids, output_hidden_states=True)
# outputs.hidden_states is tuple of 37 tensors:
#   [0]: input embeddings
#   [1-36]: outputs of layers 0-35

embeddings = outputs.hidden_states[hidden_layer]
# hidden_layer=-2 → outputs.hidden_states[-2] → layer 35 output
```

**Why this matters:**

```
Layer 0-10:   Early processing - basic token patterns
Layer 11-20: Mid processing - semantic understanding
Layer 21-30: Late processing - abstract concepts
Layer 31-35: Final processing - instruction-following (SFT-modified)
```

**Hypothesis:** Qwen3-4B was instruction-tuned (SFT), which modified later layers for "helpful assistant" behavior. Middle layers may retain more visual/concrete information.

**CLI example:**
```bash
# Test layers -2 (default), -6, -12
./experiments/sweep_caption_hidden_layer.sh --hidden-layers="-2,-6,-12"
```

---

## variable 4: vl_hidden_layers (VL embedding mode)

**What it tests:** Can we use Qwen3-VL embeddings (from processing image + caption) instead of text-only embeddings?

**Default:** None (uses text encoding)

**When enabled:** Uses VL model to extract embeddings from source image + caption together

**What actually happens:**

```python
# Standard text encoding (default):
embeddings = qwen3_4b.encode(caption)  # Only sees text

# VL embedding mode:
embeddings = qwen3_vl.extract(
    image=source_image,  # Sees the actual image
    text=caption,        # Plus the caption text
    hidden_layer=-6,     # Which VL layer to extract from
    text_tokens_only=False,  # Include image tokens
)
```

**Why VL might be better:**
- VL model has seen the actual source image
- Embeddings encode visual information, not just text description
- May capture details the caption missed

**CLI example:**
```bash
# Use VL embeddings from layers -6 and -8
./experiments/sweep_caption_vl.sh --vl-hidden-layers="-6,-8"
```

---

## variable 5: token_modes (VL only)

**What it tests:** Which tokens from VL extraction to include?

**Default:** `full`

**Options:**
- `full` - All tokens (text + image tokens)
- `text_only` - Only text tokens (strips image region)
- `image_only` - Only image tokens (visual info only)
- `image_no_markers` - Image tokens without special markers

**What actually happens:**

```python
# When processing image + caption, Qwen3-VL creates:
# [text_tokens...][<image>][image_patch_tokens...][</image>][more_text...]

# full: returns everything
# text_only: strips [<image>]...[</image>] region
# image_only: keeps only image_patch_tokens
# image_no_markers: keeps image patches, removes <image>/<image> markers
```

**Token counts (approximate for 1024x1024 image):**
```
full:              ~1100-1200 tokens (caption + image patches)
text_only:         ~100-500 tokens (just caption)
image_only:        ~1026 tokens (image patches only)
image_no_markers:  ~1024 tokens (patches without markers)
```

**CLI example:**
```bash
./experiments/sweep_caption_vl.sh \
    --vl-hidden-layers="-6" \
    --token-modes="full,text_only,image_only"
```

---

## variable 6: compression_modes

**What it tests:** How to compress embeddings when caption exceeds 1504 tokens?

**Only applies when:** Raw caption embeddings > 1504 tokens

**Options:** `truncate, interpolate, pool, attention_pool`

```python
# Applied BEFORE fill_mode when embeddings.shape[0] > 1504

if embeddings.shape[0] > 1504 and compression_mode:
    embeddings = compress_embeddings(embeddings, 1504, mode=compression_mode)
```

**Modes explained:**

```python
# truncate: Keep first 1504 tokens
result = embeddings[:1504]

# interpolate: Linear interpolation to 1504
# Smooth resampling that preserves all information
result = F.interpolate(embeddings, size=1504, mode='linear')

# pool: Average pooling with stride
# Groups of adjacent tokens are averaged
pool_size = embeddings.shape[0] // 1504
result = avg_pool(embeddings, kernel_size=pool_size)

# attention_pool: Weighted by cosine similarity to mean
# Tokens similar to the "average meaning" get more weight
weights = cosine_similarity(embeddings, embeddings.mean())
result = weighted_resample(embeddings, weights, target=1504)
```

---

## complete experiment matrix

When you run a sweep script, here's what gets tested:

### sweep_caption_fill_modes.sh (--quick)

```
Prompts: ["a cat"]
Seeds: [42]
Target lengths: [600]
Fill modes: [content_only, pad_end_zero, filler_repeat]
Hidden layers: [-2]

Total images: 1 source + 3 variants = 4 images
```

### sweep_caption_fill_modes.sh (full)

```
Prompts: ["a cat", "a mountain landscape", "a woman portrait"]
Seeds: [42, 123]
Target lengths: [600]
Fill modes: [content_only, pad_end_zero, pad_end_mean, pad_middle_zero, filler_repeat]
Hidden layers: [-2]

Total images:
  3 prompts × 2 seeds = 6 sources
  6 sources × 5 fill_modes = 30 variants
  Total: 36 images
```

### sweep_caption_hidden_layer.sh (--quick)

```
Prompts: ["a cat"]
Seeds: [42]
Target lengths: [600]
Fill modes: [content_only]
Hidden layers: [-2, -6, -12]

Total images: 1 source + 3 variants = 4 images
```

### sweep_caption_vl.sh (--quick)

```
Prompts: ["a cat"]
Seeds: [42]
Target lengths: [600]
Fill modes: [content_only]
VL hidden layers: [-6, -8]
Token modes: [full, text_only]

Total images: 1 source + (2 layers × 2 token_modes) = 5 images
```

---

## interpreting results

### output structure

```
experiments/results/caption_fill_modes_YYYYMMDD_HHMMSS/
├── source/
│   ├── a_cat_seed42.png           # Source image
│   ├── a_cat_seed42.txt           # Qwen3-VL caption
│   └── a_cat_seed42_tokens.txt    # Token count info
├── regenerated/
│   ├── a_cat_seed42_len600_content_only_L-2.png
│   ├── a_cat_seed42_len600_pad_end_zero_L-2.png
│   └── ...
├── grids/
│   ├── a_cat_seed42_length_grid.png    # Compare target lengths
│   ├── a_cat_seed42_fill_mode_grid.png # Compare fill modes
│   └── a_cat_seed42_layer_grid.png     # Compare hidden layers
├── results.json                        # Full metadata
└── summary.csv                         # SSIM scores, timing
```

### reading SSIM scores

```csv
source_prompt,seed,target_length,fill_mode,ssim
"a cat",42,600,content_only,0.847
"a cat",42,600,pad_end_zero,0.832
"a cat",42,600,filler_repeat,0.651
```

**Interpretation:**
- SSIM closer to 1.0 = regenerated image looks more like source
- Higher SSIM = that configuration preserves image content better
- If `content_only` always wins → padding hurts quality
- If `pad_end_mean` ≈ `pad_end_zero` → DiT doesn't care about padding values

### what to look for

**Fill mode comparison:**
```
content_only SSIM: 0.85
pad_end_zero SSIM: 0.83
pad_end_mean SSIM: 0.84
filler_repeat SSIM: 0.65  ← Repetition hurts!
```

**Length comparison:**
```
len=50:   SSIM 0.72  ← Too short, missing details
len=300:  SSIM 0.85  ← Sweet spot?
len=600:  SSIM 0.86
len=1000: SSIM 0.84  ← Diminishing returns
len=1504: SSIM 0.83  ← Padding not helping
```

**Hidden layer comparison:**
```
L-2:  SSIM 0.85  (default)
L-6:  SSIM 0.87  ← Better for captions?
L-12: SSIM 0.82
L-21: SSIM 0.78  ← Too early
```

---

## full command examples

### quick test (verify setup works)

```bash
# Preview what would run
./experiments/sweep_caption_fill_modes.sh --quick --dry-run

# Actually run (generates ~4 images)
./experiments/sweep_caption_fill_modes.sh --quick
```

### comprehensive fill mode study

```bash
./experiments/sweep_caption_fill_modes.sh \
    --config config.toml \
    --profile default \
    --prompts "a cat,a cyberpunk cityscape,an elderly man portrait" \
    --seeds 42,123,456 \
    --target-length 600
```

### hidden layer sweep

```bash
./experiments/sweep_caption_hidden_layer.sh \
    --hidden-layers "-2,-4,-6,-8,-12,-16,-21" \
    --prompts "a detailed forest scene"
```

### vl vs text encoding comparison

```bash
# First: text encoding at different layers
./experiments/sweep_caption_hidden_layer.sh \
    --hidden-layers "-2,-6,-8" \
    --prompts "a cat"

# Then: VL encoding at same layers
./experiments/sweep_caption_vl.sh \
    --vl-hidden-layers "-2,-6,-8" \
    --token-modes "full" \
    --prompts "a cat"

# Compare SSIM scores between the two
```

### run everything

```bash
# Quick test of all experiments (~5-10 minutes)
./experiments/run_all_caption_sweeps.sh --quick

# Full study (1-2 hours)
./experiments/run_all_caption_sweeps.sh

# Without VL (if Qwen3-VL not available)
./experiments/run_all_caption_sweeps.sh --skip-vl
```

---

## troubleshooting

### "VL model path not provided"

```bash
# Either specify in config.toml:
[default.vl]
model_path = "/path/to/Qwen3-VL-4B-Instruct"

# Or via CLI:
--vl-model-path /path/to/Qwen3-VL-4B-Instruct
```

### "CUDA out of memory"

```bash
# VL uses a lot of VRAM. Options:

# 1. Skip VL experiments
./experiments/run_all_caption_sweeps.sh --skip-vl

# 2. Use CPU for VL (slow but works)
# Edit sweep_caption_vl.sh to add --vl-device cpu
```

### "negative number interpreted as flag"

```bash
# Wrong:
--hidden-layers -2,-6,-8

# Correct (use = syntax):
--hidden-layers="-2,-6,-8"
```

### SSIM scores all similar

This might mean:
1. The DiT is robust to these variations (good finding!)
2. The variations aren't different enough
3. SSIM isn't sensitive to the changes you care about

Try looking at the visual comparison grids in `grids/` directory.
