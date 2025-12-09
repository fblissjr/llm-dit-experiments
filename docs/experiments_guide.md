# Experiments Guide

This guide covers the complete experiment infrastructure for Z-Image ablation studies.

## Overview

The experiment system consists of:
- **Standard prompt set** (60 prompts in natural language style)
- **Experiment runner** (6 predefined experiment types)
- **Metrics collection** (ImageReward + SigLIP2)
- **Web UI** (manual exploration with all parameters)

---

## Configuration

All settings are managed via `config.toml`. Copy the example and customize:

```bash
cp config.toml.example config.toml
```

### Key Settings

```toml
[default]
model_path = "/path/to/z-image-turbo"

[default.encoder]
device = "cpu"               # cpu/cuda/mps/auto
hidden_layer = -2            # -1 to -6 (or deeper)

[default.scheduler]
shift = 3.0                  # FlowMatch shift (try 2.0-5.0)

[default.generation]
num_inference_steps = 9      # 4-20 typical range

[default.pytorch]
long_prompt_mode = "truncate" # truncate/interpolate/pool/attention_pool
```

### Using Config

```bash
# Use default profile
uv run scripts/generate.py --config config.toml "A cat"

# Use specific profile
uv run scripts/generate.py --config config.toml --profile rtx4090 "A cat"

# Override specific values
uv run scripts/generate.py --config config.toml --shift 4.0 "A cat"
```

---

## Quick Start

```bash
# Dry run to see what would be generated
uv run experiments/run_ablation.py --experiment shift_sweep --dry-run

# Run shift sweep on 3 animal prompts (uses config.toml)
uv run experiments/run_ablation.py \
  --config config.toml \
  --experiment shift_sweep \
  --prompt-category animals \
  --max-prompts 3

# Run with metrics collection
uv run experiments/run_ablation.py \
  --config config.toml \
  --experiment hidden_layer \
  --prompt-category simple_objects \
  --compute-metrics
```

---

## Standard Prompt Set

Located at `experiments/prompts/standard_prompts.yaml` - 60 diverse prompts written in Qwen3's natural language style.

### Categories (6 prompts each)

| Category | Description | Example |
|----------|-------------|---------|
| `simple_objects` | Single objects, clear subjects | "A ceramic coffee mug sits on a wooden table..." |
| `animals` | Wildlife and pets | "A tabby cat lies curled in a pool of sunlight..." |
| `humans` | Portraits and figures | "A woman in her thirties stands at a rain-streaked window..." |
| `scenes` | Indoor/outdoor environments | "A cozy bookshop corner where afternoon light filters..." |
| `landscapes` | Natural environments | "Rolling hills stretch toward distant mountains..." |
| `artistic_styles` | Style-specific prompts | "In the style of Art Nouveau, a woman's portrait emerges..." |
| `lighting` | Lighting-focused scenes | "Golden hour transforms an ordinary wheat field..." |
| `abstract` | Abstract concepts | "Geometric shapes cascade through infinite space..." |
| `technical` | Technical/product shots | "A vintage mechanical watch lies open..." |
| `text_rendering` | Text in images | "A weathered wooden sign reads 'Welcome Home'..." |

### Using Prompts in Python

```python
from experiments.prompts import (
    load_standard_prompts,
    get_prompts_by_category,
    get_prompt_by_id,
    get_all_prompt_texts,
)

# Load all prompts
data = load_standard_prompts()
print(f"Total prompts: {data['metadata']['total_prompts']}")

# Get prompts by category
animals = get_prompts_by_category("animals")
for p in animals:
    print(f"{p['id']}: {p['prompt'][:50]}...")

# Get single prompt by ID
prompt = get_prompt_by_id("animal_001")
print(prompt["prompt"])

# Get just the text (for generation)
texts = get_all_prompt_texts()
```

### Listing Available Prompts

```bash
uv run experiments/run_ablation.py --list-prompts
```

---

## Experiment Types

### 1. Shift Sweep (`shift_sweep`)

Tests the scheduler shift parameter across values [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].

```bash
uv run experiments/run_ablation.py \
  --config config.toml \
  --experiment shift_sweep \
  --prompt-category landscapes
```

**What it tests:** How the FlowMatchEuler scheduler shift affects image quality and prompt adherence.

**Default shift:** 3.0 (Z-Image Turbo default)

### 2. Shift + Steps Grid (`shift_steps_grid`)

Grid search over shift [2.0, 3.0, 4.0] x steps [6, 9, 12, 15].

```bash
uv run experiments/run_ablation.py \
  --config config.toml \
  --experiment shift_steps_grid \
  --max-prompts 5
```

**What it tests:** Interaction between shift and step count. Higher steps may prefer different shift values.

### 3. Hidden Layer Ablation (`hidden_layer`)

Tests embedding extraction from layers [-1, -2, -3, -4, -5, -6].

```bash
uv run experiments/run_ablation.py \
  --config config.toml \
  --experiment hidden_layer \
  --prompt-category humans
```

**What it tests:** Which Qwen3 hidden layer produces the best image-text alignment.

**Default:** -2 (penultimate layer)

**Theory:** Deeper layers (-1) have more task-specific features, while earlier layers (-4, -5, -6) have more general semantic features.

### 4. Think Block Impact (`think_block`)

Tests different thinking content variations:
- `None` - No think block
- `""` - Empty think block (force_think_block=True)
- `"High quality, detailed, photorealistic"` - Quality keywords
- `"Soft lighting, warm colors, peaceful atmosphere"` - Mood keywords
- `"Sharp focus, crisp details, professional composition"` - Technical keywords

```bash
uv run experiments/run_ablation.py \
  --config config.toml \
  --experiment think_block \
  --prompt-category scenes
```

**What it tests:** Whether the `<think>...</think>` block content affects generation quality.

### 5. System Prompt Ablation (`system_prompt`)

Tests different system prompts:
- `None` - No system prompt
- `"You are a professional photographer."`
- `"You are an artistic painter."`
- `"You are a technical illustrator."`
- `"Generate high quality images."`

```bash
uv run experiments/run_ablation.py \
  --config config.toml \
  --experiment system_prompt \
  --prompt-category artistic_styles
```

**What it tests:** Whether system prompts steer generation style.

### 6. Steps Only (`steps_only`)

Tests step counts [4, 6, 8, 9, 10, 12, 15, 20].

```bash
uv run experiments/run_ablation.py \
  --config config.toml \
  --experiment steps_only \
  --seeds 42,123,456
```

**What it tests:** Quality vs speed tradeoff. Z-Image Turbo is optimized for 8-9 steps.

---

## CLI Reference

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--experiment` | Experiment type (see above) |
| `--config` | Path to config.toml (recommended) |

### Prompt Selection

| Argument | Description |
|----------|-------------|
| `--prompt-category` | Use prompts from one category |
| `--prompt-ids` | Comma-separated prompt IDs (e.g., `animal_001,simple_002`) |
| `--max-prompts` | Limit number of prompts |

### Execution Options

| Argument | Description |
|----------|-------------|
| `--dry-run` | Preview without generating |
| `--seeds` | Comma-separated seeds (default: `42`) |
| `--output-dir` | Output directory (default: `experiments/results`) |
| `--compute-metrics` | Calculate ImageReward and SigLIP2 scores |

### Device Placement (via config.toml)

```toml
[default.encoder]
device = "cpu"      # Qwen3 encoder

[default.pipeline]
device = "cuda"     # DiT + VAE
```

Or override via CLI: `--text-encoder-device cpu --dit-device cuda --vae-device cuda`

### Listing Options

| Argument | Description |
|----------|-------------|
| `--list-experiments` | Show all experiment types |
| `--list-prompts` | Show all prompts by category |

---

## Metrics

### ImageReward (Human Preference)

Trained on 137k human preference comparisons. Best for overall quality judgment.

```python
from experiments.metrics import compute_image_reward, ImageRewardScorer

# Quick single score
score = compute_image_reward("A cat sleeping", "output.png")
print(f"ImageReward: {score:.4f}")

# Batch scoring
scorer = ImageRewardScorer()
scores = scorer.score_batch(prompts, images)
```

**Interpretation:**
- `> 0.5`: Excellent quality
- `0.0 - 0.5`: Good quality
- `-0.5 - 0.0`: Acceptable
- `< -0.5`: Poor quality

### SigLIP2 (Image-Text Alignment)

2B parameter model (`google/siglip2-giant-opt-patch16-384`). Better than CLIP for retrieval tasks.

```python
from experiments.metrics import compute_siglip_score, SigLIPScorer

# Quick single score
score = compute_siglip_score("A cat sleeping", "output.png")
print(f"SigLIP: {score:.4f}")

# Batch scoring
scorer = SigLIPScorer()
scores = scorer.score_batch(prompts, images)
```

**Interpretation:**
- `> 0.35`: Excellent alignment
- `0.28 - 0.35`: Good alignment
- `0.20 - 0.28`: Moderate alignment
- `< 0.20`: Weak alignment

---

## Output Structure

```
experiments/results/
  shift_sweep/                    # Experiment name
    images/                       # Generated images
      shift_sweep_animal_001_shift_3.0_seed42.png
      shift_sweep_animal_001_shift_4.0_seed42.png
      ...
    metadata/                     # Per-image JSON metadata
      shift_sweep_animal_001_shift_3.0_seed42.json
      ...
    shift_sweep_summary.csv       # All results in CSV
    shift_sweep_summary.json      # Summary statistics
```

### CSV Format

```csv
prompt_id,seed,variable_name,variable_value,generation_time_seconds,token_count,image_reward,siglip_score,error,output_path
animal_001,42,shift,3.0,2.34,156,0.42,0.31,,experiments/results/shift_sweep/images/...
```

### JSON Summary

```json
{
  "experiment": "shift_sweep",
  "timestamp": "2025-12-09T15:30:00",
  "total_runs": 36,
  "successful_runs": 36,
  "failed_runs": 0,
  "total_time_seconds": 84.2,
  "image_reward": {
    "mean": 0.38,
    "min": -0.12,
    "max": 0.72,
    "count": 36
  },
  "siglip_score": {
    "mean": 0.29,
    "min": 0.18,
    "max": 0.41,
    "count": 36
  }
}
```

---

## Web UI for Manual Exploration

Start the web server:

```bash
# Using config.toml (recommended)
uv run web/server.py --config config.toml

# Or with a specific profile
uv run web/server.py --config config.toml --profile rtx4090
```

Open http://localhost:8000 (or the port specified in `[server]`).

### Available Controls

**Basic:**
- Prompt text
- Template selection
- Resolution presets
- Steps (1-50)
- Seed

**Chat Template Options:**
- System prompt
- Thinking block enable/disable
- Thinking content
- Assistant content

**Scheduler Options:**
- Guidance scale (0.0-10.0, default 0.0)
- Scheduler shift (0.5-10.0, default 3.0)
- Hidden layer (-6 to -1, default -2)

**Prompt Rewriter:**
- Template-based or custom rewriting
- Temperature, top_p, min_p, max_tokens controls

### History

All generations are saved in history (max 50). Click any thumbnail to:
- View full image
- See generation parameters
- Click "Edit" to reuse all parameters

---

## Example Workflows

### Workflow 1: Find Optimal Shift

```bash
# 1. Quick sweep with few prompts
uv run experiments/run_ablation.py \
  --config config.toml \
  --experiment shift_sweep \
  --prompt-category simple_objects \
  --max-prompts 3 \
  --compute-metrics

# 2. Review results
cat experiments/results/shift_sweep/shift_sweep_summary.json

# 3. Fine-tune around best value in web UI
```

### Workflow 2: Compare Hidden Layers

```bash
# Run hidden layer ablation
uv run experiments/run_ablation.py \
  --config config.toml \
  --experiment hidden_layer \
  --prompt-category humans \
  --seeds 42,123 \
  --compute-metrics

# Compare metrics in CSV
# Look for highest mean ImageReward and SigLIP scores
```

### Workflow 3: Exploratory Testing in Web UI

1. Start server: `uv run web/server.py --config config.toml`
2. Try different shift values (2.0, 3.0, 4.0) on same prompt + seed
3. Compare results in history panel
4. Use "Edit" to iterate on best settings

### Workflow 4: Systematic Quality Testing

```bash
# Full test across all categories
for category in simple_objects animals humans scenes landscapes; do
  uv run experiments/run_ablation.py \
    --config config.toml \
    --experiment shift_sweep \
    --prompt-category $category \
    --compute-metrics
done

# Aggregate results
# Compare metrics across categories
```

---

## Tips

1. **Start with dry-run** to verify experiment configuration before generating
2. **Use `--max-prompts 3`** for initial testing to save time
3. **Use multiple seeds** (`--seeds 42,123,456`) to reduce variance
4. **Compare metrics relatively** - absolute values vary by prompt complexity
5. **Check the formatted prompt** in web UI to understand token usage
6. **Use history** to quickly compare different parameter combinations

---

## Long Prompt Handling

The Z-Image DiT has a text sequence limit due to RoPE position encoding.

### Token Limits

| Limit | Source | Status |
|-------|--------|--------|
| 1024 | Current implementation | Active |
| 1536 | Model config (axes_lens) | Needs testing |

**Critical Discovery**: The official model config shows `axes_lens=[1536, 512, 512]`, suggesting the model may support **1536 tokens, not 1024**. This needs testing.

### Compression Modes

```bash
# Use via CLI
uv run scripts/generate.py \
  --long-prompt-mode interpolate \
  --model-path /path/to/model \
  "Your very long prompt here..."
```

| Mode | Description | Best For |
|------|-------------|----------|
| `truncate` | Cut off at limit (default) | Safety, predictability |
| `interpolate` | Linear resampling | Minor overflows (1.1-1.5x) |
| `pool` | Adaptive average pooling | Structured content with regions |
| `attention_pool` | Importance-weighted pooling | Preserving key concepts |

### Token Count Guidelines

| Content | Typical Tokens |
|---------|---------------|
| Simple prompt | 10-50 |
| Detailed prompt | 50-200 |
| With template | 100-300 |
| Full format (system + think + assistant) | 150-400 |
| Maximum safe | ~800-900 |

### Strategies for Long Prompts

1. **Omit system prompt** - saves ~50-100 tokens
2. **Skip think block** - saves ~10 tokens
3. **Use concise descriptions** - focus on key visual elements
4. **Check token count** before generation:

```python
from llm_dit import ZImageTextEncoder

encoder = ZImageTextEncoder.from_pretrained("/path/to/model")
output = encoder.encode("Your prompt here")
print(f"Token count: {output.token_counts[0]}/1024")
```

### Testing Long Prompts

```bash
# Compare compression modes
for mode in truncate interpolate pool attention_pool; do
  uv run scripts/generate.py \
    --model-path /path/to/model \
    --long-prompt-mode $mode \
    --seed 42 \
    --output "mode_${mode}.png" \
    "Your very long prompt..."
done
```

See `internal/research/long_prompt_research.md` for detailed research notes and improvement roadmap.

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Move encoder to CPU
--text-encoder-device cpu --dit-device cuda --vae-device cuda

# Or use smaller batch (one image at a time is default)
```

### Slow Generation

```bash
# Reduce steps for quick iteration
--experiment steps_only  # then pick optimal step count

# Use fewer prompts
--max-prompts 5
```

### Metrics Not Computing

```bash
# Install metric dependencies
uv add image-reward transformers

# Check specific errors
uv run python -c "from experiments.metrics import ImageRewardScorer; ImageRewardScorer()"
```
