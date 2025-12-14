# Experiments

This directory contains tools for running systematic ablation studies and evaluations on the Z-Image pipeline.

## Contents

- **run_ablation.py** - Automated experiment runner with configurable parameters
- **caption_length_study.py** - Caption-based embedding length and fill mode experiments
- **compare_embedding_models.py** - Compare Qwen3-4B vs Qwen3-Embedding-4B
- **sweep_*.sh** - Priority sweep scripts (recommended starting point)
- **sweep_caption_*.sh** - Caption length study sweep scripts
- **sweep_embedding_*.sh** - Embedding model comparison sweeps
- **run_all_sweeps.sh** - Run all priority sweeps in sequence
- **run_all_caption_sweeps.sh** - Run all caption length study sweeps
- **run_all_embedding_sweeps.sh** - Run all embedding model comparison sweeps
- **qwen3_vl/** - Vision conditioning experiments using Qwen3-VL
- **prompts/** - Standard evaluation prompts organized by category
- **research/** - Research documentation and study designs
- **metrics/** - Metric computation utilities (ImageReward, SigLIP)
- **results/** - Generated images and experiment logs

## Qwen3-VL Vision Conditioning (NEW)

Zero-shot vision-conditioned generation using Qwen3-VL embeddings. See [qwen3_vl/README.md](qwen3_vl/README.md) for full documentation.

**Key Discovery**: Qwen3-VL's text model hidden states (after processing an image) can condition Z-Image because both use Qwen3-4B architecture (hidden_size=2560).

```bash
# Extract embeddings from reference image
uv run experiments/qwen3_vl/extract_embeddings.py \
    --image reference.png \
    --output vl_embeddings.pt

# Generate with vision conditioning (30% VL + 70% text)
uv run experiments/qwen3_vl/blend_and_generate.py \
    --vl-embeddings vl_embeddings.pt \
    --prompt "Your text prompt" \
    --alpha 0.3 \
    --output result.png

# Run comprehensive comparison
uv run experiments/qwen3_vl/run_comparison.py \
    --image reference.png \
    --prompt "Your text prompt" \
    --experiment alpha_sweep \
    --output-dir results/vl_experiment/
```

**Use Cases**:
- Style transfer (low alpha ~0.2)
- Image variations (medium alpha ~0.5)
- Composition guidance (alpha ~0.3, image tokens only)

See also:
- [qwen3_vl/CONDITIONING_GUIDE.md](qwen3_vl/CONDITIONING_GUIDE.md) - All control parameters
- [qwen3_vl/RESEARCH_FINDINGS.md](qwen3_vl/RESEARCH_FINDINGS.md) - Detailed findings and comparison with IP-Adapter

## Qwen3-Embedding-4B Comparison (NEW)

Compare Qwen3-4B (baseline) vs Qwen3-Embedding-4B as text encoders for Z-Image.

**Research Question:** Does an embedding-optimized model produce better results for Z-Image than the base Qwen3-4B?

**Key Insight:** Qwen3-Embedding-4B has identical architecture to Qwen3-4B:
- hidden_size: 2560 (matches Z-Image requirement)
- 36 layers (same as Qwen3-4B)
- Specifically trained for embedding quality via contrastive learning
- Supports instruction-aware encoding

**Sweep Scripts:**

| Script | Research Question |
|--------|-------------------|
| `sweep_embedding_layers.sh` | Which layer produces most similar embeddings? |
| `sweep_embedding_instructions.sh` | Does instruction prefix improve results? |
| `sweep_embedding_generation.sh` | Do embeddings produce better images? |
| `run_all_embedding_sweeps.sh` | Run all sweeps in sequence |

```bash
# Quick comparison (stats only)
uv run experiments/compare_embedding_models.py \
    --qwen3-path /path/to/Qwen3-4B \
    --embedding-path /path/to/Qwen3-Embedding-4B \
    --prompts "A cat" "A mountain landscape"

# Run all sweeps
./experiments/run_all_embedding_sweeps.sh --quick

# Test with instructions
./experiments/sweep_embedding_instructions.sh
```

**Using the Embedding Extractor:**

```python
from llm_dit.embedding import EmbeddingExtractor

extractor = EmbeddingExtractor.from_pretrained("/path/to/Qwen3-Embedding-4B")

# For Z-Image (full sequence extraction)
embeddings = extractor.encode_for_zimage("A cat sleeping in sunlight")

# With instruction prefix (experimental)
result = extractor.extract(
    "A cat sleeping",
    instruction="Generate an embedding for text-to-image synthesis",
)
```

## Caption Length Study (NEW)

Tests how Z-Image DiT responds to different embedding sequence lengths and fill strategies.

**Research Question:** Does the DiT care about exact embedding length, or only content? Can we pad/fill embeddings to reach a target length without quality degradation?

**Workflow:**
1. Generate source images from simple prompts (e.g., "A cat")
2. Caption each with Qwen3-VL to get detailed descriptions
3. Test various target embedding lengths with different fill strategies
4. Generate comparison grids and compute SSIM metrics vs source

```bash
# Quick test with single prompt
uv run experiments/caption_length_study.py \
    --config config.toml \
    --prompts "A cat" \
    --seeds 42 \
    --target-lengths "50,300,600" \
    --fill-modes "content_only,pad_end_zero" \
    --dry-run  # Preview what would run

# Full study
uv run experiments/caption_length_study.py \
    --config config.toml \
    --prompts "A cat" "A mountain landscape" "A woman portrait" \
    --seeds 42,123 \
    --output-dir experiments/results/caption_length_study
```

**Key Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--target-lengths` | Final embedding lengths to test | 50,150,300,600,1000,1504 |
| `--fill-modes` | How to reach target length | content_only, pad_end_zero, pad_end_mean, pad_middle_zero, filler_repeat |
| `--compression-modes` | For captions >1504 tokens | truncate, interpolate, pool, attention_pool |
| `--hidden-layers` | Hidden layers for text encoding (Qwen3-4B) | -2 (from config) |
| `--vl-hidden-layers` | Hidden layers for VL extraction (enables VL mode) | None |
| `--token-modes` | Token modes for VL: full, text_only, image_only, image_no_markers | full |
| `--use-vl-embeddings` | Use VL embeddings instead of text encoding | False |

**Hidden Layer Selection:**

```bash
# Sweep text encoder hidden layers (Qwen3-4B)
uv run experiments/caption_length_study.py \
    --config config.toml \
    --prompts "A cat" \
    --hidden-layers="-2,-6,-8,-16" \
    --target-lengths "300" \
    --fill-modes "content_only"

# Use VL embeddings with layer + token mode sweep
uv run experiments/caption_length_study.py \
    --config config.toml \
    --prompts "A cat" \
    --vl-hidden-layers="-6,-8" \
    --token-modes="full,text_only,image_only" \
    --target-lengths "300" \
    --fill-modes "content_only"
```

**Fill Modes Explained:**

| Mode | Description | Final Length |
|------|-------------|--------------|
| `content_only` | Truncate caption to target, no padding | min(caption_len, target) |
| `pad_end_zero` | Caption + zero-embedding padding | Exactly target |
| `pad_end_mean` | Caption + mean-embedding padding | Exactly target |
| `pad_middle_zero` | Caption split in half, zeros in middle | Exactly target |
| `filler_repeat` | Repeat caption tokens cyclically | Exactly target |

**Output:**
- `source/` - Original images and Qwen3-VL captions
- `regenerated/` - Images at each length/fill/layer combo
- `grids/` - Comparison grids (length grid, fill mode grid, layer grid)
- `results.json` - Full experiment metadata with hidden layer info
- `summary.csv` - SSIM scores and timing

## Quick Start - Sweep Scripts (Recommended)

The sweep scripts run experiments in priority order with sensible defaults:

```bash
# Priority 1: Hidden Layer (quick, high value)
./experiments/sweep_hidden_layer.sh --quick    # ~5 min test run
./experiments/sweep_hidden_layer.sh            # Full run

# Priority 2: Shift + Steps Grid (quality ceiling)
./experiments/sweep_shift_steps.sh --quick     # ~10 min test run
./experiments/sweep_shift_steps.sh             # Full run

# Priority 3: Think Block Impact
./experiments/sweep_think_block.sh --quick     # ~5 min test run
./experiments/sweep_think_block.sh             # Full run

# Priority 4: Long Prompt Modes (only for >1504 token prompts)
./experiments/sweep_long_prompt.sh --quick     # ~5 min test run

# Priority 5: Hidden Layer Blend
./experiments/sweep_hidden_layer_blend.sh --quick    # ~20 min test run
./experiments/sweep_hidden_layer_blend.sh            # Full run

# Run all sweeps
./experiments/run_all_sweeps.sh --quick        # Quick test of all (~30 min)
./experiments/run_all_sweeps.sh                # Full run (several hours)

# Preview any sweep without generating
./experiments/sweep_hidden_layer.sh --dry-run
```

### Sweep Script Options

All sweep scripts support:

| Flag | Description |
|------|-------------|
| `--quick` | Reduced prompts and seeds for fast testing |
| `--dry-run` | Preview what would be generated |
| `--config FILE` | Use specific config file (default: config.toml) |
| `--profile NAME` | Use specific profile (default: rtx4090) |
| `--seeds LIST` | Comma-separated seeds (default varies by script) |
| `--category NAME` | Prompt category to use |

### Caption Length Study Sweeps

Dedicated sweep scripts for the caption length study:

```bash
# Fill mode comparison - test padding strategies
./experiments/sweep_caption_fill_modes.sh --quick --dry-run
./experiments/sweep_caption_fill_modes.sh              # Full run

# Length sweep - test embedding sequence lengths
./experiments/sweep_caption_lengths.sh --quick --dry-run
./experiments/sweep_caption_lengths.sh                 # Full run

# Hidden layer sweep - test Qwen3-4B layers
./experiments/sweep_caption_hidden_layer.sh --quick --dry-run
./experiments/sweep_caption_hidden_layer.sh            # Full run

# VL embedding comparison - test VL vs text encoding
./experiments/sweep_caption_vl.sh --quick --dry-run
./experiments/sweep_caption_vl.sh                      # Full run

# Run all caption sweeps
./experiments/run_all_caption_sweeps.sh --quick        # Quick test
./experiments/run_all_caption_sweeps.sh                # Full run
./experiments/run_all_caption_sweeps.sh --skip-vl      # Skip VL (no Qwen3-VL)
```

| Script | Research Question |
|--------|-------------------|
| `sweep_caption_fill_modes.sh` | Does padding strategy affect quality? |
| `sweep_caption_lengths.sh` | Is there an optimal embedding length? |
| `sweep_caption_hidden_layer.sh` | Which Qwen3-4B layer works best for captions? |
| `sweep_caption_vl.sh` | Do VL embeddings improve reconstruction? |

## Direct Usage with run_ablation.py

For more control, use the underlying Python script directly:

```bash
# List available experiments
uv run experiments/run_ablation.py --list-experiments

# Run with config file (recommended)
uv run experiments/run_ablation.py --config config.toml --experiment shift_sweep

# Dry run to preview what would be generated
uv run experiments/run_ablation.py --config config.toml --experiment shift_sweep --dry-run

# Run with specific prompts
uv run experiments/run_ablation.py \
  --config config.toml \
  --experiment shift_sweep \
  --prompt-ids animal_001,simple_002

# Run with metrics computation
uv run experiments/run_ablation.py \
  --config config.toml \
  --experiment shift_sweep \
  --compute-metrics
```

## Default Settings (DiffSynth-Compatible)

The experiment runner uses DiffSynth-compatible defaults:

| Setting | Default | Notes |
|---------|---------|-------|
| `force_think_block` | `True` | Empty `<think>\n\n</think>\n\n` block (model trained with this) |
| `long_prompt_mode` | `interpolate` | Smooth resampling for prompts >1504 tokens |
| `hidden_layer` | `-2` | Penultimate layer (DiffSynth default) |
| `shift` | `3.0` | FlowMatch shift (DiffSynth default) |
| `steps` | `9` | Turbo model optimal |

These match the official DiffSynth-Studio implementation. The `think_block` experiment specifically tests whether deviating from these defaults (e.g., no think block) affects quality.

## Available Experiments

| Experiment | Description | Variable | Values |
|------------|-------------|----------|--------|
| `shift_sweep` | Sweep shift parameter | shift | 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 |
| `shift_steps_grid` | Grid search over shift and steps | shift, steps | shift: 2.0-4.0, steps: 6-15 |
| `hidden_layer` | Compare hidden layer extraction | hidden_layer | -1, -2, -3, -4, -5, -6 |
| `hidden_layer_blend` | Blend embeddings from multiple layers | layer_weights | single layers, 2-layer blends, 3-layer blends |
| `think_block` | Test think block impact (DiffSynth uses empty) | thinking_content | empty (default), None, quality, mood, technical |
| `system_prompt` | Test impact of system prompts | system_prompt | None, photographer, painter, illustrator, generic |
| `steps_only` | Test different step counts | steps | 4, 6, 8, 9, 10, 12, 15, 20 |
| `long_prompt_mode` | Compare compression modes for long prompts | long_prompt_mode | truncate, interpolate, pool, attention_pool |

## Output Structure

Results are saved to `experiments/results/<experiment>/`:

```
results/shift_sweep/
├── metadata.json              # Experiment configuration
├── results.csv                # Tabular results with metrics
├── animal_001_shift_1.0.png   # Generated images
├── animal_001_shift_2.0.png
└── ...
```

**metadata.json** contains:
- Experiment name and description
- Model path and device placement
- Variable name and values
- Generation parameters (steps, guidance scale, etc.)
- Timestamp and system info

**results.csv** contains:
- prompt_id, prompt_text
- variable_name, variable_value
- Generation time, token count
- Optional metrics (ImageReward, SigLIP)
- Output file paths

## Comparison Tools

After running experiments, use the comparison tools to analyze results visually.

### CLI Comparison (compare.py)

```bash
# List all experiments
uv run experiments/compare.py --list

# Show experiment details
uv run experiments/compare.py -e shift_sweep --info

# Generate grid (prompts x variable values)
uv run experiments/compare.py -e shift_sweep --mode grid -o grid.png

# Side-by-side comparison
uv run experiments/compare.py -e hidden_layer --mode side-by-side \
    --values '-1,-2' --prompt animal_001

# Diff overlay (highlight/absolute/heatmap)
uv run experiments/compare.py -e think_block --mode diff \
    --values ',None' --prompt animal_001 --diff-mode highlight
```

### Comparison Modes

| Mode | Description | Output |
|------|-------------|--------|
| `grid` | NxM grid of prompts x variable values | Single composite image |
| `side-by-side` | Two images placed horizontally | Single composite image |
| `diff` | Pixel difference overlay | Highlight/absolute/heatmap visualization |

### CLI Options

```bash
# Required
-e, --experiment EXPERIMENT   # Experiment name

# Modes
--mode MODE                   # grid, side-by-side, diff
--list                        # List available experiments
--info                        # Show experiment details

# Filtering
--prompt PROMPT               # Filter to specific prompt ID
--values VALUES               # Comma-separated variable values
--seed SEED                   # Filter to specific seed

# Output
-o, --output FILE             # Output image path

# Diff options
--diff-mode MODE              # highlight (default), absolute, heatmap
```

### Web Viewer (Interactive)

The web viewer provides interactive comparison with 4 visualization modes:

```bash
# Start viewer on port 7861
uv run experiments/viewer/server.py

# Open http://localhost:7861
```

**Features:**
- Auto-discovers experiments from `experiments/results/`
- Grid View - NxM grid of prompts x variable values
- Slider - Draggable divider between two images
- A/B Toggle - Click to swap between images
- Diff Overlay - Highlight/absolute/heatmap pixel differences

**Typical workflow:**
1. Run experiments using sweep scripts
2. Start web viewer to browse results
3. Use grid view to get overview of all variations
4. Use slider/A/B toggle for detailed pairwise comparison
5. Use diff overlay to identify pixel-level changes

## Standard Prompts

Standard evaluation prompts are located in `experiments/prompts/`:

- `animals.json` - Animal subjects with varying complexity
- `simple.json` - Simple compositions (single subject, minimal context)
- `complex.json` - Complex scenes (multiple subjects, detailed environments)
- `styles.json` - Artistic styles and rendering modes
- `technical.json` - Technical challenges (lighting, perspective, materials)

Each prompt file contains structured entries:

```json
{
  "animal_001": {
    "text": "A tabby cat sleeping on a windowsill",
    "category": "animals",
    "complexity": "simple",
    "subjects": ["cat"],
    "environment": "indoor"
  }
}
```

## Research Documentation

Located in `experiments/research/`:

- **OVERVIEW.md** - High-level research directions and priorities
- **ablation_studies.md** - Detailed ablation study designs and hypotheses
- **open_questions.md** - Open research questions requiring investigation
- **metrics_and_data.md** - Evaluation methodology and metrics
- **future_directions.md** - Future work and unexplored areas
- **assumptions_to_challenge.md** - Model assumptions and hypotheses to test

## Configuration

Experiments use the same TOML config files as the main pipeline. CLI arguments override config values.

```toml
[default]
model_path = "/path/to/z-image-turbo"

[default.encoder]
device = "cpu"
torch_dtype = "bfloat16"

[default.pipeline]
device = "cuda"

[default.generation]
width = 1024
height = 1024
steps = 9
enable_thinking = true  # Empty think block (DiffSynth default)

[default.pytorch]
long_prompt_mode = "interpolate"  # Smooth resampling for >1504 tokens
```

Note: `enable_thinking = true` adds an empty `<think>\n\n</think>\n\n` block to match DiffSynth training format.

## Common Options

```bash
# Experiment selection
--experiment EXPERIMENT       # Experiment name (required)
--list-experiments           # List available experiments

# Prompts
--prompt-ids IDS             # Comma-separated prompt IDs
--prompt-category CATEGORY   # Filter by category (animals, simple, etc.)
--prompt "TEXT"              # Single custom prompt

# Output
--output-dir DIR             # Results directory (default: results/<experiment>)
--dry-run                    # Preview without generating

# Metrics
--compute-metrics            # Enable ImageReward and SigLIP scoring

# Config
--config FILE                # TOML config file
--profile PROFILE            # Config profile (default: "default")
```

## Examples

**Run shift sweep with default prompts:**
```bash
uv run experiments/run_ablation.py \
  --config config.toml \
  --experiment shift_sweep
```

**Test hidden layers on animal prompts only:**
```bash
uv run experiments/run_ablation.py \
  --config config.toml \
  --experiment hidden_layer \
  --prompt-category animals
```

**Grid search with metrics:**
```bash
uv run experiments/run_ablation.py \
  --config config.toml \
  --experiment shift_steps_grid \
  --compute-metrics
```

**Custom single prompt test:**
```bash
uv run experiments/run_ablation.py \
  --config config.toml \
  --experiment think_block \
  --prompt "A woman in a red dress standing in a field"
```
