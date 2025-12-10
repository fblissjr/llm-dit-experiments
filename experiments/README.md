# Experiments

This directory contains tools for running systematic ablation studies and evaluations on the Z-Image pipeline.

## Contents

- **run_ablation.py** - Automated experiment runner with configurable parameters
- **sweep_*.sh** - Priority sweep scripts (recommended starting point)
- **run_all_sweeps.sh** - Run all priority sweeps in sequence
- **prompts/** - Standard evaluation prompts organized by category
- **research/** - Research documentation and study designs
- **metrics/** - Metric computation utilities (ImageReward, SigLIP)
- **results/** - Generated images and experiment logs

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
