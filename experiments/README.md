# Experiments

This directory contains tools for running systematic ablation studies and evaluations on various pipelines and models.

## Contents

- **run_ablation.py** - Automated experiment runner with configurable parameters
- **sweep_*.sh** - Priority sweep scripts (recommended starting point)
- **qwen3_vl/** - Vision conditioning experiments using Qwen3-VL
- **prompts/** - Standard evaluation prompts organized by category
- **research/** - Research documentation and study designs
- **metrics/** - Metric computation utilities (ImageReward, SigLIP)
- **results/** - Generated images and experiment logs

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

## Output Structure

*Updated 2026-01-18: Outputs are now organized by pipeline.*

Results are saved to `experiments/results/{pipeline}/{experiment}_{timestamp}/`:

```
experiments/results/
├── ltx2/                              # LTX-2 video experiments
│   └── layer_ablation_20260115_191023/
│       ├── videos/
│       ├── metadata/
│       └── results.json
├── z_image/                           # Z-Image experiments
│   └── shift_sweep_20251210_143022/
│       ├── images/
│       ├── metadata/
│       └── results.json
└── archive/                           # Superseded/analysis files
```

See [CLAUDE.md](CLAUDE.md) for full output organization documentation.

Legacy flat structure (for older experiments):

```
results/shift_sweep_20251210_143022/
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

## Research Documentation

Located in `experiments/research/`:

- **OVERVIEW.md** - High-level research directions and priorities
- **ablation_studies.md** - Detailed ablation study designs and hypotheses
- **open_questions.md** - Open research questions requiring investigation
- **metrics_and_data.md** - Evaluation methodology and metrics
- **future_directions.md** - Future work and unexplored areas
- **assumptions_to_challenge.md** - Model assumptions and hypotheses to test

## Configuration

Experiments should aim to use the same TOML config files as the main pipeline. CLI arguments override config values.

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
