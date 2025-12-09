# Experiments

This directory contains tools for running systematic ablation studies and evaluations on the Z-Image pipeline.

## Contents

- **run_ablation.py** - Automated experiment runner with configurable parameters
- **prompts/** - Standard evaluation prompts organized by category
- **research/** - Research documentation and study designs
- **metrics/** - Metric computation utilities (ImageReward, SigLIP)
- **results/** - Generated images and experiment logs

## Quick Start

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

## Available Experiments

| Experiment | Description | Variable | Values |
|------------|-------------|----------|--------|
| `shift_sweep` | Sweep shift parameter | shift | 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0 |
| `shift_steps_grid` | Grid search over shift and steps | shift, steps | shift: 2.0-4.0, steps: 6-12 |
| `hidden_layer` | Compare hidden layer extraction | hidden_layer | -1, -2, -3, -4, -5, -6 |
| `think_block` | Test impact of think block content | thinking_content | None, empty, quality, mood, technical |
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
```

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
