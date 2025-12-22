#!/bin/bash
# Hidden Layer vs CFG Interaction Sweep
#
# Tests whether CFG > 0 helps when using non-default hidden layers with Z-Image.
#
# Background:
#   - Z-Image uses Decoupled-DMD which "bakes in" CFG (normally guidance_scale=0.0)
#   - The distillation was done with layer -2 embeddings
#   - Middle layers (-10 to -18) show better prompt adherence but are OOD
#   - Hypothesis: small CFG (1.5-2.5) might compensate for distribution mismatch
#
# Expected outcomes:
#   - Layer -2 + CFG 0.0: works as intended (trained distribution)
#   - Layer -2 + CFG > 0: probably over-guided/"double CFG" artifacts
#   - Layer -18 + CFG 0.0: works but maybe weaker guidance (OOD)
#   - Layer -18 + CFG > 0: might help compensate for OOD guidance loss
#
# Usage:
#   ./experiments/sweep_hidden_layer_cfg.sh              # Full grid (35 combos x prompts x seeds)
#   ./experiments/sweep_hidden_layer_cfg.sh --quick      # Quick grid (9 combos, 3 prompts, 1 seed)
#   ./experiments/sweep_hidden_layer_cfg.sh --dry-run    # Preview only
#   ./experiments/sweep_hidden_layer_cfg.sh --with-shift # Include shift parameter (27 combos)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Default settings
CONFIG="config.toml"
PROFILE="rtx4090"
SEEDS="42,123,456,789"
MAX_PROMPTS=""
PROMPT_CATEGORY="animals"
DRY_RUN=""
COMPUTE_METRICS="--compute-metrics"
EXPERIMENT="hidden_layer_cfg_grid"
OUTPUT_DIR="experiments/results/layer_cfg_$(date +%Y%m%d_%H%M%S)"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --quick)
            EXPERIMENT="hidden_layer_cfg_quick"
            MAX_PROMPTS="--max-prompts 3"
            SEEDS="42"
            shift
            ;;
        --with-shift)
            EXPERIMENT="hidden_layer_cfg_shift_grid"
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --category)
            PROMPT_CATEGORY="$2"
            shift 2
            ;;
        --no-metrics)
            COMPUTE_METRICS=""
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --quick       Quick test (9 combos, 3 prompts, 1 seed)"
            echo "  --with-shift  Include shift parameter sweep (27 combos)"
            echo "  --dry-run     Preview only, don't generate"
            echo "  --config      Config file (default: config.toml)"
            echo "  --profile     Config profile (default: rtx4090)"
            echo "  --seeds       Comma-separated seeds (default: 42,123,456,789)"
            echo "  --category    Prompt category (default: animals)"
            echo "  --no-metrics  Skip SigLIP/ImageReward computation"
            echo "  --output-dir  Custom output directory"
            exit 1
            ;;
    esac
done

# Calculate expected generations
case $EXPERIMENT in
    hidden_layer_cfg_grid)
        COMBOS="35 (5 layers x 7 CFG values)"
        ;;
    hidden_layer_cfg_quick)
        COMBOS="9 (3 layers x 3 CFG values)"
        ;;
    hidden_layer_cfg_shift_grid)
        COMBOS="27 (3 layers x 3 CFG x 3 shift)"
        ;;
esac

echo "============================================================"
echo "Hidden Layer vs CFG Interaction Sweep"
echo "============================================================"
echo "Experiment: $EXPERIMENT"
echo "Config: $CONFIG (profile: $PROFILE)"
echo "Category: $PROMPT_CATEGORY"
echo "Seeds: $SEEDS"
echo "Grid combos: $COMBOS"
echo "Metrics: ${COMPUTE_METRICS:-disabled}"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""
echo "Hypothesis being tested:"
echo "  - Layer -2 (default) should work best with CFG=0.0"
echo "  - Middle layers (-14, -18) may benefit from small CFG (1.5-2.5)"
echo "  - CFG > 2.5 likely causes over-saturation at any layer"
echo ""
echo "============================================================"
echo ""

# Run the experiment
uv run experiments/run_ablation.py \
    --config "$CONFIG" \
    --profile "$PROFILE" \
    --experiment "$EXPERIMENT" \
    --prompt-category "$PROMPT_CATEGORY" \
    --seeds "$SEEDS" \
    --output-dir "$OUTPUT_DIR" \
    $MAX_PROMPTS \
    $DRY_RUN \
    $COMPUTE_METRICS

if [[ -z "$DRY_RUN" ]]; then
    echo ""
    echo "============================================================"
    echo "Results saved to: $OUTPUT_DIR"
    echo "============================================================"
    echo ""
    echo "Analysis tips:"
    echo "  1. Compare SigLIP scores across the grid"
    echo "  2. Look for patterns: does optimal CFG vary by layer?"
    echo "  3. Check for over-saturation at high CFG values"
    echo "  4. If middle layers + small CFG > layer -2 + CFG 0, hypothesis confirmed"
    echo ""
    echo "Generate comparison grid:"
    echo "  uv run experiments/compare.py --mode grid --experiment $OUTPUT_DIR"
    echo ""
fi
