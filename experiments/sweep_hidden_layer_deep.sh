#!/bin/bash
# Deep Hidden Layer Sweep - All 36 Layers
#
# Comprehensive sweep across all transformer layers to find the
# "sweet spot" for prompt adherence. Based on observation that
# middle layers (~-15 to -21) often give best prompt following.
#
# Layers tested: -1 (last) through -35 (earliest usable)
# Samples every 3rd layer for efficiency: -1, -2, -5, -9, -12, -15, -18, -21, -24, -27, -30, -33, -35
#
# Usage:
#   ./experiments/sweep_hidden_layer_deep.sh              # Full run
#   ./experiments/sweep_hidden_layer_deep.sh --dry-run    # Preview only
#   ./experiments/sweep_hidden_layer_deep.sh --quick      # Quick test (3 prompts, 1 seed)
#   ./experiments/sweep_hidden_layer_deep.sh --metrics    # With ImageReward + SigLIP

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Default settings
CONFIG="config.toml"
PROFILE="rtx4090"
SEEDS="42,123"
MAX_PROMPTS=""
PROMPT_CATEGORY="simple_objects"
DRY_RUN=""
COMPUTE_METRICS=""
OUTPUT_DIR="experiments/results/hidden_layer_deep_$(date +%Y%m%d_%H%M%S)"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --quick)
            MAX_PROMPTS="--max-prompts 3"
            SEEDS="42"
            shift
            ;;
        --full)
            SEEDS="42,123,456"
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
        --metrics)
            COMPUTE_METRICS="--compute-metrics"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            echo "Usage: $0 [--quick|--full] [--dry-run] [--metrics] [--config FILE] [--profile NAME] [--seeds LIST] [--category NAME]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "Deep Hidden Layer Sweep (All 36 Layers)"
echo "============================================================"
echo "Config: $CONFIG (profile: $PROFILE)"
echo "Category: $PROMPT_CATEGORY"
echo "Seeds: $SEEDS"
echo "Layers: -1, -2, -5, -9, -12, -15, -18, -21, -24, -27, -30, -33, -35"
echo "Metrics: ${COMPUTE_METRICS:-disabled}"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# Run the experiment
uv run experiments/run_ablation.py \
    --config "$CONFIG" \
    --profile "$PROFILE" \
    --experiment hidden_layer_deep \
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
    echo "Next steps:"
    echo "  1. Review images in $OUTPUT_DIR/"
    echo "  2. Compare early (-35 to -25) vs middle (-24 to -12) vs late (-11 to -1)"
    echo "  3. Look for the 'sweet spot' where prompt adherence peaks"
    echo "  4. Run sweep_middle_layers.sh for fine-grained analysis of best region"
    echo ""
fi
