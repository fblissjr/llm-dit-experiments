#!/bin/bash
# Early Layer Bias Sweep
#
# Tests early layers with culturally-charged prompts to study pre-training bias.
# Uses the bias_probing prompt category which contains prompts with words like
# "freedom", "revolution", "sacred" that may trigger cultural associations.
#
# Based on observation that early layers show heavy pre-training bias:
# - "freedom" prompts showing Middle East/South Asia imagery
# - "revolution" prompts showing protest imagery
# - etc.
#
# Usage:
#   ./experiments/sweep_early_layer_bias.sh              # Full run
#   ./experiments/sweep_early_layer_bias.sh --dry-run    # Preview only
#   ./experiments/sweep_early_layer_bias.sh --quick      # Quick test (3 prompts, 1 seed)
#   ./experiments/sweep_early_layer_bias.sh --metrics    # With ImageReward + SigLIP

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Default settings
CONFIG="config.toml"
PROFILE="rtx4090"
SEEDS="42,123"
MAX_PROMPTS=""
DRY_RUN=""
COMPUTE_METRICS=""
OUTPUT_DIR="experiments/results/early_layer_bias_$(date +%Y%m%d_%H%M%S)"

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
        --metrics)
            COMPUTE_METRICS="--compute-metrics"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            echo "Usage: $0 [--quick|--full] [--dry-run] [--metrics] [--config FILE] [--profile NAME] [--seeds LIST]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "Early Layer Bias Sweep"
echo "============================================================"
echo "Config: $CONFIG (profile: $PROFILE)"
echo "Category: bias_probing (culturally-charged prompts)"
echo "Seeds: $SEEDS"
echo "Layers: -36, -34, -31, -28, -25, -19, -2 (early to default)"
echo "Metrics: ${COMPUTE_METRICS:-disabled}"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""
echo "This experiment tests if early layers show pre-training bias"
echo "by using prompts with culturally-charged words (freedom,"
echo "revolution, sacred, etc.) that may trigger associations."
echo ""

# Run the experiment
uv run experiments/run_ablation.py \
    --config "$CONFIG" \
    --profile "$PROFILE" \
    --experiment early_layer_bias \
    --prompt-category "bias_probing" \
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
    echo "Analysis guide:"
    echo "  1. Compare early layers (-36 to -28) vs middle (-19) vs default (-2)"
    echo "  2. Look for cultural/political imagery in early layer outputs"
    echo "  3. Check if middle/late layers show literal prompt content"
    echo "  4. Document which prompts show strongest bias effects"
    echo ""
    echo "Key prompts to examine:"
    echo "  - bias_001: 'freedom' (Braveheart) - expect Middle East/South Asia bias"
    echo "  - bias_002: 'revolution' - expect protest imagery"
    echo "  - bias_006: 'resistance' - expect modern conflict"
    echo ""
fi
