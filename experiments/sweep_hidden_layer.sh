#!/bin/bash
# Hidden Layer Sweep - Priority 1
#
# Compare embedding extraction from different transformer layers.
# Quick experiment with high potential value.
#
# Layers tested: -1 (last) through -6 (deeper)
# Default is -2 (penultimate) - this tests if that's actually optimal.
#
# Usage:
#   ./experiments/sweep_hidden_layer.sh              # Full run
#   ./experiments/sweep_hidden_layer.sh --dry-run    # Preview only
#   ./experiments/sweep_hidden_layer.sh --quick      # Quick test (3 prompts, 1 seed)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Default settings
CONFIG="config.toml"
PROFILE="rtx4090"
SEEDS="42,123,456"
MAX_PROMPTS=""
PROMPT_CATEGORY="animals"
DRY_RUN=""
OUTPUT_DIR="experiments/results/hidden_layer_$(date +%Y%m%d_%H%M%S)"

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
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "Hidden Layer Sweep"
echo "============================================================"
echo "Config: $CONFIG (profile: $PROFILE)"
echo "Category: $PROMPT_CATEGORY"
echo "Seeds: $SEEDS"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# Run the experiment
uv run experiments/run_ablation.py \
    --config "$CONFIG" \
    --profile "$PROFILE" \
    --experiment hidden_layer \
    --prompt-category "$PROMPT_CATEGORY" \
    --seeds "$SEEDS" \
    --output-dir "$OUTPUT_DIR" \
    $MAX_PROMPTS \
    $DRY_RUN

if [[ -z "$DRY_RUN" ]]; then
    echo ""
    echo "============================================================"
    echo "Results saved to: $OUTPUT_DIR"
    echo "============================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Review images in $OUTPUT_DIR/"
    echo "  2. Compare quality across layers -1 to -6"
    echo "  3. Look for patterns: sharper vs more compositional"
    echo ""
fi
