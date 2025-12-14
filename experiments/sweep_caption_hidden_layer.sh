#!/bin/bash
# Caption Hidden Layer Sweep
#
# Tests which Qwen3-4B hidden layer produces best embeddings for
# caption-based regeneration.
#
# Research question: Does the optimal layer change when using
# detailed captions vs simple prompts? SFT may have overwritten
# later layers for instruction-following, making middle layers
# better for visual semantics.
#
# Layers tested: -2 (default), -6, -8, -12, -16, -21
#
# Usage:
#   ./experiments/sweep_caption_hidden_layer.sh              # Full run
#   ./experiments/sweep_caption_hidden_layer.sh --dry-run    # Preview only
#   ./experiments/sweep_caption_hidden_layer.sh --quick      # Quick test

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Default settings
CONFIG="config.toml"
PROFILE="default"
SEEDS="42,123"
PROMPTS="a cat,a mountain landscape,a woman portrait"
HIDDEN_LAYERS="-2,-6,-8,-12,-16,-21"
TARGET_LENGTH="600"  # Fixed length to isolate layer effect
FILL_MODE="content_only"
DRY_RUN=""
OUTPUT_DIR="experiments/results/caption_hidden_layer_$(date +%Y%m%d_%H%M%S)"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --quick)
            SEEDS="42"
            PROMPTS="a cat"
            HIDDEN_LAYERS="-2,-6,-12"
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
        --prompts)
            PROMPTS="$2"
            shift 2
            ;;
        --hidden-layers)
            HIDDEN_LAYERS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--quick] [--config FILE] [--profile NAME] [--seeds LIST] [--prompts LIST] [--hidden-layers LIST]"
            exit 1
            ;;
    esac
done

# Convert comma-separated prompts to array for passing to Python
IFS=',' read -ra PROMPT_ARRAY <<< "$PROMPTS"
PROMPT_ARGS=""
for p in "${PROMPT_ARRAY[@]}"; do
    PROMPT_ARGS="$PROMPT_ARGS \"$p\""
done

echo "============================================================"
echo "Caption Hidden Layer Sweep"
echo "============================================================"
echo "Config: $CONFIG (profile: $PROFILE)"
echo "Prompts: $PROMPTS"
echo "Seeds: $SEEDS"
echo "Hidden layers: $HIDDEN_LAYERS"
echo "Target length: $TARGET_LENGTH"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# Run the experiment
# Use = syntax for args that may contain negative numbers
eval uv run experiments/caption_length_study.py \
    --config "$CONFIG" \
    --profile "$PROFILE" \
    --prompts $PROMPT_ARGS \
    --seeds "$SEEDS" \
    --target-lengths="$TARGET_LENGTH" \
    --fill-modes="$FILL_MODE" \
    --hidden-layers="$HIDDEN_LAYERS" \
    --output-dir "$OUTPUT_DIR" \
    $DRY_RUN

if [[ -z "$DRY_RUN" ]]; then
    echo ""
    echo "============================================================"
    echo "Results saved to: $OUTPUT_DIR"
    echo "============================================================"
    echo ""
    echo "Analysis:"
    echo "  - Compare grids/{source}_layer_grid.png"
    echo "  - Later layers (-2): may be abstract, SFT-influenced"
    echo "  - Middle layers (-12 to -21): may retain visual details"
    echo "  - Check if optimal layer differs from simple prompt default"
    echo ""
fi
