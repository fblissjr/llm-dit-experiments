#!/bin/bash
# Caption Fill Mode Comparison
#
# Tests how different fill strategies affect generation quality when
# padding embeddings to a target length.
#
# Research question: Does the DiT care about exact embedding length,
# or only the semantic content? Can we pad to fixed length without
# quality loss?
#
# Fill modes tested:
#   - content_only: No padding, variable length (baseline)
#   - pad_end_zero: Zero padding at end
#   - pad_end_mean: Mean embedding padding at end
#   - pad_middle_zero: Zeros inserted in middle
#   - filler_repeat: Repeat content cyclically
#
# Usage:
#   ./experiments/sweep_caption_fill_modes.sh              # Full run
#   ./experiments/sweep_caption_fill_modes.sh --dry-run    # Preview only
#   ./experiments/sweep_caption_fill_modes.sh --quick      # Quick test

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Default settings
CONFIG="config.toml"
PROFILE="default"
SEEDS="42,123"
PROMPTS="a cat,a mountain landscape,a woman portrait"
TARGET_LENGTH="600"  # Fixed length to compare fill modes
FILL_MODES="content_only,pad_end_zero,pad_end_mean,pad_middle_zero,filler_repeat"
DRY_RUN=""
OUTPUT_DIR="experiments/results/caption_fill_modes_$(date +%Y%m%d_%H%M%S)"

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
            FILL_MODES="content_only,pad_end_zero,filler_repeat"
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
        --target-length)
            TARGET_LENGTH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--quick] [--config FILE] [--profile NAME] [--seeds LIST] [--prompts LIST] [--target-length N]"
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
echo "Caption Fill Mode Comparison"
echo "============================================================"
echo "Config: $CONFIG (profile: $PROFILE)"
echo "Prompts: $PROMPTS"
echo "Seeds: $SEEDS"
echo "Target length: $TARGET_LENGTH"
echo "Fill modes: $FILL_MODES"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# Run the experiment
# Use = syntax for args for consistency
eval uv run experiments/caption_length_study.py \
    --config "$CONFIG" \
    --profile "$PROFILE" \
    --prompts $PROMPT_ARGS \
    --seeds "$SEEDS" \
    --target-lengths="$TARGET_LENGTH" \
    --fill-modes="$FILL_MODES" \
    --output-dir "$OUTPUT_DIR" \
    $DRY_RUN

if [[ -z "$DRY_RUN" ]]; then
    echo ""
    echo "============================================================"
    echo "Results saved to: $OUTPUT_DIR"
    echo "============================================================"
    echo ""
    echo "Analysis:"
    echo "  - Compare grids/{source}_fill_mode_grid.png"
    echo "  - If content_only == pad_end_zero: padding is harmless"
    echo "  - If pad_end_mean > pad_end_zero: DiT prefers 'real' embeddings"
    echo "  - If filler_repeat works: DiT doesn't detect repetition"
    echo ""
fi
