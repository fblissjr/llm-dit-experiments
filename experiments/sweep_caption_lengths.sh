#!/bin/bash
# Caption Length Sweep
#
# Tests how embedding sequence length affects generation quality.
# Uses Qwen3-VL to generate detailed captions, then tests regeneration
# at various target lengths.
#
# Research question: Does the DiT perform better with longer, more
# detailed embeddings? Or is there a sweet spot?
#
# Target lengths tested: 50, 150, 300, 600, 1000, 1504 (max)
#
# Usage:
#   ./experiments/sweep_caption_lengths.sh              # Full run
#   ./experiments/sweep_caption_lengths.sh --dry-run    # Preview only
#   ./experiments/sweep_caption_lengths.sh --quick      # Quick test

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Default settings
CONFIG="config.toml"
PROFILE="default"
SEEDS="42,123"
PROMPTS="a cat,a mountain landscape,a woman portrait"
TARGET_LENGTHS="50,150,300,600,1000,1504"
FILL_MODE="content_only"  # No padding, test pure length impact
DRY_RUN=""
OUTPUT_DIR="experiments/results/caption_lengths_$(date +%Y%m%d_%H%M%S)"

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
            TARGET_LENGTHS="50,300,1000"
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
        --target-lengths)
            TARGET_LENGTHS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--quick] [--config FILE] [--profile NAME] [--seeds LIST] [--prompts LIST] [--target-lengths LIST]"
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
echo "Caption Length Sweep"
echo "============================================================"
echo "Config: $CONFIG (profile: $PROFILE)"
echo "Prompts: $PROMPTS"
echo "Seeds: $SEEDS"
echo "Target lengths: $TARGET_LENGTHS"
echo "Fill mode: $FILL_MODE (no padding)"
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
    --target-lengths="$TARGET_LENGTHS" \
    --fill-modes="$FILL_MODE" \
    --output-dir "$OUTPUT_DIR" \
    $DRY_RUN

if [[ -z "$DRY_RUN" ]]; then
    echo ""
    echo "============================================================"
    echo "Results saved to: $OUTPUT_DIR"
    echo "============================================================"
    echo ""
    echo "Analysis:"
    echo "  - Compare grids/{source}_length_grid.png"
    echo "  - Check SSIM scores in summary.csv"
    echo "  - Look for quality changes at different lengths"
    echo "  - Identify optimal length for detail vs efficiency"
    echo ""
fi
