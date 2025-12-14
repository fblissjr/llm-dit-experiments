#!/bin/bash
# Caption VL Embedding Comparison
#
# Compares VL embeddings (from Qwen3-VL processing caption + source image)
# against text-only embeddings (from Qwen3-4B processing caption).
#
# Research questions:
#   1. Do VL embeddings from caption+image produce better reconstructions?
#   2. Which token mode works best: full, text_only, image_only?
#   3. Which VL hidden layer is optimal: -6, -8, or deeper?
#
# This uses the source image as reference for VL extraction, testing
# whether visual grounding improves caption-based regeneration.
#
# Usage:
#   ./experiments/sweep_caption_vl.sh              # Full run
#   ./experiments/sweep_caption_vl.sh --dry-run    # Preview only
#   ./experiments/sweep_caption_vl.sh --quick      # Quick test

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Default settings
CONFIG="config.toml"
PROFILE="default"
SEEDS="42,123"
PROMPTS="a cat,a mountain landscape,a woman portrait"
VL_HIDDEN_LAYERS="-6,-8,-12"
TOKEN_MODES="full,text_only,image_only"
TARGET_LENGTH="600"
FILL_MODE="content_only"
DRY_RUN=""
OUTPUT_DIR="experiments/results/caption_vl_$(date +%Y%m%d_%H%M%S)"

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
            VL_HIDDEN_LAYERS="-6,-8"
            TOKEN_MODES="full,text_only"
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
        --vl-hidden-layers)
            VL_HIDDEN_LAYERS="$2"
            shift 2
            ;;
        --token-modes)
            TOKEN_MODES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--quick] [--config FILE] [--profile NAME] [--seeds LIST] [--vl-hidden-layers LIST] [--token-modes LIST]"
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
echo "Caption VL Embedding Comparison"
echo "============================================================"
echo "Config: $CONFIG (profile: $PROFILE)"
echo "Prompts: $PROMPTS"
echo "Seeds: $SEEDS"
echo "VL hidden layers: $VL_HIDDEN_LAYERS"
echo "Token modes: $TOKEN_MODES"
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
    --vl-hidden-layers="$VL_HIDDEN_LAYERS" \
    --token-modes="$TOKEN_MODES" \
    --output-dir "$OUTPUT_DIR" \
    $DRY_RUN

if [[ -z "$DRY_RUN" ]]; then
    echo ""
    echo "============================================================"
    echo "Results saved to: $OUTPUT_DIR"
    echo "============================================================"
    echo ""
    echo "Analysis:"
    echo "  - Compare VL outputs vs text-only baseline"
    echo "  - Token modes:"
    echo "      full: all tokens (text + image)"
    echo "      text_only: strips image tokens (VL does nothing)"
    echo "      image_only: visual content only"
    echo "  - Layer -6 typically cleaner than -2 for VL"
    echo "  - Check SSIM in summary.csv for quantitative comparison"
    echo ""
fi
