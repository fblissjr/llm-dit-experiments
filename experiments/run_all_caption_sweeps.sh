#!/bin/bash
# Run All Caption Length Study Sweeps
#
# Runs all caption-related experiments in sequence:
#   1. Fill modes - Test padding strategies
#   2. Lengths - Test embedding sequence lengths
#   3. Hidden layers - Test Qwen3-4B layers for text encoding
#   4. VL embeddings - Test VL vs text encoding
#
# Usage:
#   ./experiments/run_all_caption_sweeps.sh              # Full run
#   ./experiments/run_all_caption_sweeps.sh --dry-run    # Preview all
#   ./experiments/run_all_caption_sweeps.sh --quick      # Quick test of all

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Default settings
DRY_RUN=""
QUICK=""
CONFIG="config.toml"
PROFILE="default"
SKIP_VL=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --quick)
            QUICK="--quick"
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
        --skip-vl)
            SKIP_VL="true"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--quick] [--config FILE] [--profile NAME] [--skip-vl]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "Running All Caption Length Study Sweeps"
echo "============================================================"
echo "Config: $CONFIG (profile: $PROFILE)"
echo "Mode: ${QUICK:-full} ${DRY_RUN}"
echo "Skip VL: ${SKIP_VL:-no}"
echo "============================================================"
echo ""

COMMON_ARGS="--config $CONFIG --profile $PROFILE $DRY_RUN $QUICK"

# Track timing
START_TIME=$(date +%s)

# 1. Fill Modes
echo ""
echo ">>> [1/4] Fill Mode Comparison"
echo "============================================================"
./experiments/sweep_caption_fill_modes.sh $COMMON_ARGS

# 2. Lengths
echo ""
echo ">>> [2/4] Length Sweep"
echo "============================================================"
./experiments/sweep_caption_lengths.sh $COMMON_ARGS

# 3. Hidden Layers
echo ""
echo ">>> [3/4] Hidden Layer Sweep"
echo "============================================================"
./experiments/sweep_caption_hidden_layer.sh $COMMON_ARGS

# 4. VL Embeddings (optional, requires Qwen3-VL)
if [[ -z "$SKIP_VL" ]]; then
    echo ""
    echo ">>> [4/4] VL Embedding Comparison"
    echo "============================================================"
    ./experiments/sweep_caption_vl.sh $COMMON_ARGS
else
    echo ""
    echo ">>> [4/4] VL Embedding Comparison - SKIPPED"
    echo "============================================================"
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "All Caption Sweeps Complete"
echo "============================================================"
echo "Total time: ${DURATION}s"
echo ""
echo "Results saved to experiments/results/caption_*"
echo ""
echo "Next steps:"
echo "  1. Review comparison grids in each results directory"
echo "  2. Compare SSIM scores in summary.csv files"
echo "  3. Identify optimal settings for caption-based generation"
echo ""
