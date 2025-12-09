#!/bin/bash
# Run All Priority Sweeps
#
# Runs experiments in recommended order:
#   1. Hidden Layer (quick, high value)
#   2. Shift + Steps Grid (quality ceiling)
#   3. Think Block (assumption validation)
#   4. Long Prompt Mode (optional, needs long prompts)
#
# Usage:
#   ./experiments/run_all_sweeps.sh              # Full run (takes hours)
#   ./experiments/run_all_sweeps.sh --quick      # Quick test of each (~30 min)
#   ./experiments/run_all_sweeps.sh --dry-run    # Preview all experiments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

QUICK=""
DRY_RUN=""
SKIP_LONG_PROMPT=""
COMPUTE_METRICS=""

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
        --skip-long-prompt)
            SKIP_LONG_PROMPT="1"
            shift
            ;;
        --metrics)
            COMPUTE_METRICS="--metrics"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            echo "Usage: $0 [--quick] [--dry-run] [--skip-long-prompt] [--metrics]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "Running All Priority Sweeps"
echo "============================================================"
if [[ -n "$QUICK" ]]; then
    echo "Mode: QUICK (reduced prompts and seeds)"
else
    echo "Mode: FULL (this will take several hours)"
fi
if [[ -n "$DRY_RUN" ]]; then
    echo "Dry run: YES (no images will be generated)"
fi
if [[ -n "$COMPUTE_METRICS" ]]; then
    echo "Metrics: ENABLED (ImageReward + SigLIP)"
fi
echo "============================================================"
echo ""

# Priority 1: Hidden Layer
echo ""
echo ">>> PRIORITY 1: Hidden Layer Sweep"
echo ""
bash "$SCRIPT_DIR/sweep_hidden_layer.sh" $QUICK $DRY_RUN $COMPUTE_METRICS

# Priority 2: Shift + Steps Grid
echo ""
echo ">>> PRIORITY 2: Shift + Steps Grid Search"
echo ""
bash "$SCRIPT_DIR/sweep_shift_steps.sh" $QUICK $DRY_RUN $COMPUTE_METRICS

# Priority 3: Think Block
echo ""
echo ">>> PRIORITY 3: Think Block Impact Test"
echo ""
bash "$SCRIPT_DIR/sweep_think_block.sh" $QUICK $DRY_RUN $COMPUTE_METRICS

# Priority 4: Long Prompt Mode (optional)
if [[ -z "$SKIP_LONG_PROMPT" ]]; then
    echo ""
    echo ">>> PRIORITY 4: Long Prompt Mode Test"
    echo ""
    bash "$SCRIPT_DIR/sweep_long_prompt.sh" $QUICK $DRY_RUN $COMPUTE_METRICS
else
    echo ""
    echo ">>> SKIPPING: Long Prompt Mode (--skip-long-prompt)"
    echo ""
fi

echo ""
echo "============================================================"
echo "All sweeps complete!"
echo "============================================================"
echo ""
echo "Results are in experiments/results/*"
echo ""
echo "Recommended analysis order:"
echo "  1. hidden_layer_* - Compare layers, look for quality patterns"
echo "  2. shift_steps_* - Find best quality/speed tradeoff"
echo "  3. think_block_* - Check if thinking content matters"
echo "  4. long_prompt_* - Only if you use >1504 token prompts"
echo ""
