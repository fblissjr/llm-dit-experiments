#!/bin/bash
# Run All Priority Sweeps
#
# Runs experiments in recommended order:
#   1. Hidden Layer (quick, high value) - basic -1 to -6
#   2. Shift + Steps Grid (quality ceiling)
#   3. Think Block (assumption validation)
#   4. Long Prompt Mode (optional, needs long prompts)
#   5. Deep Layer Sweep (optional, all 36 layers)
#   6. Middle Layer Focus (optional, fine-grained -12 to -24)
#
# Usage:
#   ./experiments/run_all_sweeps.sh              # Full run (takes hours)
#   ./experiments/run_all_sweeps.sh --quick      # Quick test of each (~30 min)
#   ./experiments/run_all_sweeps.sh --dry-run    # Preview all experiments
#   ./experiments/run_all_sweeps.sh --deep       # Include deep layer experiments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

QUICK=""
DRY_RUN=""
SKIP_LONG_PROMPT=""
INCLUDE_DEEP_LAYERS=""
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
        --deep)
            INCLUDE_DEEP_LAYERS="1"
            shift
            ;;
        --metrics)
            COMPUTE_METRICS="--metrics"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            echo "Usage: $0 [--quick] [--dry-run] [--skip-long-prompt] [--deep] [--metrics]"
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
if [[ -n "$INCLUDE_DEEP_LAYERS" ]]; then
    echo "Deep layers: ENABLED (all 36 layers + middle focus)"
fi
echo "============================================================"
echo ""

# Priority 1: Hidden Layer (basic)
echo ""
echo ">>> PRIORITY 1: Hidden Layer Sweep (layers -1 to -6)"
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

# Priority 5-6: Deep Layer Experiments (optional)
if [[ -n "$INCLUDE_DEEP_LAYERS" ]]; then
    echo ""
    echo ">>> PRIORITY 5: Deep Hidden Layer Sweep (all 36 layers)"
    echo ""
    bash "$SCRIPT_DIR/sweep_hidden_layer_deep.sh" $QUICK $DRY_RUN $COMPUTE_METRICS

    echo ""
    echo ">>> PRIORITY 6: Middle Layer Focus + Blending"
    echo ""
    bash "$SCRIPT_DIR/sweep_middle_layers.sh" $QUICK $DRY_RUN $COMPUTE_METRICS
else
    echo ""
    echo ">>> SKIPPING: Deep layer experiments (use --deep to enable)"
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
if [[ -n "$INCLUDE_DEEP_LAYERS" ]]; then
    echo "  5. hidden_layer_deep_* - Find the layer 'sweet spot'"
    echo "  6. middle_layer_* - Fine-tune and test blending"
fi
echo ""
