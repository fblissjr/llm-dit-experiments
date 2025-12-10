#!/bin/bash
# Middle Layer Sweep - Fine-grained + Blending
#
# Two-part experiment focusing on middle layers where prompt adherence peaks:
#   Part 1: Fine-grained sweep of layers -12 to -24 (every 2nd layer)
#   Part 2: Middle layer blending experiments
#
# Based on observation that middle layers give best prompt adherence while
# early layers are too raw/syntactic and late layers are over-abstracted.
#
# Usage:
#   ./experiments/sweep_middle_layers.sh              # Full run (both parts)
#   ./experiments/sweep_middle_layers.sh --dry-run    # Preview only
#   ./experiments/sweep_middle_layers.sh --quick      # Quick test (3 prompts, 1 seed)
#   ./experiments/sweep_middle_layers.sh --metrics    # With ImageReward + SigLIP
#   ./experiments/sweep_middle_layers.sh --focus-only # Only fine-grained sweep (skip blends)
#   ./experiments/sweep_middle_layers.sh --blend-only # Only blending experiments

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
RUN_FOCUS="1"
RUN_BLEND="1"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

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
        --focus-only)
            RUN_BLEND=""
            shift
            ;;
        --blend-only)
            RUN_FOCUS=""
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            echo "Usage: $0 [--quick|--full] [--dry-run] [--metrics] [--focus-only|--blend-only]"
            echo "       [--config FILE] [--profile NAME] [--seeds LIST] [--category NAME]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "Middle Layer Experiments"
echo "============================================================"
echo "Config: $CONFIG (profile: $PROFILE)"
echo "Category: $PROMPT_CATEGORY"
echo "Seeds: $SEEDS"
echo "Metrics: ${COMPUTE_METRICS:-disabled}"
if [[ -n "$RUN_FOCUS" ]]; then
    echo "Part 1: Fine-grained sweep (-12 to -24)"
fi
if [[ -n "$RUN_BLEND" ]]; then
    echo "Part 2: Middle layer blending"
fi
echo "============================================================"
echo ""

# Part 1: Fine-grained middle layer sweep
if [[ -n "$RUN_FOCUS" ]]; then
    OUTPUT_DIR="experiments/results/middle_layer_focus_${TIMESTAMP}"

    echo ""
    echo ">>> Part 1: Fine-grained Middle Layer Sweep"
    echo ">>> Layers: -12, -14, -16, -18, -20, -22, -24"
    echo ">>> Output: $OUTPUT_DIR"
    echo ""

    uv run experiments/run_ablation.py \
        --config "$CONFIG" \
        --profile "$PROFILE" \
        --experiment hidden_layer_middle_focus \
        --prompt-category "$PROMPT_CATEGORY" \
        --seeds "$SEEDS" \
        --output-dir "$OUTPUT_DIR" \
        $MAX_PROMPTS \
        $DRY_RUN \
        $COMPUTE_METRICS
fi

# Part 2: Middle layer blending
if [[ -n "$RUN_BLEND" ]]; then
    OUTPUT_DIR="experiments/results/middle_layer_blend_${TIMESTAMP}"

    echo ""
    echo ">>> Part 2: Middle Layer Blending"
    echo ">>> Testing blends of middle layers and middle+late combinations"
    echo ">>> Output: $OUTPUT_DIR"
    echo ""

    uv run experiments/run_ablation.py \
        --config "$CONFIG" \
        --profile "$PROFILE" \
        --experiment middle_layer_blend \
        --prompt-category "$PROMPT_CATEGORY" \
        --seeds "$SEEDS" \
        --output-dir "$OUTPUT_DIR" \
        $MAX_PROMPTS \
        $DRY_RUN \
        $COMPUTE_METRICS
fi

if [[ -z "$DRY_RUN" ]]; then
    echo ""
    echo "============================================================"
    echo "Middle Layer Experiments Complete"
    echo "============================================================"
    echo ""
    echo "Results saved to experiments/results/middle_layer_*_${TIMESTAMP}/"
    echo ""
    echo "Next steps:"
    echo "  1. Compare fine-grained sweep to find optimal single layer"
    echo "  2. Check if blending middle layers improves over single best"
    echo "  3. Test middle+late blends (e.g., -18 @ 70% + -2 @ 30%)"
    echo "  4. Update default hidden_layer in config.toml if better found"
    echo ""
fi
