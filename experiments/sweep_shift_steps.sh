#!/bin/bash
# Shift + Steps Grid Search - Priority 2
#
# Find the quality ceiling by testing shift and step combinations.
# Default is shift=3.0, steps=9 - this tests if we can do better.
#
# Grid:
#   shift: 2.0, 3.0, 4.0
#   steps: 6, 9, 12, 15
#   = 12 combinations per prompt
#
# Usage:
#   ./experiments/sweep_shift_steps.sh              # Full run
#   ./experiments/sweep_shift_steps.sh --dry-run    # Preview only
#   ./experiments/sweep_shift_steps.sh --quick      # Quick test (2 prompts, 1 seed)

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
OUTPUT_DIR="experiments/results/shift_steps_$(date +%Y%m%d_%H%M%S)"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --quick)
            MAX_PROMPTS="--max-prompts 2"
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
echo "Shift + Steps Grid Search"
echo "============================================================"
echo "Config: $CONFIG (profile: $PROFILE)"
echo "Category: $PROMPT_CATEGORY"
echo "Seeds: $SEEDS"
echo "Grid: shift=[2.0,3.0,4.0] x steps=[6,9,12,15]"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# Run the experiment
uv run experiments/run_ablation.py \
    --config "$CONFIG" \
    --profile "$PROFILE" \
    --experiment shift_steps_grid \
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
    echo "  2. Compare quality vs generation time"
    echo "  3. Find the pareto frontier (best quality per time)"
    echo "  4. Check if lower shift + more steps beats default"
    echo ""
fi
