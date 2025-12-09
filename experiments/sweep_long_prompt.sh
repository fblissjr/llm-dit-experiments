#!/bin/bash
# Long Prompt Compression Modes - Priority 4
#
# Compare compression modes for prompts exceeding 1504 tokens.
# Only relevant if you have very long prompts.
#
# Modes tested:
#   - truncate: Cut off at 1504 (loses end)
#   - interpolate: Linear resampling (default, preserves all)
#   - pool: Adaptive average pooling
#   - attention_pool: Cosine similarity weighted
#
# NOTE: Standard prompts are typically <1504 tokens, so this
# experiment requires custom long prompts in experiments/prompts/
#
# Usage:
#   ./experiments/sweep_long_prompt.sh              # Full run
#   ./experiments/sweep_long_prompt.sh --dry-run    # Preview only
#   ./experiments/sweep_long_prompt.sh --quick      # Quick test (1 prompt, 1 seed)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Default settings
CONFIG="config.toml"
PROFILE="rtx4090"
SEEDS="42"
MAX_PROMPTS=""
PROMPT_CATEGORY="technical"  # Most likely to have longer prompts
DRY_RUN=""
COMPUTE_METRICS=""
OUTPUT_DIR="experiments/results/long_prompt_$(date +%Y%m%d_%H%M%S)"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --quick)
            MAX_PROMPTS="--max-prompts 1"
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
        --metrics)
            COMPUTE_METRICS="--compute-metrics"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "Long Prompt Compression Mode Test"
echo "============================================================"
echo "Config: $CONFIG (profile: $PROFILE)"
echo "Category: $PROMPT_CATEGORY"
echo "Seeds: $SEEDS"
echo "Modes: truncate, interpolate, pool, attention_pool"
echo "Metrics: ${COMPUTE_METRICS:-disabled}"
echo "Output: $OUTPUT_DIR"
echo ""
echo "NOTE: Compression only triggers for prompts >1504 tokens."
echo "      Standard prompts may be too short to test this."
echo "============================================================"
echo ""

# Run the experiment
uv run experiments/run_ablation.py \
    --config "$CONFIG" \
    --profile "$PROFILE" \
    --experiment long_prompt_mode \
    --prompt-category "$PROMPT_CATEGORY" \
    --seeds "$SEEDS" \
    --output-dir "$OUTPUT_DIR" \
    $MAX_PROMPTS \
    $DRY_RUN \
    $COMPUTE_METRICS

if [[ -z "$DRY_RUN" ]]; then
    echo ""
    echo "============================================================"
    echo "Results saved to: $OUTPUT_DIR"
    echo "============================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Review images in $OUTPUT_DIR/"
    echo "  2. Check if compression was actually applied (prompt >1504 tokens)"
    echo "  3. Compare: Does interpolate preserve more content than truncate?"
    echo "  4. Does attention_pool focus on key concepts?"
    echo ""
fi
