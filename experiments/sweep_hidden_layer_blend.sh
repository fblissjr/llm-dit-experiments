#!/bin/bash
# Hidden Layer Blend Sweep - Experimental
#
# Test blending embeddings from multiple hidden layers.
# Hypothesis: Combining semantic (shallow) and structural (deep) layers
# may produce better results than any single layer.
#
# Conditions tested:
#   - Single layers: -2 (default), -1 (last), -5 (deep)
#   - Two-layer blends: various ratios of -2 and -5
#   - Three-layer blends: equal weights across layers
#
# Usage:
#   ./experiments/sweep_hidden_layer_blend.sh              # Full run
#   ./experiments/sweep_hidden_layer_blend.sh --dry-run    # Preview only
#   ./experiments/sweep_hidden_layer_blend.sh --quick      # Quick test (2 prompts, 1 seed)
#   ./experiments/sweep_hidden_layer_blend.sh --metrics    # Enable scoring

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Default settings
CONFIG="config.toml"
PROFILE="rtx4090"
SEEDS="42,123"
MAX_PROMPTS=""
PROMPT_CATEGORY="animals"
DRY_RUN=""
COMPUTE_METRICS=""
OUTPUT_DIR="experiments/results/hidden_layer_blend_$(date +%Y%m%d_%H%M%S)"

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
echo "Hidden Layer Blend Sweep (Experimental)"
echo "============================================================"
echo "Config: $CONFIG (profile: $PROFILE)"
echo "Category: $PROMPT_CATEGORY"
echo "Seeds: $SEEDS"
echo "Conditions: single layers, two-layer blends, three-layer blends"
echo "Metrics: ${COMPUTE_METRICS:-disabled}"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""
echo "This experiment tests whether blending embeddings from multiple"
echo "transformer layers produces better results than single layers."
echo ""

# Run the experiment
uv run experiments/run_ablation.py \
    --config "$CONFIG" \
    --profile "$PROFILE" \
    --experiment hidden_layer_blend \
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
    echo "  2. Compare single-layer baselines (-2, -1, -5)"
    echo "  3. Check if blends outperform single layers"
    echo "  4. Look for: better detail, composition, or prompt adherence"
    echo ""
    echo "Expected findings (based on distilled model hypothesis):"
    echo "  - Blends may show minimal difference due to distillation"
    echo "  - If differences exist, 70/30 blend may work best"
    echo ""
fi
