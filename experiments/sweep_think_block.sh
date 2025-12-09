#!/bin/bash
# Think Block Impact - Priority 3
#
# Test whether think block content actually affects generation.
# DiffSynth (and model training) ALWAYS uses empty think block - test if content helps.
#
# Conditions tested:
#   - Empty: Empty think block (DiffSynth default, model trained with this)
#   - None: No think block (deviates from DiffSynth training)
#   - Quality: "High quality, detailed, photorealistic"
#   - Mood: "Soft lighting, warm colors, peaceful atmosphere"
#   - Technical: "Sharp focus, crisp details, professional composition"
#
# Usage:
#   ./experiments/sweep_think_block.sh              # Full run
#   ./experiments/sweep_think_block.sh --dry-run    # Preview only
#   ./experiments/sweep_think_block.sh --quick      # Quick test (3 prompts, 1 seed)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Default settings
CONFIG="config.toml"
PROFILE="rtx4090"
SEEDS="42,123,456"
MAX_PROMPTS=""
PROMPT_CATEGORY="humans"
DRY_RUN=""
OUTPUT_DIR="experiments/results/think_block_$(date +%Y%m%d_%H%M%S)"

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
echo "Think Block Impact Test"
echo "============================================================"
echo "Config: $CONFIG (profile: $PROFILE)"
echo "Category: $PROMPT_CATEGORY"
echo "Seeds: $SEEDS"
echo "Conditions: empty (default), none, quality, mood, technical"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# Run the experiment
uv run experiments/run_ablation.py \
    --config "$CONFIG" \
    --profile "$PROFILE" \
    --experiment think_block \
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
    echo "  2. Compare: Does 'none' (no think block) vs 'empty' (DiffSynth default) differ?"
    echo "  3. Does quality-focused thinking improve sharpness?"
    echo "  4. Does mood thinking affect colors/atmosphere?"
    echo "  5. Note: Model was trained with empty think blocks, so 'none' might degrade quality"
    echo ""
fi
