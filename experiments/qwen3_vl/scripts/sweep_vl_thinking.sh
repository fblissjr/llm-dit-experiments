#!/bin/bash
# Comprehensive A/B comparison: VL-4B-Instruct vs VL-4B-Thinking
#
# This sweep compares the two Qwen3-VL-4B model variants to determine
# if the "thinking" training objective produces better embeddings for
# Z-Image conditioning.
#
# Hypothesis: The Thinking model may produce better embeddings because:
# 1. Training objective preserves information for multi-step reasoning
# 2. Later layers may not be as heavily overwritten by SFT objectives
# 3. Native think block support (no manual injection needed)
#
# Usage:
#   ./experiments/qwen3_vl/scripts/sweep_vl_thinking.sh
#
# Results are saved to:
#   experiments/results/vl_thinking_comparison_YYYYMMDD/

set -e

# Configuration
DATE=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="experiments/results/vl_thinking_comparison_${DATE}"
REFERENCE_IMAGE="experiments/inputs/style_anime_girl.png"
STEPS=4

# Test prompts
PROMPTS=(
    "Homer Simpson eating spaghetti"
    "A cartoon house with a red roof"
)

echo "============================================================"
echo "VL-4B Model Variant Comparison Sweep"
echo "============================================================"
echo "Output: ${OUTPUT_BASE}"
echo "Reference: ${REFERENCE_IMAGE}"
echo "Steps: ${STEPS}"
echo "Prompts: ${#PROMPTS[@]}"
echo "============================================================"

for prompt in "${PROMPTS[@]}"; do
    echo ""
    echo "============================================================"
    echo "Prompt: ${prompt}"
    echo "============================================================"

    # Sanitize prompt for directory name
    PROMPT_DIR=$(echo "${prompt}" | tr ' ' '_' | tr -cd '[:alnum:]_')

    # Run comparison with both model variants
    uv run experiments/qwen3_vl/scripts/run_comparison.py \
        --image "${REFERENCE_IMAGE}" \
        --prompt "${prompt}" \
        --vl-model-variant both \
        --alphas 0.3 0.5 1.0 \
        --layers -2 -6 -8 \
        --token-modes text_only full \
        --blend-modes adain_per_dim \
        --steps ${STEPS} \
        --no-baseline \
        --output-dir "${OUTPUT_BASE}/${PROMPT_DIR}"
done

echo ""
echo "============================================================"
echo "Sweep complete!"
echo "Results saved to: ${OUTPUT_BASE}"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Review comparison grids in each output directory"
echo "2. Compare Thinking vs Instruct at same layer/alpha settings"
echo "3. Look for differences in prompt adherence and style transfer"
