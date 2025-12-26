#!/bin/bash
# Compare Qwen3-4B vs Qwen3-Embedding-4B for Z-Image embeddings
#
# Usage:
#   ./experiments/run_embedding_comparison.sh           # Basic comparison
#   ./experiments/run_embedding_comparison.sh --test-instructions  # Test with instructions

set -e

# Required environment variables
if [ -z "$QWEN3_PATH" ] || [ -z "$EMBEDDING_PATH" ]; then
    echo "Error: Set QWEN3_PATH and EMBEDDING_PATH environment variables"
    exit 1
fi
OUTPUT_DIR="experiments/results/embedding_comparison"

# Default prompts that test different scenarios
PROMPTS=(
    "A cat sleeping in sunlight"
    "A detailed oil painting of a mountain landscape at golden hour"
    "A photorealistic portrait of a woman with red hair wearing a blue silk dress"
    "An astronaut riding a horse on the surface of mars, cinematic lighting"
)

echo "=== Qwen3-4B vs Qwen3-Embedding-4B Comparison ==="
echo "Qwen3-4B path: $QWEN3_PATH"
echo "Embedding path: $EMBEDDING_PATH"
echo ""

# Run comparison
uv run experiments/compare_embedding_models.py \
    --qwen3-path "$QWEN3_PATH" \
    --embedding-path "$EMBEDDING_PATH" \
    --prompts "${PROMPTS[@]}" \
    --qwen3-layer=-2 \
    --embedding-layer=-2 \
    --output "$OUTPUT_DIR/comparison_results.json" \
    "$@"

echo ""
echo "Results saved to: $OUTPUT_DIR/comparison_results.json"
