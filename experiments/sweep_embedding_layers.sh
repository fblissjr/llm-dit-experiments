#!/bin/bash
# Sweep: Compare hidden layers between Qwen3-4B and Qwen3-Embedding-4B
#
# Research Question: Which layers produce most similar embedding distributions?
#
# Variables:
#   - Qwen3-4B layer: -2 (baseline, what Z-Image uses)
#   - Qwen3-Embedding layers: -1, -2, -3, -4, -6, -8, -12
#
# Fixed:
#   - No instruction prefix
#   - Same prompts
#   - No scaling/normalization
#
# Usage:
#   ./experiments/sweep_embedding_layers.sh
#   ./experiments/sweep_embedding_layers.sh --quick  # Fewer prompts

set -e

QWEN3_PATH="${QWEN3_PATH:-/home/fbliss/Storage/Qwen3-4B}"
EMBEDDING_PATH="${EMBEDDING_PATH:-/home/fbliss/Storage/Qwen3-Embedding-4B}"
OUTPUT_DIR="experiments/results/embedding_layer_sweep"

# Parse arguments
QUICK=false
for arg in "$@"; do
    case $arg in
        --quick) QUICK=true ;;
    esac
done

# Prompts
if [ "$QUICK" = true ]; then
    PROMPTS=("A cat sleeping in sunlight" "A mountain landscape")
else
    PROMPTS=(
        "A cat sleeping in sunlight"
        "A detailed oil painting of a mountain landscape at golden hour"
        "A photorealistic portrait of a woman with red hair"
        "An astronaut riding a horse on mars"
        "A cozy cafe interior with warm lighting"
        "Abstract geometric shapes in vibrant colors"
    )
fi

# Layers to test for Qwen3-Embedding
EMBEDDING_LAYERS=("-1" "-2" "-3" "-4" "-6" "-8" "-12")

echo "=== Embedding Layer Sweep ==="
echo "Testing which Qwen3-Embedding layer best matches Qwen3-4B layer -2"
echo ""

mkdir -p "$OUTPUT_DIR"

for layer in "${EMBEDDING_LAYERS[@]}"; do
    echo "--- Testing Qwen3-Embedding layer $layer ---"

    uv run experiments/compare_embedding_models.py \
        --qwen3-path "$QWEN3_PATH" \
        --embedding-path "$EMBEDDING_PATH" \
        --prompts "${PROMPTS[@]}" \
        --qwen3-layer=-2 \
        --embedding-layer="$layer" \
        --output "$OUTPUT_DIR/layer${layer}_results.json"

    echo ""
done

echo "Layer sweep complete. Results in: $OUTPUT_DIR/"
