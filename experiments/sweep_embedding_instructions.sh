#!/bin/bash
# Sweep: Test instruction prefixes for Qwen3-Embedding-4B
#
# Research Question: Does instruction-aware encoding improve embeddings for Z-Image?
#
# Variables:
#   - No instruction (baseline)
#   - Generic image generation instruction
#   - Visual detail instruction
#   - Semantic encoding instruction
#   - Retrieval instruction (what model was trained for)
#
# Fixed:
#   - Layer -2 for both models
#   - Same prompts
#
# Usage:
#   ./experiments/sweep_embedding_instructions.sh
#   ./experiments/sweep_embedding_instructions.sh --quick

set -e

# Required environment variables
if [ -z "$QWEN3_PATH" ] || [ -z "$EMBEDDING_PATH" ]; then
    echo "Error: Set QWEN3_PATH and EMBEDDING_PATH environment variables"
    exit 1
fi
OUTPUT_DIR="experiments/results/embedding_instruction_sweep"

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
    )
fi

echo "=== Embedding Instruction Sweep ==="
echo "Testing if instruction prefixes improve embedding quality for Z-Image"
echo ""

mkdir -p "$OUTPUT_DIR"

# Test 1: No instruction (baseline)
echo "--- Test 1: No instruction (baseline) ---"
uv run experiments/compare_embedding_models.py \
    --qwen3-path "$QWEN3_PATH" \
    --embedding-path "$EMBEDDING_PATH" \
    --prompts "${PROMPTS[@]}" \
    --qwen3-layer=-2 \
    --embedding-layer=-2 \
    --output "$OUTPUT_DIR/no_instruction.json"
echo ""

# Test 2: Image generation instruction
echo "--- Test 2: Image generation instruction ---"
uv run experiments/compare_embedding_models.py \
    --qwen3-path "$QWEN3_PATH" \
    --embedding-path "$EMBEDDING_PATH" \
    --prompts "${PROMPTS[@]}" \
    --qwen3-layer=-2 \
    --embedding-layer=-2 \
    --instruction "Generate a semantic embedding for text-to-image synthesis" \
    --output "$OUTPUT_DIR/image_gen_instruction.json"
echo ""

# Test 3: Visual detail instruction
echo "--- Test 3: Visual detail instruction ---"
uv run experiments/compare_embedding_models.py \
    --qwen3-path "$QWEN3_PATH" \
    --embedding-path "$EMBEDDING_PATH" \
    --prompts "${PROMPTS[@]}" \
    --qwen3-layer=-2 \
    --embedding-layer=-2 \
    --instruction "Encode this text capturing all visual details, colors, composition, and style" \
    --output "$OUTPUT_DIR/visual_detail_instruction.json"
echo ""

# Test 4: Semantic encoding instruction
echo "--- Test 4: Semantic encoding instruction ---"
uv run experiments/compare_embedding_models.py \
    --qwen3-path "$QWEN3_PATH" \
    --embedding-path "$EMBEDDING_PATH" \
    --prompts "${PROMPTS[@]}" \
    --qwen3-layer=-2 \
    --embedding-layer=-2 \
    --instruction "Create a dense semantic representation of this image description" \
    --output "$OUTPUT_DIR/semantic_instruction.json"
echo ""

# Test 5: Retrieval instruction (model's native task)
echo "--- Test 5: Retrieval instruction (model's native task) ---"
uv run experiments/compare_embedding_models.py \
    --qwen3-path "$QWEN3_PATH" \
    --embedding-path "$EMBEDDING_PATH" \
    --prompts "${PROMPTS[@]}" \
    --qwen3-layer=-2 \
    --embedding-layer=-2 \
    --instruction "Given a text description, retrieve relevant visual content" \
    --output "$OUTPUT_DIR/retrieval_instruction.json"
echo ""

echo "Instruction sweep complete. Results in: $OUTPUT_DIR/"
