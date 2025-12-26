#!/bin/bash
# Sweep: Generate images with Qwen3-4B vs Qwen3-Embedding-4B
#
# Research Question: Does Qwen3-Embedding-4B produce better images than Qwen3-4B?
#
# Variables:
#   - Encoder model: Qwen3-4B (baseline) vs Qwen3-Embedding-4B
#   - Qwen3-Embedding layer: -2 (matching Qwen3-4B default)
#   - Optional scaling factor (if embedding std differs significantly)
#
# Fixed:
#   - Same prompts, seeds, generation params
#   - Same Z-Image DiT model
#
# Usage:
#   ./experiments/sweep_embedding_generation.sh --config config.toml
#   ./experiments/sweep_embedding_generation.sh --config config.toml --quick --dry-run

set -e

# Required environment variables
if [ -z "$QWEN3_PATH" ] || [ -z "$EMBEDDING_PATH" ]; then
    echo "Error: Set QWEN3_PATH and EMBEDDING_PATH environment variables"
    exit 1
fi
CONFIG=""
PROFILE="default"
OUTPUT_DIR="experiments/results/embedding_generation"
QUICK=false
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --profile) PROFILE="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --quick) QUICK=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "Error: --config is required"
    exit 1
fi

# Prompts and seeds
if [ "$QUICK" = true ]; then
    PROMPTS=("A cat sleeping in sunlight" "A mountain landscape")
    SEEDS="42"
else
    PROMPTS=(
        "A cat sleeping in sunlight"
        "A detailed oil painting of a mountain landscape at golden hour"
        "A photorealistic portrait of a woman with red hair wearing a blue dress"
        "An astronaut riding a horse on the surface of mars"
    )
    SEEDS="42,123,456"
fi

echo "=== Embedding Model Generation Sweep ==="
echo "Comparing image quality: Qwen3-4B vs Qwen3-Embedding-4B"
echo ""
echo "Config: $CONFIG"
echo "Profile: $PROFILE"
echo "Output: $OUTPUT_DIR"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN - No images will be generated]"
    echo ""
    echo "Would generate images for:"
    echo "  Prompts: ${#PROMPTS[@]}"
    echo "  Seeds: $SEEDS"
    echo "  Models: Qwen3-4B (baseline), Qwen3-Embedding-4B"
    echo ""
    echo "Total images: $((${#PROMPTS[@]} * 2 * $(echo $SEEDS | tr ',' '\n' | wc -l)))"
    exit 0
fi

mkdir -p "$OUTPUT_DIR/qwen3_4b"
mkdir -p "$OUTPUT_DIR/qwen3_embedding"
mkdir -p "$OUTPUT_DIR/grids"

# Generate with Qwen3-4B (baseline)
echo "=== Phase 1: Qwen3-4B baseline ==="
for prompt in "${PROMPTS[@]}"; do
    prompt_slug=$(echo "$prompt" | tr ' ' '_' | tr -cd 'a-zA-Z0-9_' | cut -c1-30)

    for seed in $(echo $SEEDS | tr ',' ' '); do
        echo "Generating: $prompt_slug (seed $seed) with Qwen3-4B..."

        uv run scripts/generate.py \
            --config "$CONFIG" \
            --profile "$PROFILE" \
            --seed "$seed" \
            --output "$OUTPUT_DIR/qwen3_4b/${prompt_slug}_seed${seed}.png" \
            "$prompt"
    done
done

# Generate with Qwen3-Embedding-4B
echo ""
echo "=== Phase 2: Qwen3-Embedding-4B ==="
echo "NOTE: This requires modifying the pipeline to use Qwen3-Embedding-4B"
echo "For now, use the embedding extractor manually:"
echo ""
echo "  from llm_dit.embedding import EmbeddingExtractor"
echo "  extractor = EmbeddingExtractor.from_pretrained('$EMBEDDING_PATH')"
echo "  embeddings = extractor.encode_for_zimage('your prompt')"
echo "  # Pass embeddings to pipeline.generate(prompt_embeds=embeddings)"
echo ""

echo "Generation sweep setup complete."
echo "Baseline images in: $OUTPUT_DIR/qwen3_4b/"
echo ""
echo "Next steps:"
echo "1. Run comparison script to analyze embedding statistics"
echo "2. Implement pipeline support for external embeddings"
echo "3. Generate matching images with Qwen3-Embedding-4B"
