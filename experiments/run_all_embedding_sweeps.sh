#!/bin/bash
# Run all Qwen3-Embedding-4B comparison sweeps
#
# Research Goal: Determine if Qwen3-Embedding-4B can be used as an alternative
# or better text encoder for Z-Image.
#
# Sweep Order (isolating variables):
#   1. Layer sweep - find best matching layer
#   2. Instruction sweep - test instruction prefixes
#   3. Generation sweep - actual image comparison (requires manual step)
#
# Usage:
#   ./experiments/run_all_embedding_sweeps.sh
#   ./experiments/run_all_embedding_sweeps.sh --quick
#   ./experiments/run_all_embedding_sweeps.sh --stats-only  # Skip generation

set -e

# Parse arguments
QUICK=""
STATS_ONLY=false
CONFIG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick) QUICK="--quick"; shift ;;
        --stats-only) STATS_ONLY=true; shift ;;
        --config) CONFIG="--config $2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=============================================="
echo "  QWEN3-EMBEDDING-4B COMPARISON SWEEPS"
echo "=============================================="
echo ""
echo "Environment:"
echo "  QWEN3_PATH: ${QWEN3_PATH:?Error: Set QWEN3_PATH environment variable}"
echo "  EMBEDDING_PATH: ${EMBEDDING_PATH:?Error: Set EMBEDDING_PATH environment variable}"
echo ""

# Make scripts executable
chmod +x experiments/sweep_embedding_layers.sh
chmod +x experiments/sweep_embedding_instructions.sh
chmod +x experiments/sweep_embedding_generation.sh
chmod +x experiments/run_embedding_comparison.sh

# Sweep 1: Layer comparison
echo "=============================================="
echo "  SWEEP 1: Hidden Layer Comparison"
echo "=============================================="
echo "Finding which Qwen3-Embedding layer best matches Qwen3-4B layer -2"
echo ""
./experiments/sweep_embedding_layers.sh $QUICK
echo ""

# Sweep 2: Instruction variants
echo "=============================================="
echo "  SWEEP 2: Instruction Prefix Comparison"
echo "=============================================="
echo "Testing if instruction-aware encoding improves results"
echo ""
./experiments/sweep_embedding_instructions.sh $QUICK
echo ""

# Sweep 3: Generation comparison
if [ "$STATS_ONLY" = false ]; then
    if [ -z "$CONFIG" ]; then
        echo "=============================================="
        echo "  SWEEP 3: Image Generation (SKIPPED)"
        echo "=============================================="
        echo "Requires --config flag for generation"
        echo "Run manually: ./experiments/sweep_embedding_generation.sh --config config.toml"
    else
        echo "=============================================="
        echo "  SWEEP 3: Image Generation Comparison"
        echo "=============================================="
        ./experiments/sweep_embedding_generation.sh $CONFIG $QUICK
    fi
fi

echo ""
echo "=============================================="
echo "  SWEEPS COMPLETE"
echo "=============================================="
echo ""
echo "Results locations:"
echo "  Layer sweep:       experiments/results/embedding_layer_sweep/"
echo "  Instruction sweep: experiments/results/embedding_instruction_sweep/"
echo "  Generation sweep:  experiments/results/embedding_generation/"
echo ""
echo "Next steps:"
echo "  1. Review layer sweep to find best matching layer"
echo "  2. Check if any instruction improves correlation"
echo "  3. If std ratio != 1.0, apply scaling before generation"
echo "  4. Generate comparison images to evaluate quality"
