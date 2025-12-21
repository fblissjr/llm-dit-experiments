#!/bin/bash
# DyPE Parameter Sweep
#
# Sweeps dype_scale and dype_exponent to find optimal values.
# Tests at 2048x2048 resolution with Vision YaRN method.
#
# Parameters:
#   dype_scale: Controls magnitude of DyPE effect (lambda_s)
#   dype_exponent: Controls decay speed (lambda_t, quadratic=2.0)
#
# Usage:
#   ./experiments/scripts/sweep_dype_params.sh              # Full run
#   ./experiments/scripts/sweep_dype_params.sh --quick      # Quick test (3 values each)
#   ./experiments/scripts/sweep_dype_params.sh --config config.toml --profile rtx4090

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_DIR"

# Default settings
MODEL_PATH=""
CONFIG=""
PROFILE="default"
RESOLUTION="2048"
SEED="42"
STEPS="9"
SHIFT="3.0"
QUICK=""
OUTPUT_DIR="results/dype_sweep/$(date +%Y%m%d_%H%M%S)"

# Parameter ranges
SCALES=(1.0 1.5 2.0 2.5 3.0)
EXPONENTS=(1.0 1.5 2.0 2.5 3.0)

# Quick mode ranges
QUICK_SCALES=(1.0 2.0 3.0)
QUICK_EXPONENTS=(1.0 2.0 3.0)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --shift)
            SHIFT="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --quick)
            QUICK="true"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Use quick ranges if requested
if [[ "$QUICK" == "true" ]]; then
    SCALES=("${QUICK_SCALES[@]}")
    EXPONENTS=("${QUICK_EXPONENTS[@]}")
fi

echo "============================================================"
echo "DyPE Parameter Sweep"
echo "============================================================"
echo "Resolution: ${RESOLUTION}x${RESOLUTION}"
echo "Scales: ${SCALES[*]}"
echo "Exponents: ${EXPONENTS[*]}"
echo "Total runs: $((${#SCALES[@]} * ${#EXPONENTS[@]}))"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Determine base command
if [[ -n "$CONFIG" ]]; then
    BASE_CMD="uv run scripts/generate.py --config $CONFIG --profile $PROFILE"
    MODEL_FLAG=""
else
    if [[ -z "$MODEL_PATH" ]]; then
        echo "Error: Either --config or --model-path is required"
        exit 1
    fi
    BASE_CMD="uv run scripts/generate.py --model-path $MODEL_PATH"
fi

# Standard test prompt
PROMPT="Homer Simpson eating a donut"

# Run baseline (no DyPE)
echo "Running baseline (DyPE disabled)..."
$BASE_CMD \
    --width "$RESOLUTION" \
    --height "$RESOLUTION" \
    --steps "$STEPS" \
    --shift "$SHIFT" \
    --seed "$SEED" \
    --output "$OUTPUT_DIR/baseline.png" \
    "$PROMPT"

echo ""
echo "Baseline complete. Starting parameter sweep..."
echo ""

# Track results
RESULTS_FILE="$OUTPUT_DIR/sweep_results.csv"
echo "scale,exponent,output_file" > "$RESULTS_FILE"

# Counter for progress
TOTAL_RUNS=$((${#SCALES[@]} * ${#EXPONENTS[@]}))
CURRENT_RUN=0

# Sweep parameters
for scale in "${SCALES[@]}"; do
    for exponent in "${EXPONENTS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))

        echo "[$CURRENT_RUN/$TOTAL_RUNS] Testing scale=$scale, exponent=$exponent"

        OUTPUT_FILE="$OUTPUT_DIR/dype_s${scale}_e${exponent}.png"

        $BASE_CMD \
            --width "$RESOLUTION" \
            --height "$RESOLUTION" \
            --steps "$STEPS" \
            --shift "$SHIFT" \
            --seed "$SEED" \
            --dype \
            --dype-method vision_yarn \
            --dype-scale "$scale" \
            --dype-exponent "$exponent" \
            --output "$OUTPUT_FILE" \
            "$PROMPT"

        echo "$scale,$exponent,$OUTPUT_FILE" >> "$RESULTS_FILE"
        echo ""
    done
done

echo "============================================================"
echo "Parameter sweep complete!"
echo "============================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "CSV results: $RESULTS_FILE"
echo ""
echo "Next steps:"
echo "  1. Review images in $OUTPUT_DIR/"
echo "  2. Compare quality across parameter combinations"
echo "  3. Look for optimal scale/exponent values"
echo ""
echo "Create comparison grid with:"
echo "  uv run experiments/compare.py grid --input $OUTPUT_DIR --cols 5"
echo ""
