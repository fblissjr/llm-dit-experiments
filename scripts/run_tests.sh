#!/bin/bash
# Run LTX-2 baseline tests
#
# Usage:
#   ./scripts/run_tests.sh              # Run smoke test (fastest)
#   ./scripts/run_tests.sh short        # Run short quality test (~2min)
#   ./scripts/run_tests.sh all          # Run all tests except slow
#   ./scripts/run_tests.sh reference    # Run full reference test (~10min)
#   ./scripts/run_tests.sh embedding    # Run embedding comparison test

set -e

TEST_TYPE="${1:-smoke}"

case "$TEST_TYPE" in
    smoke)
        echo "Running smoke test (fastest validation)..."
        uv run pytest tests/e2e/test_baseline_portable.py::TestBaselineSmoke -s
        ;;
    short)
        echo "Running short T2V test (~2min)..."
        uv run pytest tests/e2e/test_baseline_portable.py::TestBaselineT2V::test_t2v_short -s
        ;;
    embedding)
        echo "Running embedding comparison test..."
        uv run pytest tests/e2e/test_baseline_portable.py::TestBaselineComparison::test_text_embedding_shape -s
        ;;
    reference)
        echo "Running full reference test (~10min, requires --runslow)..."
        uv run pytest tests/e2e/test_baseline_portable.py::TestBaselineT2V::test_t2v_reference -s --runslow
        ;;
    all)
        echo "Running all tests (except slow)..."
        uv run pytest tests/e2e/test_baseline_portable.py -s
        ;;
    verbose)
        echo "Running smoke test with verbose output..."
        uv run pytest tests/e2e/test_baseline_portable.py::TestBaselineSmoke -v -s
        ;;
    *)
        echo "Unknown test type: $TEST_TYPE"
        echo ""
        echo "Usage: $0 [smoke|short|embedding|reference|all|verbose]"
        echo ""
        echo "  smoke      - Fastest validation (~30s, 14GB VRAM)"
        echo "  short      - Short T2V test (~2min, 16GB VRAM)"
        echo "  embedding  - Text embedding shape comparison"
        echo "  reference  - Full quality test (~10min, 20GB VRAM)"
        echo "  all        - All tests except slow"
        echo "  verbose    - Smoke test with verbose pytest output"
        exit 1
        ;;
esac
