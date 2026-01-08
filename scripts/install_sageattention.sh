#!/bin/bash
# Install SageAttention from source with optimized build settings
# Uses RTX 4090 optimizations by default

set -e

SAGE_DIR="${SAGE_DIR:-$HOME/dev/SageAttention}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Check if SageAttention source exists
if [ ! -d "$SAGE_DIR" ]; then
    echo "SageAttention source not found at: $SAGE_DIR"
    echo ""
    echo "Clone it first:"
    echo "  git clone https://github.com/thu-ml/SageAttention.git $SAGE_DIR"
    echo ""
    echo "Or set SAGE_DIR to your existing clone:"
    echo "  SAGE_DIR=/path/to/SageAttention $0"
    exit 1
fi

cd "$SAGE_DIR"

# RTX 4090 optimized build settings
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-24}"
export EXT_PARALLEL="${EXT_PARALLEL:-4}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:---threads 6}"

# Enable ccache if available
if command -v ccache &> /dev/null; then
    export PATH="/usr/lib/ccache:$PATH"
    echo "Using ccache"
fi

echo "Building SageAttention for CUDA arch: $TORCH_CUDA_ARCH_LIST"
echo "  MAX_JOBS=$MAX_JOBS, EXT_PARALLEL=$EXT_PARALLEL"

# Clean previous builds
rm -rf build/ dist/ *.egg-info/

# Build with uv, targeting project venv
uv pip install "$SAGE_DIR" \
    --python "$PROJECT_DIR/.venv/bin/python" \
    --no-build-isolation \
    --reinstall-package sageattention

echo ""
echo "Verifying installation..."
"$PROJECT_DIR/.venv/bin/python" -c "from sageattention import sageattn; print('SageAttention installed successfully')"
