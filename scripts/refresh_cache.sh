#!/usr/bin/env bash
# Refresh Python cache - clears bytecode and common caches
# Usage: ./scripts/refresh_cache.sh [--all]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Clearing Python bytecode cache..."
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

if [[ "$1" == "--all" ]]; then
    echo "Clearing pytest cache..."
    rm -rf .pytest_cache 2>/dev/null || true

    echo "Clearing mypy cache..."
    rm -rf .mypy_cache 2>/dev/null || true

    echo "Clearing ruff cache..."
    rm -rf .ruff_cache 2>/dev/null || true

    echo "Clearing coverage data..."
    rm -rf .coverage htmlcov 2>/dev/null || true

    echo "Clearing egg-info..."
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

    echo "Clearing build artifacts..."
    rm -rf build dist 2>/dev/null || true
fi

echo "Done. Cache cleared."
