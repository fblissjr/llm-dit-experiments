"""
LTX-2 baseline generation and comparison utilities.

Last Updated: 2026-02-02

Provides infrastructure for generating baseline videos with deterministic
settings and comparing outputs for reproducibility testing.

Usage:
    from tests.baselines import generate_baseline, compare_baselines, BaselineResult

    # Generate a baseline video
    result = generate_baseline(config_tier="smoke", seed=42)

    # Compare two videos
    comparison = compare_baselines(video1_path, video2_path)
    assert comparison["ssim"] > 0.99  # Near-identical
"""

from .ltx2_baseline_runner import (
    BaselineResult,
    ComparisonResult,
    compare_baselines,
    generate_baseline,
    generate_baseline_from_preset,
    get_baseline_config,
)

__all__ = [
    "BaselineResult",
    "ComparisonResult",
    "generate_baseline",
    "generate_baseline_from_preset",
    "compare_baselines",
    "get_baseline_config",
]
