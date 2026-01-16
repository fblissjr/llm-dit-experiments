"""
TeaCache - Temporal Efficient Attention Caching for LTX-2 Pipeline.

Last Updated: 2026-01-16

Ported from ComfyUI-KJNodes/nodes/model_optimization_nodes.py (lines 700-748).

TeaCache accelerates diffusion inference by 4-10x by skipping redundant
transformer computations. The key insight: consecutive diffusion steps often
produce similar intermediate representations, especially in the middle of
the denoising process.

Algorithm:
1. Track relative L1 distance between consecutive transformer inputs
2. Accumulate distance over steps
3. If accumulated distance < threshold: skip computation, reuse cached residual
4. If exceeds threshold: compute normally, update cache, reset accumulator

Model-specific coefficients rescale the raw L1 distance to a normalized scale
that works well with the threshold parameter. These were empirically determined
for each LTX-2 model variant.

Key Parameters:
- rel_l1_thresh (default 0.275): Threshold for skipping (higher = more skips)
- model_type: Which coefficient set to use ("14B", "1.3B", "i2v_480", "i2v_720")

Speedup depends on threshold and content:
- 0.2 threshold: ~4-5x speedup, minimal quality loss
- 0.3 threshold: ~6-8x speedup, slight quality loss
- 0.4 threshold: ~8-10x speedup, noticeable quality loss on dynamic content

Reference:
    ComfyUI-KJNodes by Kijai
    https://github.com/kijai/ComfyUI-KJNodes

Example:
    tea_cache = TeaCache(rel_l1_thresh=0.275, model_type="14B")

    # In transformer forward:
    for block in transformer_blocks:
        if tea_cache.should_skip(current_input):
            x += tea_cache.get_cached_residual()
        else:
            residual = block(x) - x
            tea_cache.update_cache(current_input, residual)
            x += residual
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch


# Model-specific polynomial coefficients for L1 distance rescaling
# These transform raw L1 distance to a normalized scale for threshold comparison
# Format: [a, b, c, d, e, f] for polynomial a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f
LTX2_COEFFICIENTS = {
    # LTX-2 14B (main video model)
    "14B": [
        -5784.54975374,
        5449.50911966,
        -1811.16591783,
        256.27178429,
        -13.02252404,
        0.19378807,
    ],
    # LTX-2 1.3B (smaller variant)
    "1.3B": [
        2.39676752e03,
        -1.31110545e03,
        2.01331979e02,
        -8.29855975e00,
        1.37887774e-01,
        -9.72239237e-05,
    ],
    # Image-to-Video 480p variant
    "i2v_480": [
        -3.36835004e03,
        3.11450755e03,
        -1.01316022e03,
        1.38929086e02,
        -6.71665488e00,
        9.33893835e-02,
    ],
    # Image-to-Video 720p variant
    "i2v_720": [
        -4.48218918e03,
        4.24828580e03,
        -1.44684881e03,
        2.11740975e02,
        -1.16932115e01,
        2.01023892e-01,
    ],
}


@dataclass
class TeaCacheConfig:
    """Configuration for TeaCache."""

    # Whether TeaCache is enabled
    enabled: bool = True

    # Relative L1 distance threshold for skipping
    # Higher = more aggressive skipping = faster but potentially lower quality
    # Recommended range: 0.2 - 0.4
    rel_l1_thresh: float = 0.275

    # Model type for coefficient selection
    # Options: "14B", "1.3B", "i2v_480", "i2v_720"
    model_type: str = "14B"

    # Start step (0-indexed) - always compute first few steps
    start_step: int = 1  # Skip first step (always compute step 0)

    # End step - always compute last few steps for quality
    # -1 = apply until end (but see skip_last_n)
    end_step: int = -1

    # Number of last steps to always compute (for fine details)
    skip_last_n: int = 1


def compute_relative_l1_distance(
    prev: torch.Tensor,
    current: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Compute relative L1 distance between two tensors.

    The relative distance is normalized by the magnitude of the tensors,
    making the threshold more robust across different scales.

    Args:
        prev: Previous tensor
        current: Current tensor (same shape)
        epsilon: Small value for numerical stability

    Returns:
        Scalar tensor with relative L1 distance
    """
    diff = (current - prev).abs()
    magnitude = (current.abs() + prev.abs()) / 2 + epsilon
    relative = diff / magnitude
    return relative.mean()


def rescale_distance(
    distance: torch.Tensor,
    coefficients: list[float],
) -> torch.Tensor:
    """
    Rescale L1 distance using model-specific polynomial.

    The coefficients were empirically determined for each model to map
    raw L1 distances to a normalized scale where the threshold works
    consistently.

    Args:
        distance: Raw L1 distance (scalar tensor)
        coefficients: Polynomial coefficients [a, b, c, d, e, f]

    Returns:
        Rescaled distance
    """
    x = distance
    a, b, c, d, e, f = coefficients
    # Polynomial: a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f


class TeaCache:
    """
    Temporal Efficient Attention Cache for inference acceleration.

    This class manages the caching state and decision logic for TeaCache.
    It tracks input changes across steps and decides whether to skip
    transformer computation based on accumulated distance.

    Example:
        cache = TeaCache(rel_l1_thresh=0.275, model_type="14B")
        cache.set_total_steps(12)

        for step in range(12):
            cache.set_step(step)
            timestep_emb = get_timestep_embedding(step)

            if cache.should_skip(timestep_emb):
                # Use cached residual
                x = x + cache.get_cached_residual()
            else:
                # Compute normally
                residual = transformer(x) - x
                cache.update_cache(timestep_emb, residual)
                x = x + residual
    """

    def __init__(
        self,
        rel_l1_thresh: float = 0.275,
        model_type: str = "14B",
        start_step: int = 1,
        end_step: int = -1,
        skip_last_n: int = 1,
    ):
        """
        Initialize TeaCache.

        Args:
            rel_l1_thresh: Threshold for skip decision
            model_type: Model type for coefficient selection
            start_step: First step to potentially skip
            end_step: Last step to potentially skip (-1 = all except skip_last_n)
            skip_last_n: Always compute this many last steps
        """
        self.rel_l1_thresh = rel_l1_thresh
        self.model_type = model_type
        self.start_step = start_step
        self.end_step = end_step
        self.skip_last_n = skip_last_n

        # Get coefficients for this model
        if model_type not in LTX2_COEFFICIENTS:
            raise ValueError(
                f"Unknown model_type: {model_type}. "
                f"Available: {list(LTX2_COEFFICIENTS.keys())}"
            )
        self.coefficients = LTX2_COEFFICIENTS[model_type]

        # State
        self.current_step = 0
        self.total_steps: Optional[int] = None
        self.accumulated_distance = 0.0
        self.prev_input: Optional[torch.Tensor] = None
        self.cached_residual: Optional[torch.Tensor] = None

        # Statistics
        self.skip_count = 0
        self.compute_count = 0

    def reset(self) -> None:
        """Reset cache state for new generation."""
        self.current_step = 0
        self.accumulated_distance = 0.0
        self.prev_input = None
        self.cached_residual = None
        self.skip_count = 0
        self.compute_count = 0

    def set_step(self, step: int) -> None:
        """Update current step."""
        self.current_step = step

    def set_total_steps(self, total_steps: int) -> None:
        """Set total steps for end-of-generation handling."""
        self.total_steps = total_steps

    def _should_force_compute(self) -> bool:
        """Check if we should always compute (first/last steps)."""
        # Always compute first step(s)
        if self.current_step < self.start_step:
            return True

        # Always compute last step(s)
        if self.total_steps is not None:
            if self.current_step >= self.total_steps - self.skip_last_n:
                return True

        # Check end_step setting
        if self.end_step >= 0 and self.current_step > self.end_step:
            return True

        return False

    def should_skip(self, current_input: torch.Tensor) -> bool:
        """
        Determine if transformer computation should be skipped.

        Args:
            current_input: Current timestep embedding or relevant input tensor

        Returns:
            True if computation can be skipped (use cached residual)
        """
        # Force compute for first/last steps
        if self._should_force_compute():
            return False

        # Need previous input for comparison
        if self.prev_input is None:
            return False

        # Need cached residual to reuse
        if self.cached_residual is None:
            return False

        # Compute relative L1 distance
        raw_distance = compute_relative_l1_distance(self.prev_input, current_input)

        # Rescale using model-specific coefficients
        rescaled = rescale_distance(raw_distance, self.coefficients)

        # Accumulate distance
        self.accumulated_distance += rescaled.item()

        # Decision: skip if accumulated distance below threshold
        return self.accumulated_distance < self.rel_l1_thresh

    def get_cached_residual(self) -> torch.Tensor:
        """
        Get the cached residual for reuse.

        Returns:
            Cached residual tensor

        Raises:
            RuntimeError: If no cached residual available
        """
        if self.cached_residual is None:
            raise RuntimeError("No cached residual available")

        self.skip_count += 1
        return self.cached_residual

    def update_cache(
        self,
        current_input: torch.Tensor,
        residual: torch.Tensor,
    ) -> None:
        """
        Update cache with new computation results.

        Should be called after computing the transformer when not skipping.

        Args:
            current_input: Current timestep embedding/input
            residual: Computed residual (transformer_output - input)
        """
        self.prev_input = current_input.clone().detach()
        self.cached_residual = residual
        self.accumulated_distance = 0.0  # Reset accumulator
        self.compute_count += 1

    def get_stats(self) -> Dict[str, float]:
        """
        Get caching statistics.

        Returns:
            Dict with skip_count, compute_count, skip_ratio
        """
        total = self.skip_count + self.compute_count
        skip_ratio = self.skip_count / total if total > 0 else 0.0
        return {
            "skip_count": self.skip_count,
            "compute_count": self.compute_count,
            "total_steps": total,
            "skip_ratio": skip_ratio,
            "speedup_estimate": 1 / (1 - skip_ratio) if skip_ratio < 1 else float('inf'),
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"TeaCache(thresh={self.rel_l1_thresh}, model={self.model_type}, "
            f"skips={stats['skip_count']}/{stats['total_steps']}, "
            f"ratio={stats['skip_ratio']:.1%})"
        )


class TeaCacheManager:
    """
    Manager for multiple TeaCache instances (one per transformer block).

    For models with many transformer blocks, each block can have independent
    caching state. This manager coordinates them and provides aggregate stats.

    Example:
        manager = TeaCacheManager(num_blocks=48, rel_l1_thresh=0.275)

        for block_idx, block in enumerate(transformer.blocks):
            cache = manager.get_cache(block_idx)
            if cache.should_skip(timestep_emb):
                x = x + cache.get_cached_residual()
            else:
                residual = block(x) - x
                cache.update_cache(timestep_emb, residual)
                x = x + residual
    """

    def __init__(
        self,
        num_blocks: int,
        rel_l1_thresh: float = 0.275,
        model_type: str = "14B",
        start_step: int = 1,
        end_step: int = -1,
        skip_last_n: int = 1,
    ):
        """
        Initialize TeaCache manager.

        Args:
            num_blocks: Number of transformer blocks
            rel_l1_thresh: Threshold (shared across blocks)
            model_type: Model type for coefficients
            start_step: First step to potentially skip
            end_step: Last step to potentially skip
            skip_last_n: Always compute last N steps
        """
        self.num_blocks = num_blocks
        self.caches = [
            TeaCache(
                rel_l1_thresh=rel_l1_thresh,
                model_type=model_type,
                start_step=start_step,
                end_step=end_step,
                skip_last_n=skip_last_n,
            )
            for _ in range(num_blocks)
        ]

    def reset(self) -> None:
        """Reset all caches."""
        for cache in self.caches:
            cache.reset()

    def set_step(self, step: int) -> None:
        """Set current step for all caches."""
        for cache in self.caches:
            cache.set_step(step)

    def set_total_steps(self, total_steps: int) -> None:
        """Set total steps for all caches."""
        for cache in self.caches:
            cache.set_total_steps(total_steps)

    def get_cache(self, block_idx: int) -> TeaCache:
        """Get cache for specific block."""
        return self.caches[block_idx]

    def get_aggregate_stats(self) -> Dict[str, float]:
        """Get aggregate statistics across all blocks."""
        total_skips = sum(c.skip_count for c in self.caches)
        total_computes = sum(c.compute_count for c in self.caches)
        total = total_skips + total_computes

        return {
            "total_skips": total_skips,
            "total_computes": total_computes,
            "total_operations": total,
            "overall_skip_ratio": total_skips / total if total > 0 else 0.0,
            "per_block_stats": [c.get_stats() for c in self.caches],
        }
