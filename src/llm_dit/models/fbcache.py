"""Forward Block Cache (FBCache) for accelerating DiT inference.

FBCache skips redundant transformer block computations when residual changes
between diffusion steps are minimal. This provides 30-50% speedup with minimal
quality degradation.

Based on DiffSynth-Engine implementation with adaptive thresholds by sigma phase.

Usage:
    from llm_dit.models.fbcache import FBCacheConfig, FBCacheState

    config = FBCacheConfig(enabled=True, log_residuals=True)
    state = FBCacheState(config, num_inference_steps=8)

    for step, t in enumerate(timesteps):
        sigma = scheduler.sigmas[step]

        # First block computation
        first_residual = first_block(hidden_states) - hidden_states

        if state.should_skip(first_residual, sigma, step):
            # Reuse cached residual from previous step
            output = hidden_states + state.cached_residual
        else:
            # Full computation
            output = run_all_blocks(hidden_states)
            state.update_cache(first_residual, output - hidden_after_first_block)

        state.step_count += 1
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class FBCacheConfig:
    """Configuration for Forward Block Cache.

    Attributes:
        enabled: Master toggle for FBCache
        early_threshold: Threshold for sigma > 0.7 (structure discovery phase)
        middle_threshold: Threshold for 0.3 < sigma < 0.7 (detail refinement)
        late_threshold: Threshold for sigma < 0.3 (fine details phase)
        log_residuals: Whether to log residual statistics for analysis
        log_file: Optional file path for residual logs (None = logger only)
    """

    enabled: bool = False

    # Phase-aware thresholds (relative L1 distance)
    # Lower = more conservative (fewer skips), Higher = more aggressive (more skips)
    early_threshold: float = 0.01  # 1% - conservative during structure discovery
    middle_threshold: float = 0.05  # 5% - aggressive during detail refinement
    late_threshold: float = 0.01  # 1% - conservative for fine details

    # Sigma boundaries for phase detection
    early_sigma_min: float = 0.7
    late_sigma_max: float = 0.3

    # Logging
    log_residuals: bool = False
    log_file: Optional[str] = None


def relative_l1_distance(
    current: torch.Tensor,
    previous: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Compute relative L1 distance between tensors.

    Formula: |current - previous|_1 / (|previous|_1 + eps)

    This measures how much the residual has changed relative to its magnitude.
    A value of 0.05 means 5% relative change.

    Args:
        current: Current residual tensor
        previous: Previous step's residual tensor
        eps: Small value for numerical stability

    Returns:
        Relative L1 distance as a scalar float
    """
    with torch.no_grad():
        diff_norm = torch.abs(current - previous).sum().item()
        prev_norm = torch.abs(previous).sum().item()
        return diff_norm / (prev_norm + eps)


@dataclass
class FBCacheState:
    """Tracks FBCache state across diffusion steps.

    This class manages:
    - Step counting for first/last step detection
    - Residual history for skip decisions
    - Cached outputs for reuse
    - Residual statistics logging
    """

    config: FBCacheConfig
    num_inference_steps: int

    # Internal state (initialized per generation)
    step_count: int = field(default=0, init=False)
    prev_first_residual: Optional[torch.Tensor] = field(default=None, init=False)
    cached_remaining_residual: Optional[torch.Tensor] = field(default=None, init=False)

    # Statistics
    skips_count: int = field(default=0, init=False)
    computes_count: int = field(default=0, init=False)
    residual_log: list = field(default_factory=list, init=False)

    def reset(self, num_inference_steps: Optional[int] = None) -> None:
        """Reset state for new generation.

        Args:
            num_inference_steps: Optionally update inference steps count
        """
        if num_inference_steps is not None:
            self.num_inference_steps = num_inference_steps

        self.step_count = 0
        self.prev_first_residual = None
        self.cached_remaining_residual = None
        self.skips_count = 0
        self.computes_count = 0
        self.residual_log.clear()

    def get_threshold_for_sigma(self, sigma: float) -> float:
        """Get adaptive threshold based on sigma phase.

        Args:
            sigma: Current noise level (0 = clean, 1 = noisy)

        Returns:
            Threshold for this sigma phase
        """
        if sigma > self.config.early_sigma_min:
            return self.config.early_threshold
        elif sigma < self.config.late_sigma_max:
            return self.config.late_threshold
        else:
            return self.config.middle_threshold

    def should_skip(
        self,
        first_residual: torch.Tensor,
        sigma: float,
    ) -> bool:
        """Determine if remaining blocks should be skipped.

        Logic:
        - Always compute on first step (no prior residual)
        - Always compute on last step (final quality)
        - Otherwise, skip if residual change is below threshold

        Args:
            first_residual: Residual from first transformer block
            sigma: Current noise level

        Returns:
            True if remaining blocks should be skipped
        """
        if not self.config.enabled:
            return False

        is_first_step = self.step_count == 0
        is_last_step = self.step_count == self.num_inference_steps - 1

        # Always compute first and last steps
        if is_first_step or is_last_step:
            return False

        # No prior residual to compare
        if self.prev_first_residual is None:
            return False

        # No cached output to reuse
        if self.cached_remaining_residual is None:
            return False

        # Compute relative change
        threshold = self.get_threshold_for_sigma(sigma)
        rel_diff = relative_l1_distance(first_residual, self.prev_first_residual)

        should_skip = rel_diff < threshold

        # Log if enabled
        if self.config.log_residuals:
            self._log_residual(sigma, rel_diff, threshold, should_skip)

        return should_skip

    def update_cache(
        self,
        first_residual: torch.Tensor,
        remaining_residual: torch.Tensor,
    ) -> None:
        """Update cache after full computation.

        Args:
            first_residual: Residual from first block (for next step's comparison)
            remaining_residual: Residual from blocks 2-N (for skip reuse)
        """
        # Clone to avoid reference issues
        self.prev_first_residual = first_residual.clone()
        self.cached_remaining_residual = remaining_residual.clone()
        self.computes_count += 1

    def mark_skipped(self) -> None:
        """Record that this step was skipped."""
        self.skips_count += 1

    def advance_step(self) -> None:
        """Advance to next diffusion step."""
        self.step_count += 1

    def _log_residual(
        self,
        sigma: float,
        rel_diff: float,
        threshold: float,
        skipped: bool,
    ) -> None:
        """Log residual statistics."""
        entry = {
            "step": self.step_count,
            "sigma": sigma,
            "rel_diff": rel_diff,
            "threshold": threshold,
            "skipped": skipped,
        }
        self.residual_log.append(entry)

        if logger.isEnabledFor(logging.DEBUG):
            action = "SKIP" if skipped else "COMPUTE"
            logger.debug(
                f"[FBCache] Step {self.step_count}: sigma={sigma:.3f}, "
                f"rel_diff={rel_diff:.4f}, threshold={threshold:.3f} -> {action}"
            )

    def get_stats(self) -> dict:
        """Get FBCache statistics for this generation.

        Returns:
            Dictionary with skip/compute counts and efficiency ratio
        """
        total = self.skips_count + self.computes_count
        skip_ratio = self.skips_count / total if total > 0 else 0.0

        return {
            "skips": self.skips_count,
            "computes": self.computes_count,
            "total_steps": total,
            "skip_ratio": skip_ratio,
            "estimated_speedup": 1.0 / (1.0 - skip_ratio * 0.8) if skip_ratio > 0 else 1.0,
        }

    def write_log_file(self) -> None:
        """Write residual log to file if configured."""
        if not self.config.log_file or not self.residual_log:
            return

        import json

        with open(self.config.log_file, "w") as f:
            json.dump(
                {
                    "config": {
                        "early_threshold": self.config.early_threshold,
                        "middle_threshold": self.config.middle_threshold,
                        "late_threshold": self.config.late_threshold,
                    },
                    "stats": self.get_stats(),
                    "residuals": self.residual_log,
                },
                f,
                indent=2,
            )
        logger.info(f"[FBCache] Wrote residual log to {self.config.log_file}")


class FBCacheLayersWrapper(torch.nn.Module):
    """Wrapper for transformer layers that implements FBCache skip logic.

    This wraps the original ModuleList and intercepts iteration to:
    1. Always run layer 0 (first block)
    2. Compare first-block residual to previous step
    3. If similar, skip layers 1-N and add cached residual
    4. If different, run all layers and cache the difference

    The wrapper is installed by replacing transformer.layers temporarily.
    """

    def __init__(self, original_layers: torch.nn.ModuleList, state: FBCacheState):
        super().__init__()
        self._original_layers = original_layers
        self._state = state
        self._current_sigma: float = 1.0
        self._skip_remaining: bool = False
        self._unified_after_first: Optional[torch.Tensor] = None

    def set_sigma(self, sigma: float) -> None:
        """Set current sigma before each forward pass."""
        self._current_sigma = sigma
        self._skip_remaining = False
        self._unified_after_first = None

    def __len__(self):
        return len(self._original_layers)

    def __iter__(self):
        """Custom iteration that implements FBCache logic.

        The Z-Image forward does:
            for layer_idx, layer in enumerate(self.layers):
                unified = layer(unified, ...)

        We intercept this by yielding wrapped layers that can skip.
        """
        for layer_idx, layer in enumerate(self._original_layers):
            yield _FBCacheLayerWrapper(
                layer=layer,
                layer_idx=layer_idx,
                parent=self,
            )

    def __getitem__(self, idx):
        """Support direct indexing (used by controlnet checks)."""
        return self._original_layers[idx]


class _FBCacheLayerWrapper(torch.nn.Module):
    """Wrapper for a single transformer layer with FBCache logic."""

    def __init__(self, layer: torch.nn.Module, layer_idx: int, parent: FBCacheLayersWrapper):
        super().__init__()
        self._layer = layer
        self._layer_idx = layer_idx
        self._parent = parent

    def forward(self, unified, *args, **kwargs):
        """Forward with FBCache skip logic.

        Args:
            unified: Hidden states tensor
            *args, **kwargs: Additional arguments for the layer
        """
        state = self._parent._state

        if self._layer_idx == 0:
            # First layer - always compute
            unified_before = unified.clone()
            unified_out = self._layer(unified, *args, **kwargs)

            # Compute residual for skip decision
            first_residual = unified_out - unified_before

            # Store output for later residual computation
            self._parent._unified_after_first = unified_out

            # Decide if we should skip remaining layers
            should_skip = state.should_skip(first_residual, self._parent._current_sigma)

            if should_skip and state.cached_remaining_residual is not None:
                # Mark for skipping
                self._parent._skip_remaining = True
                state.mark_skipped()

                if state.config.log_residuals:
                    logger.info(
                        f"[FBCache] Step {state.step_count}: SKIP "
                        f"(sigma={self._parent._current_sigma:.3f})"
                    )
            else:
                # Will compute all layers - update first residual for next step's comparison
                self._parent._skip_remaining = False
                state.prev_first_residual = first_residual.clone()

            return unified_out

        else:
            # Layers 1-N
            is_last_layer = self._layer_idx == len(self._parent) - 1

            if self._parent._skip_remaining:
                # Skip computation for this layer
                if is_last_layer:
                    # On last layer when skipping: add cached residual
                    if state.cached_remaining_residual is not None:
                        return unified + state.cached_remaining_residual.to(
                            device=unified.device, dtype=unified.dtype
                        )
                # Not last layer: return input unchanged
                return unified
            else:
                # Compute normally
                unified_out = self._layer(unified, *args, **kwargs)

                # If this is the last layer, update cache
                if is_last_layer:
                    # Compute remaining residual: output - (output after first layer)
                    if self._parent._unified_after_first is not None:
                        remaining_residual = unified_out - self._parent._unified_after_first
                        # Update cached remaining residual
                        state.cached_remaining_residual = remaining_residual.clone()
                        state.computes_count += 1

                        if state.config.log_residuals:
                            logger.info(
                                f"[FBCache] Step {state.step_count}: COMPUTE "
                                f"(sigma={self._parent._current_sigma:.3f})"
                            )

                return unified_out


class FBCacheContext:
    """Context manager for applying FBCache to Z-Image transformer.

    This replaces the transformer's layers with a wrapper that implements
    block-level caching. The original layers are restored on exit.

    Usage:
        from llm_dit.models.fbcache import FBCacheConfig, FBCacheState, FBCacheContext

        config = FBCacheConfig(enabled=True)
        state = FBCacheState(config, num_inference_steps=8)

        with FBCacheContext(transformer, state) as ctx:
            for i, t in enumerate(timesteps):
                sigma = scheduler.sigmas[i].item()
                ctx.set_sigma(sigma)
                output = transformer(...)  # Uses FBCache skip logic
                ctx.advance_step()

        # After context exits, print stats
        print(state.get_stats())
    """

    def __init__(self, transformer, state: FBCacheState):
        """Initialize FBCache context.

        Args:
            transformer: Transformer model instance (Z-Image or Qwen-Image)
            state: FBCacheState for tracking across steps
        """
        self.transformer = transformer
        self.state = state
        self._original_layers = None
        self._wrapper = None
        self._layers_attr = None

        # Detect transformer structure (Z-Image uses 'layers', Qwen-Image uses 'transformer_blocks')
        if hasattr(transformer, 'layers'):
            self._layers_attr = 'layers'
        elif hasattr(transformer, 'transformer_blocks'):
            self._layers_attr = 'transformer_blocks'
        else:
            raise ValueError(
                "FBCache requires transformer with 'layers' or 'transformer_blocks' attribute. "
                "Supported: ZImageTransformer2DModel, QwenImageTransformer2DModel"
            )

    def set_sigma(self, sigma: float) -> None:
        """Set current sigma for threshold selection.

        Call this BEFORE each transformer forward pass.

        Args:
            sigma: Current noise level
        """
        if self._wrapper is not None:
            self._wrapper.set_sigma(sigma)

    def advance_step(self) -> None:
        """Advance to next diffusion step.

        Call this AFTER each transformer forward pass.
        """
        self.state.advance_step()

    def apply_cached_residual(self, unified: torch.Tensor) -> torch.Tensor:
        """Apply cached residual if skip mode was active.

        Call this after the layer loop completes to add the cached residual.

        Args:
            unified: Output from layer loop

        Returns:
            unified with cached residual added (if skip mode) or unchanged
        """
        if self._wrapper is not None and self._wrapper._skip_remaining:
            if self.state.cached_remaining_residual is not None:
                return unified + self.state.cached_remaining_residual
        return unified

    def __enter__(self):
        """Install FBCache layers wrapper."""
        if not self.state.config.enabled:
            return self

        # Get and wrap the layers using the detected attribute name
        self._original_layers = getattr(self.transformer, self._layers_attr)
        self._wrapper = FBCacheLayersWrapper(self._original_layers, self.state)
        setattr(self.transformer, self._layers_attr, self._wrapper)

        logger.info(
            f"[FBCache] Enabled with thresholds: "
            f"early={self.state.config.early_threshold:.2%}, "
            f"middle={self.state.config.middle_threshold:.2%}, "
            f"late={self.state.config.late_threshold:.2%}"
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original layers."""
        if self._original_layers is not None:
            setattr(self.transformer, self._layers_attr, self._original_layers)

            # Log final stats
            if self.state.config.enabled:
                stats = self.state.get_stats()
                logger.info(
                    f"[FBCache] Complete: {stats['skips']} skips, "
                    f"{stats['computes']} computes, "
                    f"ratio={stats['skip_ratio']:.1%}, "
                    f"est. speedup={stats['estimated_speedup']:.2f}x"
                )

                # Write log file if configured
                if self.state.config.log_file:
                    self.state.write_log_file()

        return False
