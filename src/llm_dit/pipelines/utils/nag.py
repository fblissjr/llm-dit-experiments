"""
Normalized Attention Guidance (NAG) for LTX-2 Pipeline.

Last Updated: 2026-01-16

Ported from ComfyUI-KJNodes/nodes/model_optimization_nodes.py (lines 1237-1264).

NAG improves classifier-free guidance quality by normalizing attention outputs
to prevent divergence at high CFG scales. The core problem: when CFG scale is
high, the guidance term (positive - negative) can explode in magnitude, causing
artifacts like oversaturation and distortion.

Algorithm:
1. Compute separate attention outputs for positive and negative prompts
2. Apply CFG formula: nag_guidance = positive * scale - negative * (scale - 1)
3. Compute L1 norms of positive and guided outputs
4. If guided norm exceeds threshold (tau × positive norm), adaptively clamp
5. Blend guided and positive outputs using alpha parameter

Key Parameters:
- nag_scale (default 11.0): Strength of negative guidance (higher = stronger effect)
- nag_alpha (default 0.25): Blend ratio between guided and positive (0=all positive)
- nag_tau (default 2.5): Maximum allowed norm ratio (prevents divergence)

Note: "clamp" here means torch.clamp() limiting values, NOT CLIP the model.

Reference:
    ComfyUI-KJNodes by Kijai
    https://github.com/kijai/ComfyUI-KJNodes

Example:
    # In attention computation:
    x_positive = attention(query, k_positive, v_positive)
    x_negative = attention(query, k_negative, v_negative)
    x_guided = normalized_attention_guidance(
        x_positive, x_negative,
        nag_scale=11.0, nag_alpha=0.25, nag_tau=2.5
    )
"""

from dataclasses import dataclass
from typing import Optional, Callable

import torch


@dataclass
class NAGConfig:
    """Configuration for Normalized Attention Guidance."""

    # Whether NAG is enabled
    enabled: bool = True

    # Strength of negative guidance (higher = stronger CFG effect)
    # Typical range: 5.0 - 15.0
    scale: float = 11.0

    # Balance between guided and positive-only output
    # 0.0 = pure positive, 1.0 = pure guided
    # Typical range: 0.1 - 0.5
    alpha: float = 0.25

    # Maximum allowed norm ratio (guided_norm / positive_norm)
    # Values exceeding this are clamped to prevent divergence
    # Typical range: 1.5 - 5.0
    tau: float = 2.5

    # Start step (0-indexed) - skip early noisy steps
    start_step: int = 0

    # End step - stop before final refinement steps
    # -1 = apply until end
    end_step: int = -1


def normalized_attention_guidance(
    x_positive: torch.Tensor,
    x_negative: torch.Tensor,
    nag_scale: float = 11.0,
    nag_alpha: float = 0.25,
    nag_tau: float = 2.5,
) -> torch.Tensor:
    """
    Apply Normalized Attention Guidance to attention outputs.

    This function implements the core NAG algorithm from ComfyUI-KJNodes.
    It normalizes the CFG guidance to prevent attention divergence at
    high guidance scales.

    Args:
        x_positive: Attention output for positive prompt [B, H, S, D] or [B, S, D]
        x_negative: Attention output for negative prompt (same shape)
        nag_scale: CFG scale for guidance (higher = stronger negative influence)
        nag_alpha: Blend ratio (0=all positive, 1=all guided)
        nag_tau: Maximum norm ratio threshold for clamping

    Returns:
        Normalized guided attention output with same shape as inputs
    """
    # Standard CFG guidance formula
    # guidance = positive * scale - negative * (scale - 1)
    # This amplifies positive while suppressing negative
    nag_guidance = x_positive * nag_scale - x_negative * (nag_scale - 1)

    # Compute L1 norms for normalization
    # L1 norm = sum of absolute values, more robust to outliers than L2
    norm_positive = torch.norm(x_positive, p=1, dim=-1, keepdim=True)
    norm_guidance = torch.norm(nag_guidance, p=1, dim=-1, keepdim=True)

    # Compute scale ratio with numerical stability
    # Handle edge case where norm_guidance could be 0
    scale_ratio = torch.nan_to_num(
        norm_guidance / (norm_positive + 1e-7),
        nan=10.0,
        posinf=10.0,
        neginf=0.0,
    )

    # Adaptive clamping: if scale exceeds tau, clamp to prevent divergence
    # This is the key insight - CFG can cause norms to explode, so we
    # adaptively rescale to keep within bounds
    mask = scale_ratio > nag_tau
    adjustment = (norm_positive * nag_tau) / (norm_guidance + 1e-7)

    # Apply adjustment only where mask is True
    nag_guidance = torch.where(mask, nag_guidance * adjustment, nag_guidance)

    # Final blend between guided and positive-only
    # alpha=0.25 means 25% guided, 75% positive (conservative default)
    output = nag_guidance * nag_alpha + x_positive * (1 - nag_alpha)

    return output


def normalized_attention_guidance_batched(
    x_cond: torch.Tensor,
    x_uncond: torch.Tensor,
    cfg_scale: float = 3.5,
    nag_scale: float = 11.0,
    nag_alpha: float = 0.25,
    nag_tau: float = 2.5,
) -> torch.Tensor:
    """
    Apply NAG with standard CFG batch structure.

    This variant is designed for use with diffusers-style batched CFG where
    conditional and unconditional predictions are processed together.

    Args:
        x_cond: Conditional (positive) prediction
        x_uncond: Unconditional (negative) prediction
        cfg_scale: Standard CFG guidance scale (for comparison/logging)
        nag_scale: NAG-specific scale parameter
        nag_alpha: Blend ratio
        nag_tau: Clamping threshold

    Returns:
        NAG-normalized output
    """
    return normalized_attention_guidance(
        x_positive=x_cond,
        x_negative=x_uncond,
        nag_scale=nag_scale,
        nag_alpha=nag_alpha,
        nag_tau=nag_tau,
    )


class NAGEnhancer:
    """
    NAG enhancement wrapper for transformer attention.

    This class provides a stateful interface for applying NAG enhancement
    across diffusion steps, with optional step-based enabling/disabling.

    Example:
        nag = NAGEnhancer(nag_scale=11.0, nag_alpha=0.25)

        # In diffusion loop:
        for step in range(num_steps):
            nag.set_step(step)
            # After separate attention computation:
            output = nag.enhance(x_positive, x_negative)
    """

    def __init__(
        self,
        nag_scale: float = 11.0,
        nag_alpha: float = 0.25,
        nag_tau: float = 2.5,
        start_step: int = 0,
        end_step: int = -1,
        total_steps: Optional[int] = None,
    ):
        """
        Initialize NAG enhancer.

        Args:
            nag_scale: CFG scale for guidance
            nag_alpha: Blend ratio between guided and positive
            nag_tau: Clamping threshold
            start_step: First step to apply NAG
            end_step: Last step to apply NAG (-1 = all)
            total_steps: Total diffusion steps (for end_step=-1 resolution)
        """
        self.nag_scale = nag_scale
        self.nag_alpha = nag_alpha
        self.nag_tau = nag_tau
        self.start_step = start_step
        self.end_step = end_step
        self.total_steps = total_steps
        self.current_step = 0

    def set_step(self, step: int) -> None:
        """Update current diffusion step."""
        self.current_step = step

    def set_total_steps(self, total_steps: int) -> None:
        """Set total steps (useful for resolving end_step=-1)."""
        self.total_steps = total_steps

    def should_apply(self, step: Optional[int] = None) -> bool:
        """Check if NAG should be applied at current/given step."""
        step = step if step is not None else self.current_step

        if step < self.start_step:
            return False

        end = self.end_step
        if end == -1:
            # Apply until end
            return True
        return step <= end

    def enhance(
        self,
        x_positive: torch.Tensor,
        x_negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply NAG enhancement to attention outputs.

        Args:
            x_positive: Attention output for positive prompt
            x_negative: Attention output for negative prompt

        Returns:
            NAG-enhanced output
        """
        if not self.should_apply():
            return x_positive

        return normalized_attention_guidance(
            x_positive=x_positive,
            x_negative=x_negative,
            nag_scale=self.nag_scale,
            nag_alpha=self.nag_alpha,
            nag_tau=self.nag_tau,
        )

    def __repr__(self) -> str:
        return (
            f"NAGEnhancer(scale={self.nag_scale}, "
            f"alpha={self.nag_alpha}, tau={self.nag_tau}, "
            f"steps={self.start_step}-{self.end_step})"
        )


def create_nag_cfg_function(
    nag_scale: float = 11.0,
    nag_alpha: float = 0.25,
    nag_tau: float = 2.5,
) -> Callable:
    """
    Create a NAG-enhanced CFG function for use as a post-CFG callback.

    This can be used to replace the standard CFG combination with
    NAG-normalized guidance.

    Args:
        nag_scale: CFG scale for guidance
        nag_alpha: Blend ratio
        nag_tau: Clamping threshold

    Returns:
        Function that takes (cond, uncond, cfg_scale) and returns NAG output
    """

    def nag_cfg_function(
        cond: torch.Tensor,
        uncond: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        """Apply NAG-enhanced CFG."""
        return normalized_attention_guidance(
            x_positive=cond,
            x_negative=uncond,
            nag_scale=nag_scale,
            nag_alpha=nag_alpha,
            nag_tau=nag_tau,
        )

    return nag_cfg_function
