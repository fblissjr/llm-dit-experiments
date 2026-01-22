"""
LTX-2 Schedulers for diffusion sampling.

Last Updated: 2026-01-18

Pure PyTorch implementations of LTX-2 schedulers, ported from the official
ltx-core repository with full attribution.

Original source: https://github.com/Lightricks/LTX-2
License: LTX-2 Community License
Copyright (c) 2025 Lightricks Ltd.

This file is a modified derivative work. Key changes:
- Removed dependency on ltx_core.components.protocols
- Added Python Protocol for type checking
- Added additional documentation and type hints
- Integrated with llm-dit scheduler patterns

Usage:
    from llm_dit.schedulers import LTX2Scheduler

    scheduler = LTX2Scheduler()
    sigmas = scheduler.execute(steps=30, latent=latent_tensor)

    # Use sigmas for diffusion sampling loop
    for i, sigma in enumerate(sigmas[:-1]):
        sigma_next = sigmas[i + 1]
        # ... denoising step
"""

import math
from typing import Protocol, runtime_checkable

import torch


# =============================================================================
# Constants from official LTX-2 implementation
# =============================================================================

BASE_SHIFT_ANCHOR = 1024
"""Token count for base shift value."""

MAX_SHIFT_ANCHOR = 4096
"""Token count for max shift value."""


# =============================================================================
# Protocol Definition
# =============================================================================


@runtime_checkable
class SchedulerProtocol(Protocol):
    """
    Protocol for schedulers that provide a sigma schedule tensor.

    Schedulers generate sigma values for each step of the diffusion
    denoising process. The returned tensor has shape [steps + 1] where
    sigmas[0] is the starting noise level and sigmas[-1] is typically 0.
    """

    def execute(self, steps: int, **kwargs) -> torch.FloatTensor:
        """
        Generate sigma schedule for the given number of steps.

        Args:
            steps: Number of denoising steps
            **kwargs: Scheduler-specific parameters

        Returns:
            Tensor of sigma values with shape [steps + 1]
        """
        ...


# =============================================================================
# LTX-2 Default Scheduler
# =============================================================================


class LTX2Scheduler:
    """
    Default scheduler for LTX-2 diffusion sampling.

    Generates a sigma schedule with token-count-dependent shifting and optional
    stretching to a terminal value. This scheduler adapts the noise schedule
    based on the number of latent tokens, which varies with video resolution
    and frame count.

    The shift is computed as a linear interpolation between base_shift and
    max_shift based on the number of tokens:
        shift = base_shift + (max_shift - base_shift) * (tokens - 1024) / (4096 - 1024)

    For a 768x512 @ 33 frames video:
        - Latent tokens: 5 * 16 * 24 = 1920
        - Shift ≈ 1.24 (between base 0.95 and max 2.05)

    Ported from: ltx_core.components.schedulers.LTX2Scheduler
    """

    def execute(
        self,
        steps: int,
        latent: torch.Tensor | None = None,
        max_shift: float = 2.05,
        base_shift: float = 0.95,
        stretch: bool = True,
        terminal: float = 0.1,
        **_kwargs,
    ) -> torch.FloatTensor:
        """
        Generate sigma schedule for LTX-2 diffusion.

        Args:
            steps: Number of denoising steps
            latent: Latent tensor to determine token count. Shape: [B, T, H, W, C]
                   or [B, tokens, C]. If None, uses MAX_SHIFT_ANCHOR tokens.
            max_shift: Shift value at MAX_SHIFT_ANCHOR tokens (default: 2.05)
            base_shift: Shift value at BASE_SHIFT_ANCHOR tokens (default: 0.95)
            stretch: Whether to stretch schedule so final sigma equals terminal
            terminal: Final sigma value when stretch=True (default: 0.1)

        Returns:
            Tensor of sigma values with shape [steps + 1]
        """
        # Determine token count from latent shape
        tokens = math.prod(latent.shape[2:]) if latent is not None else MAX_SHIFT_ANCHOR

        # Linear sigma schedule from 1.0 to 0.0
        sigmas = torch.linspace(1.0, 0.0, steps + 1)

        # Compute shift based on token count (linear interpolation)
        x1 = BASE_SHIFT_ANCHOR
        x2 = MAX_SHIFT_ANCHOR
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        sigma_shift = tokens * mm + b

        # Apply shifted sigmoid transformation
        # This warps the linear schedule to front-load denoising
        power = 1
        sigmas = torch.where(
            sigmas != 0,
            math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1) ** power),
            0,
        )

        # Stretch sigmas so final value matches terminal
        if stretch:
            non_zero_mask = sigmas != 0
            non_zero_sigmas = sigmas[non_zero_mask]
            one_minus_z = 1.0 - non_zero_sigmas
            scale_factor = one_minus_z[-1] / (1.0 - terminal)
            stretched = 1.0 - (one_minus_z / scale_factor)
            sigmas[non_zero_mask] = stretched

        return sigmas.to(torch.float32)
