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
from functools import lru_cache
from typing import Protocol, runtime_checkable

import numpy
import scipy.stats
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


# =============================================================================
# Alternative Schedulers
# =============================================================================


class LinearQuadraticScheduler:
    """
    Scheduler with linear steps followed by quadratic steps.

    Produces a sigma schedule that transitions linearly up to a threshold,
    then follows a quadratic curve for the remaining steps. This can be
    useful for spending more steps in high-noise regions.

    Ported from: ltx_core.components.schedulers.LinearQuadraticScheduler
    """

    def execute(
        self,
        steps: int,
        threshold_noise: float = 0.025,
        linear_steps: int | None = None,
        **_kwargs,
    ) -> torch.FloatTensor:
        """
        Generate linear-quadratic sigma schedule.

        Args:
            steps: Number of denoising steps
            threshold_noise: Noise level at transition point (default: 0.025)
            linear_steps: Number of linear steps. If None, uses steps // 2.

        Returns:
            Tensor of sigma values with shape [steps + 1]
        """
        if steps == 1:
            return torch.FloatTensor([1.0, 0.0])

        if linear_steps is None:
            linear_steps = steps // 2

        # Linear portion
        linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]

        # Quadratic portion
        threshold_noise_step_diff = linear_steps - threshold_noise * steps
        quadratic_steps = steps - linear_steps
        quadratic_sigma_schedule = []

        if quadratic_steps > 0:
            quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
            linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
            const = quadratic_coef * (linear_steps**2)
            quadratic_sigma_schedule = [
                quadratic_coef * (i**2) + linear_coef * i + const
                for i in range(linear_steps, steps)
            ]

        # Combine and transform
        sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
        sigma_schedule = [1.0 - x for x in sigma_schedule]

        return torch.FloatTensor(sigma_schedule)


class BetaScheduler:
    """
    Scheduler using a beta distribution to sample timesteps.

    Based on: https://arxiv.org/abs/2407.12173

    Uses a beta distribution to non-uniformly sample timesteps, which can
    improve sample quality by spending more iterations at certain noise levels.

    Note: The number of steps in the output may be less than steps+1 due to
    deduplication of identical timesteps.

    Ported from: ltx_core.components.schedulers.BetaScheduler
    """

    shift = 2.37
    """Time shift parameter for flux-style shifting."""

    timesteps_length = 10000
    """Resolution of the precomputed sigma table."""

    def execute(
        self,
        steps: int,
        alpha: float = 0.6,
        beta: float = 0.6,
        **_kwargs,
    ) -> torch.FloatTensor:
        """
        Generate beta-distributed sigma schedule.

        Args:
            steps: Number of denoising steps
            alpha: Alpha parameter for beta distribution (default: 0.6)
            beta: Beta parameter for beta distribution (default: 0.6)

        Returns:
            Tensor of sigma values. Length may be less than steps+1 due to
            deduplication.
        """
        model_sampling_sigmas = _precalculate_model_sampling_sigmas(self.shift, self.timesteps_length)
        total_timesteps = len(model_sampling_sigmas) - 1

        # Sample timesteps using beta distribution
        ts = 1 - numpy.linspace(0, 1, steps, endpoint=False)
        ts = numpy.rint(scipy.stats.beta.ppf(ts, alpha, beta) * total_timesteps).tolist()

        # Remove duplicates while preserving order
        ts = list(dict.fromkeys(ts))

        # Map timesteps to sigma values
        sigmas = [float(model_sampling_sigmas[int(t)]) for t in ts] + [0.0]

        return torch.FloatTensor(sigmas)


# =============================================================================
# Helper Functions
# =============================================================================


@lru_cache(maxsize=5)
def _precalculate_model_sampling_sigmas(shift: float, timesteps_length: int) -> torch.Tensor:
    """
    Precompute sigma values for all timesteps using flux-style shifting.

    Args:
        shift: Time shift parameter (mu in flux_time_shift)
        timesteps_length: Number of timesteps to compute

    Returns:
        Tensor of sigma values for each timestep
    """
    timesteps = torch.arange(1, timesteps_length + 1, 1) / timesteps_length
    return torch.Tensor([flux_time_shift(shift, 1.0, t) for t in timesteps])


def flux_time_shift(mu: float, sigma: float, t: float) -> float:
    """
    Apply flux-style time shifting.

    Formula: exp(mu) / (exp(mu) + (1/t - 1)^sigma)

    Args:
        mu: Shift parameter (controls where the transition happens)
        sigma: Power parameter (controls sharpness of transition)
        t: Input time value in (0, 1]

    Returns:
        Shifted time value
    """
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


# =============================================================================
# Euler Diffusion Step (for sampling loop)
# =============================================================================


class EulerDiffusionStep:
    """
    Euler method for diffusion sampling.

    Performs a single step of the Euler ODE solver for flow matching:
        x_{t-1} = x_t + v * (sigma_{t-1} - sigma_t)

    This is the standard sampling method used in LTX-2.
    """

    @staticmethod
    def execute(
        sample: torch.Tensor,
        velocity: torch.Tensor,
        sigma: float,
        sigma_next: float,
    ) -> torch.Tensor:
        """
        Perform one Euler step.

        Args:
            sample: Current sample x_t
            velocity: Model velocity prediction v
            sigma: Current sigma value
            sigma_next: Next sigma value (sigma_next < sigma for denoising)

        Returns:
            Updated sample x_{t-1}
        """
        dt = sigma_next - sigma  # Negative for denoising
        return sample + velocity * dt


# =============================================================================
# CFG Guider
# =============================================================================


class CFGGuider:
    """
    Classifier-Free Guidance guider.

    Combines conditional and unconditional predictions to steer generation
    toward the conditioning.

    Formula: output = uncond + guidance_scale * (cond - uncond)
    """

    def __init__(self, guidance_scale: float = 7.5):
        """
        Initialize CFG guider.

        Args:
            guidance_scale: CFG scale. Higher values follow conditioning more
                closely but may reduce diversity. Default: 7.5
        """
        self.guidance_scale = guidance_scale

    def execute(
        self,
        cond: torch.Tensor,
        uncond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply classifier-free guidance.

        Args:
            cond: Conditional model prediction
            uncond: Unconditional model prediction

        Returns:
            Guided prediction
        """
        delta = (cond - uncond) * self.guidance_scale
        return uncond + delta
