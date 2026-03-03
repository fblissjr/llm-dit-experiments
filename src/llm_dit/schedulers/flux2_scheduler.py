"""
FLUX.2 timestep scheduler with SNR-based shifting.

Last Updated: 2026-03-03

Provides resolution-aware timestep schedules for FLUX.2 denoising.
Higher resolution images get different shift parameters via empirical
mu computation.
"""

import math

import torch


def generalized_time_snr_shift(t: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """
    Apply SNR-based timestep shift.

    Args:
        t: Linear timesteps in [0, 1]
        mu: Shift parameter (computed from image size)
        sigma: Scale parameter (typically 1.0)

    Returns:
        Shifted timesteps
    """
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """
    Compute empirical mu parameter for timestep shifting.

    Higher resolution images need different shift schedules.

    Args:
        image_seq_len: Number of image tokens
        num_steps: Number of denoising steps

    Returns:
        Computed mu value
    """
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


def get_schedule(num_steps: int, image_seq_len: int) -> list[float]:
    """
    Generate timestep schedule with SNR-based shifting.

    Args:
        num_steps: Number of denoising steps
        image_seq_len: Number of image tokens

    Returns:
        List of timesteps from 1.0 to ~0
    """
    mu = compute_empirical_mu(image_seq_len, num_steps)
    timesteps = torch.linspace(1, 0, num_steps + 1)
    timesteps = generalized_time_snr_shift(timesteps, mu, 1.0)
    return timesteps.tolist()
