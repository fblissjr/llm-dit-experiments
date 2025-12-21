"""Vision YaRN position encoding for high-resolution generation.

This module implements Vision YaRN (Yet another RoPE extensioN) with DyPE
(Dynamic Position Extrapolation) support. Vision YaRN enables generating
images at resolutions higher than the model's training resolution by
dynamically scaling RoPE frequencies.

Key concepts:
- Beta mask: Blends linear scaling (low-freq) with NTK scaling (mid-freq)
- Gamma mask: Preserves highest frequencies untouched (stability)
- DyPE modulation: Scales mask parameters based on diffusion timestep

Based on ComfyUI-DyPE implementation.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch


def find_correction_factor(
    num_rotations: float,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> float:
    """Calculate the frequency index for a given rotation count.

    Args:
        num_rotations: Target number of rotations
        dim: Embedding dimension
        base: RoPE theta base
        max_position_embeddings: Maximum position length

    Returns:
        Frequency index where the given rotation count occurs
    """
    return (
        dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
    ) / (2 * math.log(base))


def find_correction_range(
    low_ratio: float,
    high_ratio: float,
    dim: int,
    base: float,
    ori_max_pe_len: int,
) -> Tuple[int, int]:
    """Find the frequency index range for mask application.

    Args:
        low_ratio: Lower bound ratio
        high_ratio: Upper bound ratio
        dim: Embedding dimension
        base: RoPE theta base
        ori_max_pe_len: Original maximum position length

    Returns:
        Tuple of (low_index, high_index) for mask application
    """
    low = int(np.floor(find_correction_factor(low_ratio, dim, base, ori_max_pe_len)))
    high = int(np.ceil(find_correction_factor(high_ratio, dim, base, ori_max_pe_len)))
    return max(low, 0), min(high, dim - 1)


def linear_ramp_mask(min_val: float, max_val: float, dim: int) -> torch.Tensor:
    """Create a linear ramp mask from 0 to 1.

    Args:
        min_val: Index where mask starts (value 0)
        max_val: Index where mask ends (value 1)
        dim: Dimension of the mask

    Returns:
        Tensor of shape (dim,) with values ramping from 0 to 1
    """
    if min_val == max_val:
        max_val += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (
        max_val - min_val
    )
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def find_newbase_ntk(dim: int, base: float, scale: float) -> float:
    """Calculate NTK-scaled theta base.

    Args:
        dim: Embedding dimension
        base: Original RoPE theta base
        scale: Scale factor

    Returns:
        New theta base for NTK scaling
    """
    return base * (scale ** (dim / (dim - 2)))


def get_1d_vision_yarn_pos_embed(
    dim: int,
    pos: torch.Tensor,
    theta: float,
    linear_scale: float,
    ntk_scale: float,
    ori_max_pe_len: int,
    dype: bool = True,
    current_timestep: float = 1.0,
    dype_scale: float = 2.0,
    dype_exponent: float = 2.0,
    override_mscale: float | None = None,
    use_real: bool = True,
    repeat_interleave_real: bool = True,
    freqs_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Vision YaRN position embeddings with optional DyPE modulation.

    This is the core Vision YaRN implementation that enables high-resolution
    generation by blending three frequency baselines using dual masks:
    1. Linear scaling for low frequencies (aspect ratio robust)
    2. NTK scaling for mid frequencies (extends range)
    3. Original frequencies for high frequencies (stability)

    Args:
        dim: Embedding dimension per axis
        pos: Position indices tensor of shape (..., seq_len)
        theta: RoPE base frequency
        linear_scale: Scale factor for linear (low-freq) component
        ntk_scale: Global scale factor for NTK component
        ori_max_pe_len: Original maximum position length (base patches)
        dype: Whether to apply DyPE timestep modulation
        current_timestep: Current diffusion timestep (0=clean, 1=noise)
        dype_scale: DyPE magnitude (lambda_s)
        dype_exponent: DyPE decay speed (lambda_t)
        override_mscale: Optional override for amplitude scaling
        use_real: Return real cos/sin instead of complex
        repeat_interleave_real: Interleave cos/sin for rotation matrices
        freqs_dtype: Data type for frequency computation

    Returns:
        Tuple of (cos, sin) tensors for position embeddings
    """
    device = pos.device
    linear_scale = max(linear_scale, 1.0)
    ntk_scale = max(ntk_scale, 1.0)

    # Base mask parameters
    beta_0, beta_1 = 1.25, 0.75  # Frequency correction range boundaries
    gamma_0, gamma_1 = 16, 2  # Additional frequency correction

    # DyPE: Scale mask parameters based on timestep
    if dype:
        k_t = dype_scale * (current_timestep**dype_exponent)
        beta_0 = beta_0**k_t
        beta_1 = beta_1**k_t
        gamma_0 = gamma_0**k_t
        gamma_1 = gamma_1**k_t

    # Compute three frequency baselines
    freq_indices = torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim
    freqs_base = 1.0 / (theta**freq_indices)  # Original frequencies
    freqs_linear = freqs_base / linear_scale  # Linear scaled (low-freq)

    # NTK scaled frequencies (high-freq)
    new_base = find_newbase_ntk(dim, theta, ntk_scale)
    if isinstance(new_base, torch.Tensor) and new_base.dim() > 0:
        new_base = new_base.view(-1, 1)
    freqs_ntk = 1.0 / torch.pow(
        torch.tensor(new_base, dtype=freqs_dtype, device=device), freq_indices
    )
    if isinstance(freqs_ntk, torch.Tensor) and freqs_ntk.dim() > 1:
        freqs_ntk = freqs_ntk.squeeze()

    # Beta mask: Blend linear and NTK frequencies
    low, high = find_correction_range(beta_0, beta_1, dim, theta, ori_max_pe_len)
    low, high = max(0, low), min(dim // 2, high)
    mask_beta = 1 - linear_ramp_mask(low, high, dim // 2).to(device).to(freqs_dtype)
    freqs = freqs_linear * (1 - mask_beta) + freqs_ntk * mask_beta

    # Gamma mask: Preserve highest frequencies
    low, high = find_correction_range(gamma_0, gamma_1, dim, theta, ori_max_pe_len)
    low, high = max(0, low), min(dim // 2, high)
    mask_gamma = 1 - linear_ramp_mask(low, high, dim // 2).to(device).to(freqs_dtype)
    freqs = freqs * (1 - mask_gamma) + freqs_base * mask_gamma

    # Compute position embeddings
    freqs = torch.einsum("...s,d->...sd", pos.to(freqs_dtype), freqs)

    if use_real and repeat_interleave_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=-1).float()
        freqs_sin = freqs.sin().repeat_interleave(2, dim=-1).float()

        # Amplitude compensation (mscale)
        if override_mscale is not None:
            mscale = torch.tensor(override_mscale, dtype=freqs_dtype, device=device)
        else:
            mscale_val = 1.0 + 0.1 * math.log(ntk_scale) / math.sqrt(ntk_scale)
            mscale = torch.tensor(mscale_val, dtype=freqs_dtype, device=device)

        return freqs_cos * mscale, freqs_sin * mscale
    elif use_real:
        return freqs.cos().float(), freqs.sin().float()
    else:
        return torch.polar(torch.ones_like(freqs), freqs)


def get_1d_yarn_pos_embed(
    dim: int,
    pos: torch.Tensor,
    theta: float,
    max_pe_len: torch.Tensor,
    ori_max_pe_len: int,
    dype: bool = True,
    current_timestep: float = 1.0,
    dype_scale: float = 2.0,
    dype_exponent: float = 2.0,
    use_aggressive_mscale: bool = False,
    use_real: bool = True,
    repeat_interleave_real: bool = True,
    freqs_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute standard YaRN position embeddings with optional DyPE modulation.

    Similar to Vision YaRN but uses a simpler single-scale approach.

    Args:
        dim: Embedding dimension per axis
        pos: Position indices tensor
        theta: RoPE base frequency
        max_pe_len: Current maximum position length
        ori_max_pe_len: Original maximum position length
        dype: Whether to apply DyPE timestep modulation
        current_timestep: Current diffusion timestep (0=clean, 1=noise)
        dype_scale: DyPE magnitude (lambda_s)
        dype_exponent: DyPE decay speed (lambda_t)
        use_aggressive_mscale: Use aggressive mscale formula
        use_real: Return real cos/sin instead of complex
        repeat_interleave_real: Interleave cos/sin for rotation matrices
        freqs_dtype: Data type for frequency computation

    Returns:
        Tuple of (cos, sin) tensors for position embeddings
    """
    device = pos.device
    scale = torch.clamp_min(max_pe_len / ori_max_pe_len, 1.0)

    beta_0, beta_1 = 1.25, 0.75
    gamma_0, gamma_1 = 16, 2

    freq_indices = torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim
    freqs_base = 1.0 / (theta**freq_indices)
    freqs_linear = 1.0 / torch.einsum(
        "..., f -> ... f", scale, theta**freq_indices
    )

    new_base = find_newbase_ntk(dim, theta, scale)
    if isinstance(new_base, torch.Tensor) and new_base.dim() > 0:
        new_base = new_base.view(-1, 1)
    freqs_ntk = 1.0 / torch.pow(new_base, freq_indices)
    if isinstance(freqs_ntk, torch.Tensor) and freqs_ntk.dim() > 1:
        freqs_ntk = freqs_ntk.squeeze()

    if dype:
        k_t = dype_scale * (current_timestep**dype_exponent)
        beta_0 = beta_0**k_t
        beta_1 = beta_1**k_t

    low, high = find_correction_range(beta_0, beta_1, dim, theta, ori_max_pe_len)
    low, high = max(0, low), min(dim // 2, high)
    freqs_mask = 1 - linear_ramp_mask(low, high, dim // 2).to(device).to(freqs_dtype)
    freqs = freqs_linear * (1 - freqs_mask) + freqs_ntk * freqs_mask

    if dype:
        k_t = dype_scale * (current_timestep**dype_exponent)
        gamma_0 = gamma_0**k_t
        gamma_1 = gamma_1**k_t

    low, high = find_correction_range(gamma_0, gamma_1, dim, theta, ori_max_pe_len)
    low, high = max(0, low), min(dim // 2, high)
    freqs_mask = 1 - linear_ramp_mask(low, high, dim // 2).to(device).to(freqs_dtype)
    freqs = freqs * (1 - freqs_mask) + freqs_base * freqs_mask

    freqs = torch.einsum("...s,d->...sd", pos.to(freqs_dtype), freqs)

    if use_real and repeat_interleave_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=-1).float()
        freqs_sin = freqs.sin().repeat_interleave(2, dim=-1).float()

        if use_aggressive_mscale:
            mscale = torch.where(
                scale <= 1.0,
                torch.tensor(1.0, device=device),
                0.1 * torch.log(scale) + 1.0,
            ).to(scale)
        else:
            mscale = torch.where(
                scale <= 1.0,
                torch.tensor(1.0, device=device),
                1.0 + 0.1 * torch.log(scale) / torch.sqrt(scale),
            ).to(scale)

        return freqs_cos * mscale, freqs_sin * mscale
    elif use_real:
        return freqs.cos().float(), freqs.sin().float()
    else:
        return torch.polar(torch.ones_like(freqs), freqs)


def get_1d_ntk_pos_embed(
    dim: int,
    pos: torch.Tensor,
    theta: float,
    ntk_factor: float = 1.0,
    use_real: bool = True,
    repeat_interleave_real: bool = True,
    freqs_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute NTK-scaled position embeddings.

    Simple NTK scaling of the RoPE theta base.

    Args:
        dim: Embedding dimension per axis
        pos: Position indices tensor
        theta: RoPE base frequency
        ntk_factor: NTK scaling factor
        use_real: Return real cos/sin instead of complex
        repeat_interleave_real: Interleave cos/sin for rotation matrices
        freqs_dtype: Data type for frequency computation

    Returns:
        Tuple of (cos, sin) tensors for position embeddings
    """
    device = pos.device
    theta_ntk = theta * ntk_factor
    freq_indices = torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim
    freqs = 1.0 / (theta_ntk**freq_indices)
    freqs = torch.einsum("...s,d->...sd", pos.to(freqs_dtype), freqs)

    if use_real and repeat_interleave_real:
        return (
            freqs.cos().repeat_interleave(2, dim=-1).float(),
            freqs.sin().repeat_interleave(2, dim=-1).float(),
        )
    elif use_real:
        return freqs.cos().float(), freqs.sin().float()
    else:
        return torch.polar(torch.ones_like(freqs), freqs)
