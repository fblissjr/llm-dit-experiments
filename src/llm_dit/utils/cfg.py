"""CFG (Classifier-Free Guidance) utilities.

This module provides shared CFG normalization, truncation, and dynamic shift
functions used across all pipelines (Z-Image, Qwen-Image, etc.).

Last updated: 2025-01-05
"""

from typing import Literal

import torch

CFGNormMode = Literal["clamp", "match", "none"]


def apply_cfg_normalization(
    pred: torch.Tensor,
    pos: torch.Tensor,
    cfg_normalization: float,
    cfg_norm_mode: CFGNormMode = "clamp",
) -> torch.Tensor:
    """Apply CFG normalization to combined prediction.

    Prevents over-saturation and artifacts from high CFG scales by constraining
    the prediction magnitude relative to the positive prediction.

    Args:
        pred: Combined prediction (pos + scale * (pos - neg))
        pos: Positive (conditional) prediction
        cfg_normalization: Normalization strength/factor.
            - For "clamp" mode: Maximum ratio of pred_norm to pos_norm (e.g., 1.0)
            - For "match" mode: Ignored (always scales to match pos_norm)
            - 0 or negative: Disabled
        cfg_norm_mode: Normalization mode:
            - "clamp": Clamp pred norm to cfg_normalization * pos_norm
            - "match": Scale pred to exactly match pos_norm (DiffSynth-style)
            - "none": No normalization (same as cfg_normalization <= 0)

    Returns:
        Normalized prediction tensor

    Example:
        >>> pred = pos + 7.5 * (pos - neg)  # CFG scale 7.5
        >>> pred_normalized = apply_cfg_normalization(pred, pos, 1.0, "clamp")
    """
    if cfg_normalization <= 0 or cfg_norm_mode == "none":
        return pred

    pos_norm = torch.linalg.vector_norm(pos)
    pred_norm = torch.linalg.vector_norm(pred)

    # Avoid division by zero and degenerate cases
    MIN_NORM = 1e-6
    pos_norm = torch.where(pos_norm < MIN_NORM, torch.ones_like(pos_norm), pos_norm)
    pred_norm = torch.where(pred_norm < MIN_NORM, torch.ones_like(pred_norm), pred_norm)

    if cfg_norm_mode == "match":
        # DiffSynth-style: directly scale pred to match pos norm
        scale_factor = pos_norm / pred_norm
    else:  # "clamp" (default)
        # Original: clamp pred norm to cfg_normalization * pos_norm
        max_allowed_norm = pos_norm * cfg_normalization
        scale_factor = torch.clamp(max_allowed_norm / pred_norm, max=1.0)

    return pred * scale_factor


def apply_cfg_truncation(
    progress: float,
    cfg_truncation: float,
) -> bool:
    """Determine if CFG should be disabled based on denoising progress.

    CFG truncation disables guidance in the final stages of denoising,
    which can reduce over-saturation and speed up inference (by skipping
    the negative prompt forward pass).

    Args:
        progress: Current denoising progress (0.0 = start, 1.0 = end)
        cfg_truncation: Progress threshold after which CFG is disabled.
            - 1.0: Never truncate (CFG active for all steps)
            - 0.7: Truncate after 70% progress (CFG off for last 30%)
            - 0.0: Truncate immediately (equivalent to CFG=1.0)

    Returns:
        True if CFG should be DISABLED (truncated), False if CFG should be active

    Example:
        >>> for i, t in enumerate(timesteps):
        ...     progress = i / len(timesteps)
        ...     if apply_cfg_truncation(progress, cfg_truncation=0.7):
        ...         cfg_scale = 1.0  # Disable CFG
        ...     else:
        ...         cfg_scale = 7.5  # Normal CFG
    """
    if cfg_truncation >= 1.0:
        return False  # Never truncate
    return progress > cfg_truncation


def get_cfg_scale_with_truncation(
    cfg_scale: float,
    progress: float,
    cfg_truncation: float,
) -> float:
    """Get effective CFG scale considering truncation.

    Convenience function that returns the actual CFG scale to use,
    accounting for truncation.

    Args:
        cfg_scale: Base CFG scale (e.g., 7.5)
        progress: Current denoising progress (0.0 = start, 1.0 = end)
        cfg_truncation: Progress threshold after which CFG is disabled

    Returns:
        Effective CFG scale (1.0 if truncated, cfg_scale otherwise)

    Example:
        >>> effective_cfg = get_cfg_scale_with_truncation(7.5, 0.8, 0.7)
        >>> # Returns 1.0 because 0.8 > 0.7 (truncated)
    """
    if apply_cfg_truncation(progress, cfg_truncation):
        return 1.0
    return cfg_scale


def calculate_dynamic_shift(
    seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Calculate dynamic shift for flow matching scheduler based on sequence length.

    Higher resolutions benefit from larger scheduler shifts. This function
    linearly interpolates the shift value based on sequence length.

    There are two formulations:

    1. DiffSynth original (simpler):
        shift = (seq_len / base_seq_len) ** 0.5

    2. DiffSynth extended (with bounds):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        shift = seq_len * m + b

    This function implements the extended version for more control.

    Args:
        seq_len: Current sequence length (latent H * W / patch_size^2)
            For Z-Image: (H/16) * (W/16) / 4 = H*W/1024
            For 1024x1024: 1024*1024/1024 = 1024
        base_seq_len: Base sequence length for minimum shift (default: 256)
        max_seq_len: Maximum sequence length for maximum shift (default: 4096)
        base_shift: Shift value at base_seq_len (default: 0.5)
        max_shift: Shift value at max_seq_len (default: 1.15)

    Returns:
        Dynamic shift value, clamped to [base_shift, max_shift]

    Example:
        >>> # 512x512 image: seq_len = 256
        >>> shift = calculate_dynamic_shift(256)  # Returns ~0.5
        >>> # 2048x2048 image: seq_len = 4096
        >>> shift = calculate_dynamic_shift(4096)  # Returns ~1.15
    """
    # Linear interpolation
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    shift = seq_len * m + b

    # Clamp to valid range
    return max(base_shift, min(max_shift, shift))


def calculate_dynamic_shift_simple(
    seq_len: int,
    base_seq_len: int = 256,
) -> float:
    """Calculate dynamic shift using the simple DiffSynth formula.

    This is the original DiffSynth formulation:
        shift = (seq_len / base_seq_len) ** 0.5

    Args:
        seq_len: Current sequence length
        base_seq_len: Base sequence length (default: 256)

    Returns:
        Dynamic shift value (unbounded)
    """
    return (seq_len / base_seq_len) ** 0.5
