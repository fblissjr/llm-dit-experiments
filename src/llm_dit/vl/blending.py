"""
Embedding blending utilities for VL conditioning.

This module provides functions for blending vision-conditioned embeddings
with text embeddings to control the influence of reference images on
Z-Image generation.

Key Finding (2025-12-12):
    Qwen3-VL text tokens have 0.999 correlation with Qwen3-4B per-dimension
    statistics, but image tokens have only 0.737 correlation with extreme
    outliers (some dimensions have 600x+ std ratio). Per-dimension
    normalization is critical for image token quality.
"""

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Default target statistics (measured from Qwen3-4B text embeddings)
# Updated based on comprehensive analysis with 10 diverse prompts.
DEFAULT_TARGET_STD = 61.0  # Updated from 70.0 based on empirical measurement

# Path to precomputed Qwen3-4B statistics
_STATS_PATH = Path(__file__).parent / "qwen3_4b_stats.npz"


@lru_cache(maxsize=1)
def _load_reference_stats() -> dict[str, np.ndarray]:
    """Load precomputed Qwen3-4B per-dimension statistics.

    Returns:
        Dictionary with 'per_dim_mean', 'per_dim_std', 'global_mean', 'global_std'
    """
    if not _STATS_PATH.exists():
        logger.warning(
            f"Reference stats not found at {_STATS_PATH}. "
            "Per-dimension normalization will fall back to global scaling."
        )
        return {}

    data = np.load(_STATS_PATH)
    return {
        "per_dim_mean": data["per_dim_mean"],
        "per_dim_std": data["per_dim_std"],
        "global_mean": float(data["global_mean"]),
        "global_std": float(data["global_std"]),
    }


def get_reference_stats() -> dict[str, np.ndarray]:
    """Get precomputed Qwen3-4B reference statistics.

    Returns:
        Dictionary with per-dimension and global statistics, or empty dict if unavailable.
    """
    return _load_reference_stats()


def scale_embeddings(
    embeddings: torch.Tensor,
    target_std: float = DEFAULT_TARGET_STD,
) -> torch.Tensor:
    """
    Scale embeddings to match target global statistics.

    VL embeddings typically have lower std (~13) than text embeddings (~61).
    Scaling helps align the distributions for better blending.

    Note:
        This uses global scaling only. For image tokens with extreme per-dimension
        outliers, use `normalize_per_dimension()` instead.

    Args:
        embeddings: Input embeddings to scale
        target_std: Target standard deviation (default: 61.0 from Qwen3-4B)

    Returns:
        Scaled embeddings
    """
    original_std = embeddings.std().item()
    if original_std <= 0:
        logger.warning("Embeddings have zero std, skipping scaling")
        return embeddings

    scale_factor = target_std / original_std
    scaled = embeddings * scale_factor

    logger.debug(
        f"Scaled embeddings: std {original_std:.2f} -> {scaled.std().item():.2f} "
        f"(factor: {scale_factor:.2f})"
    )

    return scaled


def normalize_per_dimension(
    embeddings: torch.Tensor,
    clip_outliers: bool = True,
    outlier_sigma: float = 3.0,
) -> torch.Tensor:
    """
    Normalize embeddings per-dimension to match Qwen3-4B statistics.

    This is critical for image tokens from Qwen3-VL, which have extreme
    per-dimension distribution mismatches (up to 600x std ratio for some dims).

    The normalization:
    1. Z-scores embeddings using their own per-dim mean/std
    2. Rescales to Qwen3-4B's per-dim mean/std

    Args:
        embeddings: Input embeddings (seq_len, hidden_dim)
        clip_outliers: If True, clip values beyond outlier_sigma before scaling
        outlier_sigma: Number of std devs to clip at (default: 3.0)

    Returns:
        Normalized embeddings matching Qwen3-4B per-dimension distribution
    """
    stats = _load_reference_stats()
    if not stats:
        logger.warning("No reference stats available, falling back to global scaling")
        return scale_embeddings(embeddings)

    ref_mean = torch.from_numpy(stats["per_dim_mean"]).to(
        device=embeddings.device, dtype=embeddings.dtype
    )
    ref_std = torch.from_numpy(stats["per_dim_std"]).to(
        device=embeddings.device, dtype=embeddings.dtype
    )

    # Compute input per-dim statistics
    input_mean = embeddings.mean(dim=0)
    input_std = embeddings.std(dim=0)

    # Avoid division by zero
    input_std = torch.clamp(input_std, min=1e-6)
    ref_std = torch.clamp(ref_std, min=1e-6)

    # Z-score using input stats
    z_scored = (embeddings - input_mean) / input_std

    # Optional: clip outliers to prevent extreme values
    if clip_outliers:
        z_scored = torch.clamp(z_scored, -outlier_sigma, outlier_sigma)

    # Rescale to reference distribution
    normalized = z_scored * ref_std + ref_mean

    logger.debug(
        f"Per-dim normalized: input_std range [{input_std.min():.1f}, {input_std.max():.1f}] "
        f"-> ref_std range [{ref_std.min():.1f}, {ref_std.max():.1f}]"
    )

    return normalized


def normalize_hybrid(
    embeddings: torch.Tensor,
    per_dim_weight: float = 0.5,
) -> torch.Tensor:
    """
    Hybrid normalization: blend per-dimension and global scaling.

    This is a softer approach than pure per-dimension normalization,
    useful when you want partial correction without fully overriding
    the input's distribution characteristics.

    Args:
        embeddings: Input embeddings
        per_dim_weight: Weight for per-dim normalization (0=global only, 1=per-dim only)

    Returns:
        Hybrid normalized embeddings
    """
    global_scaled = scale_embeddings(embeddings)
    per_dim_scaled = normalize_per_dimension(embeddings, clip_outliers=True)

    return per_dim_weight * per_dim_scaled + (1 - per_dim_weight) * global_scaled


def blend_embeddings(
    vl_emb: torch.Tensor,
    text_emb: torch.Tensor,
    alpha: float,
    match_lengths: bool = True,
) -> torch.Tensor:
    """
    Blend VL and text embeddings with linear interpolation.

    Args:
        vl_emb: Vision-conditioned embeddings from Qwen3-VL (seq, dim)
        text_emb: Pure text embeddings from text encoder (seq, dim)
        alpha: Blend ratio (0.0 = pure text, 1.0 = pure VL)
        match_lengths: If True, truncate to shorter sequence length

    Returns:
        Blended embeddings: alpha * vl_emb + (1 - alpha) * text_emb

    Raises:
        ValueError: If alpha is not in [0, 1] range
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Alpha must be in [0, 1], got {alpha}")

    # Short-circuit for pure cases
    if alpha == 0.0:
        return text_emb
    if alpha == 1.0:
        return vl_emb

    # Ensure same device and dtype
    device = text_emb.device
    dtype = text_emb.dtype
    vl_emb = vl_emb.to(device=device, dtype=dtype)

    # Match sequence lengths if needed
    if match_lengths and vl_emb.shape[0] != text_emb.shape[0]:
        min_len = min(vl_emb.shape[0], text_emb.shape[0])
        vl_truncated = vl_emb[:min_len]
        text_truncated = text_emb[:min_len]
        logger.debug(
            f"Matched sequence lengths: VL {vl_emb.shape[0]} -> {min_len}, "
            f"text {text_emb.shape[0]} -> {min_len}"
        )
    else:
        vl_truncated = vl_emb
        text_truncated = text_emb

    # Linear interpolation
    blended = alpha * vl_truncated + (1 - alpha) * text_truncated

    logger.debug(
        f"Blended embeddings (alpha={alpha}): "
        f"shape={blended.shape}, std={blended.std().item():.2f}"
    )

    return blended


def compute_blend_stats(
    vl_emb: torch.Tensor,
    text_emb: torch.Tensor,
) -> dict:
    """
    Compute statistics about embeddings for debugging.

    Args:
        vl_emb: Vision-conditioned embeddings
        text_emb: Text embeddings

    Returns:
        Dictionary with embedding statistics
    """
    return {
        "vl_shape": list(vl_emb.shape),
        "text_shape": list(text_emb.shape),
        "vl_mean": vl_emb.mean().item(),
        "vl_std": vl_emb.std().item(),
        "vl_min": vl_emb.min().item(),
        "vl_max": vl_emb.max().item(),
        "text_mean": text_emb.mean().item(),
        "text_std": text_emb.std().item(),
        "text_min": text_emb.min().item(),
        "text_max": text_emb.max().item(),
    }


# =============================================================================
# Outlier Dimension Masking
# =============================================================================
# VL image tokens have extreme per-dimension outliers that cause artifacts.
# Key outliers: dimension 396 (617x std ratio), dimension 4 (42x std ratio).


def mask_outlier_dimensions(
    embeddings: torch.Tensor,
    threshold: float = 10.0,
    mode: str = "zero",
    reference_stats: dict[str, np.ndarray] | None = None,
) -> tuple[torch.Tensor, dict]:
    """Mask dimensions with extreme std ratios vs reference.

    VL image tokens have extreme per-dimension outliers (e.g., dimension 396 has
    617x std ratio vs Qwen3-4B reference). This function masks those dimensions
    to reduce artifacts.

    Args:
        embeddings: VL embeddings (seq_len, hidden_dim)
        threshold: Max allowed std ratio before masking (default: 10.0)
        mode: How to handle outlier dimensions:
            - "zero": Zero out the dimension entirely
            - "clamp": Scale dimension to threshold level
            - "scale": Proportionally reduce dimension values
        reference_stats: Pre-loaded stats dict (loads from file if None)

    Returns:
        Tuple of (masked_embeddings, info_dict) where info_dict contains:
            - masked_dimensions: List of dimension indices that were masked
            - ratios: Dict mapping dimension index to its std ratio
            - mode: The masking mode used
            - threshold: The threshold used
    """
    # Load reference stats if not provided
    if reference_stats is None:
        reference_stats = _load_reference_stats()

    if not reference_stats:
        logger.warning("No reference stats available, skipping outlier masking")
        return embeddings, {"masked_dimensions": [], "ratios": {}}

    ref_std = torch.from_numpy(reference_stats["per_dim_std"]).to(
        device=embeddings.device, dtype=embeddings.dtype
    )

    # Compute per-dimension std of input
    input_std = embeddings.std(dim=0)

    # Avoid division by zero
    ref_std_safe = torch.clamp(ref_std, min=1e-6)

    # Calculate ratios
    ratios = input_std / ref_std_safe

    # Identify outlier dimensions
    outlier_mask = ratios > threshold
    outlier_indices = torch.where(outlier_mask)[0].cpu().tolist()

    if not outlier_indices:
        logger.debug(f"No outlier dimensions found above threshold {threshold}")
        return embeddings, {
            "masked_dimensions": [],
            "ratios": {},
            "mode": mode,
            "threshold": threshold,
        }

    # Get ratios for outlier dimensions
    ratio_dict = {idx: ratios[idx].item() for idx in outlier_indices}

    # Clone to avoid modifying input
    masked = embeddings.clone()

    if mode == "zero":
        # Zero out outlier dimensions entirely
        masked[:, outlier_mask] = 0
        logger.debug(f"Zeroed {len(outlier_indices)} outlier dimensions: {outlier_indices}")

    elif mode == "clamp":
        # Scale outlier dimensions down to threshold level
        # New value = value * (threshold * ref_std) / input_std
        for idx in outlier_indices:
            scale = (threshold * ref_std_safe[idx]) / input_std[idx]
            masked[:, idx] = masked[:, idx] * scale
        logger.debug(f"Clamped {len(outlier_indices)} outlier dimensions to {threshold}x level")

    elif mode == "scale":
        # Proportionally reduce: scale = threshold / actual_ratio
        # This gradually reduces high outliers more than low outliers
        for idx in outlier_indices:
            scale = threshold / ratios[idx]
            masked[:, idx] = masked[:, idx] * scale
        logger.debug(f"Scaled {len(outlier_indices)} outlier dimensions proportionally")

    else:
        raise ValueError(f"Unknown masking mode: {mode}. Use 'zero', 'clamp', or 'scale'")

    return masked, {
        "masked_dimensions": outlier_indices,
        "ratios": ratio_dict,
        "mode": mode,
        "threshold": threshold,
    }


def get_outlier_dimensions(
    embeddings: torch.Tensor,
    threshold: float = 10.0,
    reference_stats: dict[str, np.ndarray] | None = None,
) -> list[tuple[int, float]]:
    """Return list of (dim_index, ratio) for dimensions exceeding threshold.

    Useful for analysis without modifying embeddings.

    Args:
        embeddings: VL embeddings (seq_len, hidden_dim)
        threshold: Std ratio threshold (default: 10.0)
        reference_stats: Pre-loaded stats dict (loads from file if None)

    Returns:
        List of (dimension_index, std_ratio) tuples, sorted by ratio descending
    """
    if reference_stats is None:
        reference_stats = _load_reference_stats()

    if not reference_stats:
        logger.warning("No reference stats available")
        return []

    ref_std = torch.from_numpy(reference_stats["per_dim_std"]).to(
        device=embeddings.device, dtype=embeddings.dtype
    )

    # Compute per-dimension std of input
    input_std = embeddings.std(dim=0)

    # Avoid division by zero
    ref_std_safe = torch.clamp(ref_std, min=1e-6)

    # Calculate ratios
    ratios = input_std / ref_std_safe

    # Find outliers
    outlier_mask = ratios > threshold
    outlier_indices = torch.where(outlier_mask)[0].cpu().tolist()

    # Build result list sorted by ratio (highest first)
    result = [(idx, ratios[idx].item()) for idx in outlier_indices]
    result.sort(key=lambda x: x[1], reverse=True)

    return result


# =============================================================================
# Advanced Blending Strategies
# =============================================================================
# These address the issue of VL embeddings overriding text content, not just style.


def blend_style_only(
    vl_emb: torch.Tensor,
    text_emb: torch.Tensor,
    style_alpha: float = 0.3,
    style_dims: tuple[int, int] | None = None,
) -> torch.Tensor:
    """
    Blend only style-related dimensions, preserving text semantic content.

    This is an experimental approach that assumes style information is encoded
    differently than semantic content in the embedding space.

    Args:
        vl_emb: Vision-conditioned embeddings
        text_emb: Text embeddings
        style_alpha: Alpha for style dimensions only
        style_dims: Optional (start, end) indices for style dimensions.
                   If None, uses last 1/3 of dimensions as heuristic.

    Returns:
        Blended embeddings with style from VL, content from text
    """
    if style_dims is None:
        # Heuristic: later dimensions often encode more abstract/style info
        dim = text_emb.shape[-1]
        style_dims = (dim * 2 // 3, dim)

    # Match lengths
    min_len = min(vl_emb.shape[0], text_emb.shape[0])
    vl_truncated = vl_emb[:min_len].clone()
    text_truncated = text_emb[:min_len].clone()

    # Start with text embeddings (preserves semantic content)
    blended = text_truncated.clone()

    # Only blend in the style dimensions
    start, end = style_dims
    blended[:, start:end] = (
        style_alpha * vl_truncated[:, start:end]
        + (1 - style_alpha) * text_truncated[:, start:end]
    )

    logger.debug(
        f"Style-only blend: alpha={style_alpha} for dims [{start}:{end}], "
        f"keeping text content in dims [0:{start}]"
    )

    return blended


def blend_per_token(
    vl_emb: torch.Tensor,
    text_emb: torch.Tensor,
    token_alphas: torch.Tensor | list[float],
) -> torch.Tensor:
    """
    Apply different alpha values per token position.

    This allows fine-grained control: use more VL influence for some tokens
    (e.g., style tokens) and more text influence for others (e.g., subject).

    Args:
        vl_emb: Vision-conditioned embeddings (seq, dim)
        text_emb: Text embeddings (seq, dim)
        token_alphas: Per-token alpha values (seq,) - 0.0=text, 1.0=VL

    Returns:
        Per-token blended embeddings
    """
    # Match lengths
    min_len = min(vl_emb.shape[0], text_emb.shape[0])
    vl_truncated = vl_emb[:min_len]
    text_truncated = text_emb[:min_len]

    # Convert to tensor if list
    if isinstance(token_alphas, list):
        token_alphas = torch.tensor(token_alphas)

    # Truncate alphas to match
    token_alphas = token_alphas[:min_len]

    # Expand alphas to match embedding dim: (seq,) -> (seq, 1)
    alphas = token_alphas.unsqueeze(-1).to(
        device=text_truncated.device, dtype=text_truncated.dtype
    )

    blended = alphas * vl_truncated + (1 - alphas) * text_truncated

    logger.debug(
        f"Per-token blend: {min_len} tokens, "
        f"alpha range [{token_alphas.min():.2f}, {token_alphas.max():.2f}]"
    )

    return blended


def create_graduated_alpha(
    seq_len: int,
    start_alpha: float = 0.0,
    end_alpha: float = 0.5,
    curve: str = "linear",
) -> torch.Tensor:
    """
    Create graduated per-token alpha values.

    Useful for applying more VL influence to later tokens (often style/mood)
    while preserving text influence for early tokens (often subject/content).

    Args:
        seq_len: Sequence length
        start_alpha: Alpha for first token
        end_alpha: Alpha for last token
        curve: Interpolation curve ("linear", "ease_in", "ease_out")

    Returns:
        Tensor of per-token alpha values
    """
    t = torch.linspace(0, 1, seq_len)

    if curve == "linear":
        alphas = start_alpha + t * (end_alpha - start_alpha)
    elif curve == "ease_in":
        # Quadratic ease in (slow start, fast end)
        alphas = start_alpha + (t ** 2) * (end_alpha - start_alpha)
    elif curve == "ease_out":
        # Quadratic ease out (fast start, slow end)
        alphas = start_alpha + (1 - (1 - t) ** 2) * (end_alpha - start_alpha)
    else:
        raise ValueError(f"Unknown curve: {curve}")

    return alphas


def blend_attention_weighted(
    vl_emb: torch.Tensor,
    text_emb: torch.Tensor,
    alpha: float,
    attention_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Blend using attention-weighted importance.

    Tokens that are more "important" to the text prompt get lower VL influence,
    preserving semantic content while allowing style transfer on less important tokens.

    Args:
        vl_emb: Vision-conditioned embeddings
        text_emb: Text embeddings
        alpha: Base alpha value
        attention_weights: Per-token importance weights (seq,). If None, uses uniform.

    Returns:
        Attention-weighted blended embeddings
    """
    min_len = min(vl_emb.shape[0], text_emb.shape[0])
    vl_truncated = vl_emb[:min_len]
    text_truncated = text_emb[:min_len]

    if attention_weights is None:
        # No attention weights = uniform blend
        return blend_embeddings(vl_emb, text_emb, alpha)

    # Normalize weights to [0, 1]
    weights = attention_weights[:min_len]
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)

    # Invert: high importance = low VL alpha
    # This preserves text semantics for important tokens
    per_token_alpha = alpha * (1 - weights * 0.8)  # Scale down by up to 80%

    return blend_per_token(vl_truncated, text_truncated, per_token_alpha)
