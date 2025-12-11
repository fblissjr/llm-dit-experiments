"""
Embedding blending utilities for VL conditioning.

This module provides functions for blending vision-conditioned embeddings
with text embeddings to control the influence of reference images on
Z-Image generation.
"""

import logging

import torch

logger = logging.getLogger(__name__)

# Default target statistics (measured from Qwen3-4B text embeddings)
# Note: This is an approximation. Actual std varies by prompt but ~70 is typical.
# The original 58.75 value was incorrect and caused corrupted outputs.
DEFAULT_TARGET_STD = 70.0


def scale_embeddings(
    embeddings: torch.Tensor,
    target_std: float = DEFAULT_TARGET_STD,
) -> torch.Tensor:
    """
    Scale embeddings to match target statistics.

    VL embeddings typically have lower std (~13) than text embeddings (~58).
    Scaling helps align the distributions for better blending.

    Args:
        embeddings: Input embeddings to scale
        target_std: Target standard deviation (default: 58.75 from text embeddings)

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
