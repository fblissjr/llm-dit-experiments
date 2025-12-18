"""
Long prompt handling utilities for Z-Image.

The Z-Image DiT transformer has a maximum text sequence length of 1504 tokens
due to RoPE position encoding limits. The config specifies axes_lens=[1536, 512, 512]
but the actual working limit is 1504 (47 * 32, where 32 is axes_dims[0]).

This module provides strategies to handle longer prompts without truncation:
- interpolate: Resample embeddings to fit using linear interpolation
- pool: Use adaptive average pooling to compress embeddings
- attention_pool: Use attention-weighted pooling (preserves important tokens)

Usage:
    from llm_dit.utils.long_prompt import compress_embeddings, LongPromptMode

    # Interpolation (smooth resampling)
    compressed = compress_embeddings(embeddings, max_len=1504, mode="interpolate")

    # Pooling (local averaging)
    compressed = compress_embeddings(embeddings, max_len=1504, mode="pool")

Note:
    These are EXPERIMENTAL features. Quality impact at different compression ratios
    is still being evaluated.
"""

import logging
from enum import Enum
from typing import Literal, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LongPromptMode(str, Enum):
    """Strategies for handling prompts exceeding MAX_TEXT_SEQ_LEN."""

    TRUNCATE = "truncate"  # Default: cut off at 1504 tokens
    INTERPOLATE = "interpolate"  # Resample embeddings using linear interpolation
    POOL = "pool"  # Adaptive average pooling
    ATTENTION_POOL = "attention_pool"  # Attention-weighted pooling


def compress_embeddings(
    embeddings: torch.Tensor,
    max_len: int = 1504,
    mode: Literal["truncate", "interpolate", "pool", "attention_pool"] = "truncate",
) -> torch.Tensor:
    """
    Compress embeddings to fit within max_len tokens.

    Args:
        embeddings: Input embeddings [seq_len, hidden_dim]
        max_len: Maximum sequence length (default: 1024)
        mode: Compression strategy:
            - "truncate": Simply cut off (default, safest)
            - "interpolate": Linear interpolation resampling
            - "pool": Adaptive average pooling
            - "attention_pool": Attention-weighted pooling

    Returns:
        Compressed embeddings [new_seq_len, hidden_dim] where new_seq_len <= max_len

    Example:
        >>> embeddings = torch.randn(1500, 2560)  # 1500 tokens
        >>> compressed = compress_embeddings(embeddings, max_len=1024, mode="interpolate")
        >>> compressed.shape
        torch.Size([1024, 2560])
    """
    seq_len, hidden_dim = embeddings.shape

    if seq_len <= max_len:
        return embeddings

    original_len = seq_len
    compression_ratio = seq_len / max_len

    if mode == "truncate":
        result = embeddings[:max_len]
        logger.warning(
            f"[LongPrompt] Truncated {original_len} -> {max_len} tokens "
            f"(compression ratio: {compression_ratio:.2f}x)"
        )
    elif mode == "interpolate":
        result = _interpolate_embeddings(embeddings, max_len)
        logger.info(
            f"[LongPrompt] Interpolated {original_len} -> {max_len} tokens "
            f"(compression ratio: {compression_ratio:.2f}x)"
        )
    elif mode == "pool":
        result = _pool_embeddings(embeddings, max_len)
        logger.info(
            f"[LongPrompt] Pooled {original_len} -> {max_len} tokens "
            f"(compression ratio: {compression_ratio:.2f}x)"
        )
    elif mode == "attention_pool":
        result = _attention_pool_embeddings(embeddings, max_len)
        logger.info(
            f"[LongPrompt] Attention-pooled {original_len} -> {max_len} tokens "
            f"(compression ratio: {compression_ratio:.2f}x)"
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use: truncate, interpolate, pool, attention_pool")

    return result


def _interpolate_embeddings(embeddings: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Resample embeddings using linear interpolation.

    This is similar in spirit to RoPE interpolation - we're scaling the sequence
    to fit within the target length while preserving relative positions.

    The interpolation happens along the sequence dimension, so each output position
    is a weighted combination of nearby input positions.
    """
    seq_len, hidden_dim = embeddings.shape

    # Reshape for F.interpolate: [batch, channels, length] = [1, hidden_dim, seq_len]
    x = embeddings.t().unsqueeze(0)  # [1, hidden_dim, seq_len]

    # Interpolate along sequence dimension
    x_interp = F.interpolate(
        x,
        size=target_len,
        mode="linear",
        align_corners=True,
    )

    # Reshape back: [target_len, hidden_dim]
    return x_interp.squeeze(0).t()


def _pool_embeddings(embeddings: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Compress embeddings using adaptive average pooling.

    This groups consecutive tokens and averages them. It's like a local
    summarization where each output token represents a region of the input.
    """
    seq_len, hidden_dim = embeddings.shape

    # Reshape for adaptive_avg_pool1d: [batch, channels, length]
    x = embeddings.t().unsqueeze(0)  # [1, hidden_dim, seq_len]

    # Pool to target length
    x_pooled = F.adaptive_avg_pool1d(x, target_len)

    # Reshape back: [target_len, hidden_dim]
    return x_pooled.squeeze(0).t()


def _attention_pool_embeddings(embeddings: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Compress embeddings using attention-weighted pooling with cosine similarity.

    This computes token importance based on how distinctive each token is
    compared to its neighbors. Tokens that are semantically unique (low cosine
    similarity to neighbors) get higher weights, preserving important concepts.

    The importance score for each token is computed as:
        importance = 1 - avg_cosine_similarity_to_neighbors

    This is better than L2 norm because:
    1. It captures semantic distinctiveness, not just magnitude
    2. Transition tokens (like commas, conjunctions) often have high norms but low importance
    3. Key concepts may have moderate norms but be very different from surrounding tokens
    """
    seq_len, hidden_dim = embeddings.shape

    # Normalize embeddings for cosine similarity
    embeddings_normalized = F.normalize(embeddings, p=2, dim=-1)  # [seq_len, hidden_dim]

    # Vectorized computation of neighbor similarities using convolution-like approach
    # Compute full similarity matrix for a window of +/- 2 tokens
    # Instead of loops, use matrix multiplication and masking

    # Compute all pairwise cosine similarities
    # For efficiency, only compute similarities within a band (window of 5)
    window_size = 2

    # Create shifted versions of normalized embeddings for neighbor comparison
    # Pad the sequence to handle boundaries
    padded = F.pad(embeddings_normalized, (0, 0, window_size, window_size), mode='constant', value=0)

    # Compute similarities with each offset position
    neighbor_sims = []
    for offset in range(-window_size, window_size + 1):
        if offset == 0:
            continue  # Skip self
        shifted = padded[window_size + offset:window_size + offset + seq_len]  # [seq_len, hidden_dim]
        # Cosine similarity (embeddings are already normalized)
        sim = (embeddings_normalized * shifted).sum(dim=-1)  # [seq_len]
        neighbor_sims.append(sim)

    # Stack and average similarities
    neighbor_sims = torch.stack(neighbor_sims, dim=0)  # [4, seq_len]

    # Handle boundary tokens (they have fewer neighbors due to padding with zeros)
    # Create a mask for valid neighbors at each position
    valid_counts = torch.ones(seq_len, device=embeddings.device)
    for i in range(window_size):
        valid_counts[i] = window_size + 1 + i  # Tokens at start have fewer left neighbors
        valid_counts[-(i + 1)] = window_size + 1 + i  # Tokens at end have fewer right neighbors

    # Average similarity (importance = 1 - avg_similarity)
    avg_similarity = neighbor_sims.sum(dim=0) / valid_counts.clamp(min=1)
    importance_scores = 1.0 - avg_similarity

    # Shift to ensure all scores are positive for softmax stability
    importance_scores = importance_scores - importance_scores.min() + 0.1

    # Vectorized pooling: assign each source token to a target region
    # and compute weighted averages per region
    pool_size = seq_len / target_len

    # Create region assignments for each token
    token_indices = torch.arange(seq_len, device=embeddings.device, dtype=torch.float32)
    region_assignments = (token_indices / pool_size).long().clamp(max=target_len - 1)

    # Compute softmax weights within each region using scatter operations
    # First, compute per-region max for numerical stability
    region_max = torch.zeros(target_len, device=embeddings.device, dtype=embeddings.dtype)
    region_max.scatter_reduce_(0, region_assignments, importance_scores, reduce='amax', include_self=False)

    # Compute exp(score - max) for softmax
    exp_scores = torch.exp(importance_scores - region_max[region_assignments])

    # Sum exp scores per region
    region_exp_sum = torch.zeros(target_len, device=embeddings.device, dtype=embeddings.dtype)
    region_exp_sum.scatter_add_(0, region_assignments, exp_scores)

    # Normalize to get softmax weights
    weights = exp_scores / region_exp_sum[region_assignments].clamp(min=1e-8)

    # Weighted sum of embeddings per region
    weighted_embeddings = embeddings * weights.unsqueeze(-1)  # [seq_len, hidden_dim]

    # Sum weighted embeddings per region
    output = torch.zeros(target_len, hidden_dim, device=embeddings.device, dtype=embeddings.dtype)
    output.scatter_add_(0, region_assignments.unsqueeze(-1).expand(-1, hidden_dim), weighted_embeddings)

    return output


def estimate_quality_loss(original_len: int, target_len: int, mode: str) -> str:
    """
    Estimate potential quality degradation from compression.

    Returns a human-readable assessment.
    """
    ratio = original_len / target_len

    if ratio <= 1.0:
        return "None (no compression needed)"

    if mode == "truncate":
        tokens_lost = original_len - target_len
        pct_lost = (tokens_lost / original_len) * 100
        if pct_lost > 50:
            return f"HIGH - {pct_lost:.0f}% of tokens discarded"
        elif pct_lost > 25:
            return f"Medium - {pct_lost:.0f}% of tokens discarded"
        else:
            return f"Low - {pct_lost:.0f}% of tokens discarded"

    elif mode == "interpolate":
        if ratio > 2.0:
            return f"HIGH - {ratio:.1f}x compression may blur details"
        elif ratio > 1.5:
            return f"Medium - {ratio:.1f}x compression, some detail loss expected"
        else:
            return f"Low - {ratio:.1f}x compression should preserve most details"

    elif mode == "pool":
        if ratio > 2.0:
            return f"HIGH - {ratio:.1f}x pooling loses token boundaries"
        elif ratio > 1.5:
            return f"Medium - {ratio:.1f}x pooling may blur adjacent concepts"
        else:
            return f"Low - {ratio:.1f}x pooling should work reasonably"

    elif mode == "attention_pool":
        if ratio > 2.0:
            return f"Medium-HIGH - {ratio:.1f}x but importance-weighted"
        elif ratio > 1.5:
            return f"Medium - {ratio:.1f}x with importance weighting"
        else:
            return f"Low - {ratio:.1f}x with importance weighting"

    return "Unknown"
