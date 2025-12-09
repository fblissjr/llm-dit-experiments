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

    # Compute pairwise cosine similarity with neighbors
    # We use a window of +/- 2 tokens for context
    importance_scores = torch.zeros(seq_len, device=embeddings.device, dtype=embeddings.dtype)

    for i in range(seq_len):
        # Get neighbor indices (within window, excluding self)
        start = max(0, i - 2)
        end = min(seq_len, i + 3)
        neighbor_indices = [j for j in range(start, end) if j != i]

        if not neighbor_indices:
            # No neighbors (shouldn't happen in practice)
            importance_scores[i] = 1.0
            continue

        # Compute cosine similarity to neighbors
        token_vec = embeddings_normalized[i]  # [hidden_dim]
        neighbor_vecs = embeddings_normalized[neighbor_indices]  # [num_neighbors, hidden_dim]
        similarities = torch.mv(neighbor_vecs, token_vec)  # [num_neighbors]

        # Importance = 1 - average similarity (more unique = more important)
        avg_similarity = similarities.mean()
        importance_scores[i] = 1.0 - avg_similarity

    # Shift to ensure all scores are positive for softmax stability
    importance_scores = importance_scores - importance_scores.min() + 0.1

    # Determine pool regions
    pool_size = seq_len / target_len
    output = []

    for i in range(target_len):
        # Compute region boundaries (can be fractional)
        start_idx = int(i * pool_size)
        end_idx = min(int((i + 1) * pool_size) + 1, seq_len)

        if start_idx >= seq_len:
            # Edge case: use last token
            output.append(embeddings[-1])
            continue

        # Get tokens and importance scores in this region
        region_embeddings = embeddings[start_idx:end_idx]  # [region_len, hidden_dim]
        region_importance = importance_scores[start_idx:end_idx]  # [region_len]

        # Softmax weights based on importance
        weights = F.softmax(region_importance, dim=0)  # [region_len]

        # Weighted average
        pooled = (region_embeddings * weights.unsqueeze(-1)).sum(dim=0)  # [hidden_dim]
        output.append(pooled)

    return torch.stack(output, dim=0)  # [target_len, hidden_dim]


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
