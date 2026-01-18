"""
Rotary Position Embeddings (RoPE) for LTX-2.

Last Updated: 2026-01-18

Implements rotary position embeddings for the LTX-2 video diffusion transformer.
Supports both interleaved and split RoPE variants, with 3D positional encoding
for video (temporal, height, width dimensions).

Ported from: coderef/LTX-2/packages/ltx-core/src/ltx_core/model/transformer/rope.py

Usage:
    from llm_dit.models.ltx2 import precompute_freqs_cis, apply_rotary_emb, LTXRopeType

    # Precompute position embeddings for video latents
    cos_freq, sin_freq = precompute_freqs_cis(
        indices_grid=positions,  # [B, 3, T] for (t, h, w)
        dim=4096,
        out_dtype=torch.bfloat16,
        max_pos=[20, 2048, 2048],  # Max positions for T, H, W
        rope_type=LTXRopeType.INTERLEAVED,
    )

    # Apply to query and key in attention
    q = apply_rotary_emb(q, (cos_freq, sin_freq), LTXRopeType.INTERLEAVED)
    k = apply_rotary_emb(k, (cos_freq, sin_freq), LTXRopeType.INTERLEAVED)
"""

import functools
import math
from enum import Enum
from typing import Callable, Tuple

import numpy as np
import torch
from einops import rearrange


class LTXRopeType(Enum):
    """RoPE implementation variant.

    INTERLEAVED: Pairs of (cos, sin) for adjacent dimensions
        - Used in LTX-2 default configuration
        - Better numerical stability for some cases

    SPLIT: First half cos, second half sin
        - Alternative layout, used in some other DiT models
    """
    INTERLEAVED = "interleaved"
    SPLIT = "split"


def apply_rotary_emb(
    input_tensor: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
) -> torch.Tensor:
    """
    Apply rotary position embeddings to input tensor.

    Args:
        input_tensor: Query or key tensor to apply RoPE to
        freqs_cis: Tuple of (cos_freqs, sin_freqs) from precompute_freqs_cis
        rope_type: Which RoPE variant to use

    Returns:
        Position-encoded tensor with same shape as input
    """
    if rope_type == LTXRopeType.INTERLEAVED:
        return apply_interleaved_rotary_emb(input_tensor, *freqs_cis)
    elif rope_type == LTXRopeType.SPLIT:
        return apply_split_rotary_emb(input_tensor, *freqs_cis)
    else:
        raise ValueError(f"Invalid rope type: {rope_type}")


def apply_interleaved_rotary_emb(
    input_tensor: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor
) -> torch.Tensor:
    """
    Apply interleaved rotary embeddings.

    For interleaved RoPE, adjacent dimension pairs are rotated together:
        dim[0], dim[1] -> rotate together
        dim[2], dim[3] -> rotate together
        etc.

    Args:
        input_tensor: [B, T, D] or [B, H, T, D] tensor
        cos_freqs: Cosine frequencies
        sin_freqs: Sine frequencies

    Returns:
        Rotated tensor with same shape
    """
    # Reshape to pairs: (..., D) -> (..., D/2, 2)
    t_dup = rearrange(input_tensor, "... (d r) -> ... d r", r=2)
    t1, t2 = t_dup.unbind(dim=-1)

    # Rotate: [t1, t2] -> [-t2, t1] (standard 2D rotation)
    t_dup = torch.stack((-t2, t1), dim=-1)
    input_tensor_rot = rearrange(t_dup, "... d r -> ... (d r)")

    # Apply rotation: x * cos + rot(x) * sin
    out = input_tensor * cos_freqs + input_tensor_rot * sin_freqs

    return out


def apply_split_rotary_emb(
    input_tensor: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor
) -> torch.Tensor:
    """
    Apply split rotary embeddings.

    For split RoPE, first half and second half are treated as a pair:
        dim[0:D/2] and dim[D/2:D] are rotated together

    Args:
        input_tensor: [B, T, D] or [B, H, T, D] tensor
        cos_freqs: Cosine frequencies
        sin_freqs: Sine frequencies

    Returns:
        Rotated tensor with same shape
    """
    needs_reshape = False
    if input_tensor.ndim != 4 and cos_freqs.ndim == 4:
        b, h, t, _ = cos_freqs.shape
        input_tensor = input_tensor.reshape(b, t, h, -1).swapaxes(1, 2)
        needs_reshape = True

    # Split into two halves
    split_input = rearrange(input_tensor, "... (d r) -> ... d r", d=2)
    first_half_input = split_input[..., :1, :]
    second_half_input = split_input[..., 1:, :]

    # Apply cos component
    output = split_input * cos_freqs.unsqueeze(-2)
    first_half_output = output[..., :1, :]
    second_half_output = output[..., 1:, :]

    # Apply sin component with cross-term rotation
    first_half_output.addcmul_(-sin_freqs.unsqueeze(-2), second_half_input)
    second_half_output.addcmul_(sin_freqs.unsqueeze(-2), first_half_input)

    output = rearrange(output, "... d r -> ... (d r)")
    if needs_reshape:
        output = output.swapaxes(1, 2).reshape(b, t, -1)

    return output


@functools.lru_cache(maxsize=5)
def generate_freq_grid_np(
    positional_embedding_theta: float,
    positional_embedding_max_pos_count: int,
    inner_dim: int
) -> torch.Tensor:
    """
    Generate frequency grid using numpy for higher precision.

    Uses numpy for the linspace computation which can be more precise
    for very large theta values. Cached to avoid recomputation.

    Args:
        positional_embedding_theta: Base theta value (typically 10000)
        positional_embedding_max_pos_count: Number of position dimensions (e.g., 3 for video)
        inner_dim: Hidden dimension of the model

    Returns:
        Frequency indices tensor
    """
    theta = positional_embedding_theta
    start = 1
    end = theta

    # Number of elements based on position count (T, H, W = 3 dims)
    n_elem = 2 * positional_embedding_max_pos_count

    # Log-spaced frequencies for each dimension
    pow_indices = np.power(
        theta,
        np.linspace(
            np.log(start) / np.log(theta),
            np.log(end) / np.log(theta),
            inner_dim // n_elem,
            dtype=np.float64,
        ),
    )
    return torch.tensor(pow_indices * math.pi / 2, dtype=torch.float32)


@functools.lru_cache(maxsize=5)
def generate_freq_grid_pytorch(
    positional_embedding_theta: float,
    positional_embedding_max_pos_count: int,
    inner_dim: int
) -> torch.Tensor:
    """
    Generate frequency grid using pure PyTorch.

    Slightly less precise than numpy version for large theta values,
    but faster and avoids numpy dependency in forward pass.

    Args:
        positional_embedding_theta: Base theta value (typically 10000)
        positional_embedding_max_pos_count: Number of position dimensions
        inner_dim: Hidden dimension

    Returns:
        Frequency indices tensor
    """
    theta = positional_embedding_theta
    start = 1
    end = theta
    n_elem = 2 * positional_embedding_max_pos_count

    indices = theta ** (
        torch.linspace(
            math.log(start, theta),
            math.log(end, theta),
            inner_dim // n_elem,
            dtype=torch.float32,
        )
    )
    indices = indices.to(dtype=torch.float32)
    indices = indices * math.pi / 2

    return indices


def get_fractional_positions(
    indices_grid: torch.Tensor,
    max_pos: list[int]
) -> torch.Tensor:
    """
    Convert absolute positions to fractional positions in [0, 1].

    Args:
        indices_grid: [B, n_dims, T] grid of position indices
        max_pos: Maximum position for each dimension

    Returns:
        Fractional positions in [0, 1] range
    """
    n_pos_dims = indices_grid.shape[1]
    assert n_pos_dims == len(max_pos), (
        f"Number of position dimensions ({n_pos_dims}) must match max_pos length ({len(max_pos)})"
    )

    fractional_positions = torch.stack(
        [indices_grid[:, i] / max_pos[i] for i in range(n_pos_dims)],
        dim=-1,
    )
    return fractional_positions


def generate_freqs(
    indices: torch.Tensor,
    indices_grid: torch.Tensor,
    max_pos: list[int],
    use_middle_indices_grid: bool
) -> torch.Tensor:
    """
    Generate frequency values from position grid.

    Args:
        indices: Base frequency indices from generate_freq_grid_*
        indices_grid: Position indices [B, n_dims, T] or [B, n_dims, T, 2] for start/end
        max_pos: Maximum positions per dimension
        use_middle_indices_grid: If True, use middle of start/end range

    Returns:
        Frequency values for position embedding
    """
    if use_middle_indices_grid:
        # For temporal interpolation - use middle of range
        assert len(indices_grid.shape) == 4
        assert indices_grid.shape[-1] == 2
        indices_grid_start, indices_grid_end = indices_grid[..., 0], indices_grid[..., 1]
        indices_grid = (indices_grid_start + indices_grid_end) / 2.0
    elif len(indices_grid.shape) == 4:
        indices_grid = indices_grid[..., 0]

    fractional_positions = get_fractional_positions(indices_grid, max_pos)
    indices = indices.to(device=fractional_positions.device)

    # Scale fractional positions to [-1, 1] range and multiply with frequencies
    freqs = (indices * (fractional_positions.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)
    return freqs


def split_freqs_cis(
    freqs: torch.Tensor,
    pad_size: int,
    num_attention_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cos/sin frequencies for split RoPE variant.

    Args:
        freqs: Raw frequency values
        pad_size: Amount of padding needed
        num_attention_heads: Number of attention heads

    Returns:
        Tuple of (cos_freq, sin_freq) tensors
    """
    cos_freq = freqs.cos()
    sin_freq = freqs.sin()

    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = torch.zeros_like(sin_freq[:, :, :pad_size])

        cos_freq = torch.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = torch.concatenate([sin_padding, sin_freq], axis=-1)

    # Reshape freqs to be compatible with multi-head attention
    b = cos_freq.shape[0]
    t = cos_freq.shape[1]

    cos_freq = cos_freq.reshape(b, t, num_attention_heads, -1)
    sin_freq = sin_freq.reshape(b, t, num_attention_heads, -1)

    cos_freq = torch.swapaxes(cos_freq, 1, 2)  # (B, H, T, D//2)
    sin_freq = torch.swapaxes(sin_freq, 1, 2)  # (B, H, T, D//2)
    return cos_freq, sin_freq


def interleaved_freqs_cis(
    freqs: torch.Tensor,
    pad_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cos/sin frequencies for interleaved RoPE variant.

    Args:
        freqs: Raw frequency values
        pad_size: Amount of padding needed

    Returns:
        Tuple of (cos_freq, sin_freq) tensors
    """
    # Interleave: [f0, f1, f2] -> [f0, f0, f1, f1, f2, f2]
    cos_freq = freqs.cos().repeat_interleave(2, dim=-1)
    sin_freq = freqs.sin().repeat_interleave(2, dim=-1)

    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = torch.zeros_like(cos_freq[:, :, :pad_size])
        cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
        sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)

    return cos_freq, sin_freq


def precompute_freqs_cis(
    indices_grid: torch.Tensor,
    dim: int,
    out_dtype: torch.dtype,
    theta: float = 10000.0,
    max_pos: list[int] | None = None,
    use_middle_indices_grid: bool = False,
    num_attention_heads: int = 32,
    rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
    freq_grid_generator: Callable[[float, int, int], torch.Tensor] = generate_freq_grid_pytorch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute rotary position embedding frequencies.

    This is the main entry point for computing RoPE embeddings. Takes a grid
    of position indices and returns cos/sin frequency tensors ready for
    application in attention.

    Args:
        indices_grid: Position indices [B, n_dims, T] where n_dims is typically 3 for video (t, h, w)
        dim: Inner dimension of the attention layer (num_heads * head_dim)
        out_dtype: Output dtype (typically bf16 or fp16)
        theta: RoPE base frequency (default 10000)
        max_pos: Maximum positions for each dimension, default [20, 2048, 2048] for video
        use_middle_indices_grid: Use middle of position range (for temporal interpolation)
        num_attention_heads: Number of attention heads (for split RoPE reshaping)
        rope_type: Which RoPE variant to use
        freq_grid_generator: Function to generate base frequencies

    Returns:
        Tuple of (cos_freq, sin_freq) tensors ready for apply_rotary_emb

    Example:
        # For video with 33 frames at 512x768 (T=33, H=16, W=24 latent)
        positions = torch.stack([
            temporal_indices,  # [B, T]
            height_indices,    # [B, T]
            width_indices,     # [B, T]
        ], dim=1)  # [B, 3, T]

        cos_freq, sin_freq = precompute_freqs_cis(
            positions, dim=4096, out_dtype=torch.bfloat16
        )
    """
    if max_pos is None:
        max_pos = [20, 2048, 2048]

    # Generate base frequencies
    indices = freq_grid_generator(theta, indices_grid.shape[1], dim)

    # Generate position-specific frequencies
    freqs = generate_freqs(indices, indices_grid, max_pos, use_middle_indices_grid)

    # Convert to cos/sin based on RoPE type
    if rope_type == LTXRopeType.SPLIT:
        expected_freqs = dim // 2
        current_freqs = freqs.shape[-1]
        pad_size = expected_freqs - current_freqs
        cos_freq, sin_freq = split_freqs_cis(freqs, pad_size, num_attention_heads)
    else:
        # Interleaved RoPE (default for LTX-2)
        # 2 for cos/sin, times number of position dimensions (3 for video)
        n_elem = 2 * indices_grid.shape[1]
        cos_freq, sin_freq = interleaved_freqs_cis(freqs, dim % n_elem)

    return cos_freq.to(out_dtype), sin_freq.to(out_dtype)
