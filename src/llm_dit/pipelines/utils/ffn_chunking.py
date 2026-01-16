"""
FFN Chunking for Memory-Efficient LTX-2 Inference.

Last Updated: 2026-01-16

Ported from ComfyUI-KJNodes/nodes/ltxv_nodes.py (lines 417-472).

FFN (Feed-Forward Network) chunking reduces peak VRAM by processing the
feedforward layers in smaller chunks rather than all at once. This is
particularly effective for video transformers where sequence length is
very long (frames × height × width patches).

The memory usage of FFN layers scales quadratically with sequence length
in the intermediate activations. By chunking along the sequence dimension
and processing sequentially, we reduce peak memory at the cost of some
speed (more kernel launches, less parallelism).

Key Parameters:
- num_chunks (default 4): Number of chunks to split sequence into
- dim_threshold (default 4096): Only chunk if sequence length > threshold

Memory savings:
- 4 chunks: ~4x reduction in FFN activation memory
- 8 chunks: ~8x reduction in FFN activation memory

Speed impact:
- 4 chunks: ~10-20% slower (more kernel overhead)
- 8 chunks: ~20-40% slower

Reference:
    ComfyUI-KJNodes by Kijai
    https://github.com/kijai/ComfyUI-KJNodes

Example:
    # Patch FFN layers to use chunking
    from llm_dit.pipelines.utils.ffn_chunking import patch_ffn_chunking

    pipe = LTX2Pipeline.from_pretrained(...)
    patch_ffn_chunking(pipe.transformer, num_chunks=4)
"""

from dataclasses import dataclass
from typing import Optional, Callable, Any
import functools

import torch
import torch.nn as nn


@dataclass
class FFNChunkingConfig:
    """Configuration for FFN chunking."""

    # Whether FFN chunking is enabled
    enabled: bool = True

    # Number of chunks to split sequence into
    # More chunks = less memory but slower
    num_chunks: int = 4

    # Only apply chunking if sequence dim > threshold
    # Avoids overhead for short sequences
    dim_threshold: int = 4096

    # Which dimension to chunk on (typically 1 for [B, S, D])
    chunk_dim: int = 1


def chunked_ffn_forward(
    ffn_module: nn.Module,
    x: torch.Tensor,
    num_chunks: int = 4,
    dim_threshold: int = 4096,
    chunk_dim: int = 1,
) -> torch.Tensor:
    """
    Process FFN in chunks to reduce peak VRAM.

    Instead of processing the entire sequence at once, splits along the
    sequence dimension and processes each chunk sequentially. This reduces
    peak memory for the intermediate FFN activations.

    Args:
        ffn_module: The FFN module to apply
        x: Input tensor [B, S, D] or similar
        num_chunks: Number of chunks to split into
        dim_threshold: Only chunk if dim > this threshold
        chunk_dim: Dimension to chunk along (typically 1 for sequence)

    Returns:
        Output tensor with same shape as input
    """
    # Check if chunking is beneficial
    seq_len = x.shape[chunk_dim]
    if seq_len <= dim_threshold:
        # Short sequence - just run normally
        return ffn_module(x)

    # Ensure we don't have more chunks than elements
    actual_chunks = min(num_chunks, seq_len)

    # Split into chunks along sequence dimension
    chunks = torch.chunk(x, actual_chunks, dim=chunk_dim)

    # Process each chunk
    output_chunks = []
    for chunk in chunks:
        output_chunk = ffn_module(chunk)
        output_chunks.append(output_chunk)

    # Concatenate results
    return torch.cat(output_chunks, dim=chunk_dim)


def create_chunked_ffn_wrapper(
    original_forward: Callable,
    num_chunks: int = 4,
    dim_threshold: int = 4096,
    chunk_dim: int = 1,
) -> Callable:
    """
    Create a wrapped forward function that chunks FFN computation.

    This creates a wrapper around an existing FFN forward method that
    transparently applies chunking.

    Args:
        original_forward: Original forward method
        num_chunks: Number of chunks
        dim_threshold: Threshold for chunking
        chunk_dim: Dimension to chunk

    Returns:
        Wrapped forward function
    """

    @functools.wraps(original_forward)
    def chunked_forward(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        seq_len = x.shape[chunk_dim]

        if seq_len <= dim_threshold:
            return original_forward(x, *args, **kwargs)

        actual_chunks = min(num_chunks, seq_len)
        chunks = torch.chunk(x, actual_chunks, dim=chunk_dim)

        output_chunks = []
        for chunk in chunks:
            output_chunk = original_forward(chunk, *args, **kwargs)
            output_chunks.append(output_chunk)

        return torch.cat(output_chunks, dim=chunk_dim)

    return chunked_forward


class ChunkedFFN(nn.Module):
    """
    Wrapper module that applies chunking to any FFN module.

    This wraps an existing FFN module and applies chunking during forward pass.

    Example:
        original_ffn = transformer.blocks[0].ffn
        chunked = ChunkedFFN(original_ffn, num_chunks=4)
        transformer.blocks[0].ffn = chunked
    """

    def __init__(
        self,
        ffn: nn.Module,
        num_chunks: int = 4,
        dim_threshold: int = 4096,
        chunk_dim: int = 1,
    ):
        """
        Initialize chunked FFN wrapper.

        Args:
            ffn: Original FFN module
            num_chunks: Number of chunks
            dim_threshold: Threshold for enabling chunking
            chunk_dim: Dimension to chunk along
        """
        super().__init__()
        self.ffn = ffn
        self.num_chunks = num_chunks
        self.dim_threshold = dim_threshold
        self.chunk_dim = chunk_dim

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward with chunking."""
        return chunked_ffn_forward(
            self.ffn,
            x,
            num_chunks=self.num_chunks,
            dim_threshold=self.dim_threshold,
            chunk_dim=self.chunk_dim,
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped FFN."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.ffn, name)


def patch_ffn_chunking(
    model: nn.Module,
    num_chunks: int = 4,
    dim_threshold: int = 4096,
    chunk_dim: int = 1,
    ffn_names: Optional[list[str]] = None,
) -> int:
    """
    Patch all FFN layers in a model to use chunking.

    Finds FFN modules by name pattern and wraps them with ChunkedFFN.
    This is the main entry point for applying FFN chunking to a model.

    Args:
        model: Model to patch (e.g., transformer)
        num_chunks: Number of chunks for each FFN
        dim_threshold: Threshold for enabling chunking
        chunk_dim: Dimension to chunk along
        ffn_names: List of FFN attribute names to look for.
                   Default: ["ffn", "mlp", "feed_forward", "ff"]

    Returns:
        Number of FFN modules patched

    Example:
        from diffusers import LTX2Pipeline
        pipe = LTX2Pipeline.from_pretrained(...)

        # Patch transformer FFNs
        num_patched = patch_ffn_chunking(pipe.transformer, num_chunks=4)
        print(f"Patched {num_patched} FFN modules")
    """
    if ffn_names is None:
        ffn_names = ["ffn", "mlp", "feed_forward", "ff", "MLP"]

    patched_count = 0

    # Iterate through all named modules
    for name, module in list(model.named_modules()):
        # Check if this module has an FFN submodule we should patch
        for ffn_name in ffn_names:
            if hasattr(module, ffn_name):
                original_ffn = getattr(module, ffn_name)

                # Skip if already wrapped
                if isinstance(original_ffn, ChunkedFFN):
                    continue

                # Skip if not a Module
                if not isinstance(original_ffn, nn.Module):
                    continue

                # Wrap with chunking
                chunked = ChunkedFFN(
                    original_ffn,
                    num_chunks=num_chunks,
                    dim_threshold=dim_threshold,
                    chunk_dim=chunk_dim,
                )

                # Replace
                setattr(module, ffn_name, chunked)
                patched_count += 1

    return patched_count


def unpatch_ffn_chunking(model: nn.Module) -> int:
    """
    Remove FFN chunking wrappers from a model.

    Restores original FFN modules by unwrapping ChunkedFFN instances.

    Args:
        model: Model to unpatch

    Returns:
        Number of FFN modules unpatched
    """
    unpatched_count = 0

    for name, module in list(model.named_modules()):
        for attr_name in dir(module):
            try:
                attr = getattr(module, attr_name)
                if isinstance(attr, ChunkedFFN):
                    # Restore original
                    setattr(module, attr_name, attr.ffn)
                    unpatched_count += 1
            except Exception:
                continue

    return unpatched_count


def estimate_memory_savings(
    sequence_length: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_chunks: int = 4,
    dtype_bytes: int = 2,  # bfloat16
) -> dict[str, float]:
    """
    Estimate memory savings from FFN chunking.

    FFN layers typically expand to 4x hidden dim then back down.
    Intermediate activations are the main memory consumer.

    Args:
        sequence_length: Sequence length (e.g., frames * height/8 * width/8)
        hidden_dim: Model hidden dimension
        intermediate_dim: FFN intermediate dimension (typically 4 * hidden_dim)
        num_chunks: Number of chunks
        dtype_bytes: Bytes per element (2 for bf16/fp16, 4 for fp32)

    Returns:
        Dict with memory estimates in GB
    """
    # Without chunking: full sequence in intermediate dim
    no_chunk_memory = sequence_length * intermediate_dim * dtype_bytes / 1e9

    # With chunking: 1/num_chunks of sequence at a time
    with_chunk_memory = (sequence_length / num_chunks) * intermediate_dim * dtype_bytes / 1e9

    return {
        "without_chunking_gb": no_chunk_memory,
        "with_chunking_gb": with_chunk_memory,
        "savings_gb": no_chunk_memory - with_chunk_memory,
        "savings_ratio": 1 - (with_chunk_memory / no_chunk_memory),
    }
