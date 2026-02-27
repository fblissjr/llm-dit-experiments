"""
Multi-backend attention for LTX-2 transformer.

Last Updated: 2026-01-18

Implements attention with support for multiple backends:
- PyTorch SDPA (default, always available)
- xFormers memory-efficient attention (optional)
- FlashAttention2 (optional, for Ampere+ GPUs)
- FlashAttention3 (optional, for Hopper GPUs)

Includes RoPE position embedding application and Q/K RMSNorm.

torch.compile support:
    Set LLM_DIT_COMPILE=1 to enable torch.compile for attention kernels.
    Set LLM_DIT_COMPILE_MODE to control compile mode (default: reduce-overhead).
    Valid modes: default, reduce-overhead, max-autotune

Ported from: coderef/LTX-2/packages/ltx-core/src/ltx_core/model/transformer/attention.py

Usage:
    from llm_dit.models.ltx2 import Attention, AttentionFunction

    # Create attention with default (auto-detect best available)
    attn = Attention(
        query_dim=4096,
        heads=32,
        dim_head=128,
        context_dim=4096,  # None for self-attention
    )

    # Forward pass with RoPE
    out = attn(x, context=encoder_hidden_states, pe=position_embeddings)

    # Enable torch.compile via environment
    # export LLM_DIT_COMPILE=1
    # export LLM_DIT_COMPILE_MODE=reduce-overhead
"""

import os
from enum import Enum
from typing import Optional, Protocol, Tuple

import torch
import torch.nn as nn

# torch.compile configuration
LLM_DIT_COMPILE = os.environ.get("LLM_DIT_COMPILE", "0") == "1"
LLM_DIT_COMPILE_MODE = os.environ.get("LLM_DIT_COMPILE_MODE", "reduce-overhead")

from llm_dit.models.ltx2.rope import LTXRopeType, apply_rotary_emb

# Try to import optional attention backends
memory_efficient_attention = None
HAS_FLASH_ATTN_3 = False
HAS_FLASH_ATTN_2 = False

try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    memory_efficient_attention = None

# FlashAttention 3 (Hopper GPUs - H100, etc.)
try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    HAS_FLASH_ATTN_3 = True
except ImportError:
    flash_attn_3_func = None  # type: ignore[misc, assignment]

# FlashAttention 2 (Ampere+ - RTX 3090, 4090, A100, etc.)
try:
    from flash_attn import flash_attn_func as flash_attn_2_func
    HAS_FLASH_ATTN_2 = True
except ImportError:
    flash_attn_2_func = None  # type: ignore[misc, assignment]


# =============================================================================
# Compiled Attention Kernels
# =============================================================================
# Pure functions that can be compiled with torch.compile for better performance.
# When LLM_DIT_COMPILE=1, these are compiled at module load time.


def _sdpa_attention_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Pure SDPA attention kernel suitable for torch.compile.

    Args:
        q: Query tensor [B, T, H*D]
        k: Key tensor [B, S, H*D]
        v: Value tensor [B, S, H*D]
        heads: Number of attention heads
        mask: Optional attention mask

    Returns:
        Output tensor [B, T, H*D]
    """
    b, _, dim_head = q.shape
    dim_head //= heads

    # Reshape to [B, H, T, D]
    q = q.view(b, -1, heads, dim_head).transpose(1, 2)
    k = k.view(b, -1, heads, dim_head).transpose(1, 2)
    v = v.view(b, -1, heads, dim_head).transpose(1, 2)

    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False
    )

    # Reshape back to [B, T, H*D]
    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


def _fa2_attention_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Pure FlashAttention2 kernel suitable for torch.compile.

    Note: mask is not supported but kept in signature for consistency.
    """
    if mask is not None:
        raise NotImplementedError("Mask is not supported for FlashAttention2")

    b, _, dim_head = q.shape
    dim_head //= heads

    # FA2 expects (B, S, H, D)
    q = q.view(b, -1, heads, dim_head)
    k = k.view(b, -1, heads, dim_head)
    v = v.view(b, -1, heads, dim_head)

    # FA2 is only called via classes that check availability first
    out = flash_attn_2_func(q.to(v.dtype), k.to(v.dtype), v)  # type: ignore[misc]
    return out.reshape(b, -1, heads * dim_head)


def _fa3_attention_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Pure FlashAttention3 kernel suitable for torch.compile.

    Note: mask is not supported but kept in signature for consistency.
    """
    if mask is not None:
        raise NotImplementedError("Mask is not supported for FlashAttention3")

    b, _, dim_head = q.shape
    dim_head //= heads

    q = q.view(b, -1, heads, dim_head)
    k = k.view(b, -1, heads, dim_head)
    v = v.view(b, -1, heads, dim_head)

    # FA3 is only called via classes that check availability first
    out = flash_attn_3_func(q.to(v.dtype), k.to(v.dtype), v)  # type: ignore[misc]
    return out.reshape(b, -1, heads * dim_head)


# Compile kernels if enabled
# Note: We use dynamic=True to handle variable sequence lengths in video generation
if LLM_DIT_COMPILE:
    _compile_options = {"mode": LLM_DIT_COMPILE_MODE, "dynamic": True}

    sdpa_attention_kernel = torch.compile(_sdpa_attention_kernel, **_compile_options)

    # Only compile FA kernels if available
    if HAS_FLASH_ATTN_2:
        fa2_attention_kernel = torch.compile(_fa2_attention_kernel, **_compile_options)
    else:
        fa2_attention_kernel = _fa2_attention_kernel

    if HAS_FLASH_ATTN_3:
        fa3_attention_kernel = torch.compile(_fa3_attention_kernel, **_compile_options)
    else:
        fa3_attention_kernel = _fa3_attention_kernel
else:
    # No compilation - use raw functions
    sdpa_attention_kernel = _sdpa_attention_kernel
    fa2_attention_kernel = _fa2_attention_kernel
    fa3_attention_kernel = _fa3_attention_kernel


class AttentionCallable(Protocol):
    """Protocol for attention function implementations."""

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        ...


class PytorchAttention(AttentionCallable):
    """
    PyTorch SDPA (Scaled Dot-Product Attention) backend.

    Uses torch.nn.functional.scaled_dot_product_attention which automatically
    selects the best available kernel (FlashAttention, Memory-Efficient, or Math).

    This is always available and serves as the fallback.
    When LLM_DIT_COMPILE=1, uses the compiled kernel for better performance.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return sdpa_attention_kernel(q, k, v, heads, mask)


class XFormersAttention(AttentionCallable):
    """
    xFormers memory-efficient attention backend.

    Uses xformers.ops.memory_efficient_attention for better memory
    efficiency on older GPUs or when SDPA doesn't use FlashAttention.

    Raises:
        RuntimeError: If xFormers is not installed
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if memory_efficient_attention is None:
            raise RuntimeError("XFormersAttention was selected but `xformers` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        # xformers expects [B, M, H, K] (batch, sequence, heads, head_dim)
        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            # Add singleton batch dimension
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # Add singleton heads dimension
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            # Pad to multiple of 8 for efficiency
            pad = 8 - mask.shape[-1] % 8
            mask_out = torch.empty(
                [mask.shape[0], mask.shape[1], q.shape[1], mask.shape[-1] + pad],
                dtype=q.dtype,
                device=q.device
            )
            mask_out[..., : mask.shape[-1]] = mask
            mask = mask_out[..., : mask.shape[-1]]
            mask = mask.expand(b, heads, -1, -1)

        out = memory_efficient_attention(
            q.to(v.dtype),
            k.to(v.dtype),
            v,
            attn_bias=mask,
            p=0.0
        )

        out = out.reshape(b, -1, heads * dim_head)
        return out


class FlashAttention2(AttentionCallable):
    """
    FlashAttention2 backend for Ampere+ GPUs.

    Uses flash_attn for maximum performance on RTX 3090/4090, A100, etc.
    Note: Mask support is not implemented for FA2.
    When LLM_DIT_COMPILE=1, uses the compiled kernel for better performance.

    Raises:
        RuntimeError: If FlashAttention2 is not installed
        NotImplementedError: If mask is provided
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not HAS_FLASH_ATTN_2:
            raise RuntimeError("FlashAttention2 was selected but `flash-attn` is not installed.")
        return fa2_attention_kernel(q, k, v, heads, mask)


class FlashAttention3(AttentionCallable):
    """
    FlashAttention3 backend for Hopper GPUs.

    Uses flash_attn_interface for maximum performance on H100/H200 GPUs.
    Note: Mask support is not implemented for FA3.
    When LLM_DIT_COMPILE=1, uses the compiled kernel for better performance.

    Raises:
        RuntimeError: If FlashAttention3 is not installed
        NotImplementedError: If mask is provided
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not HAS_FLASH_ATTN_3:
            raise RuntimeError("FlashAttention3 was selected but `flash_attn_interface` is not installed.")
        return fa3_attention_kernel(q, k, v, heads, mask)


class AttentionFunction(Enum):
    """
    Enum for selecting attention backend.

    PYTORCH: Use PyTorch's SDPA (always available)
    XFORMERS: Use xFormers memory-efficient attention
    FLASH_ATTENTION_2: Use FlashAttention2 (Ampere+ GPUs)
    FLASH_ATTENTION_3: Use FlashAttention3 (Hopper only)
    DEFAULT: Auto-select best available backend

    Priority order for DEFAULT:
    1. FlashAttention3 (if available)
    2. FlashAttention2 (if available)
    3. xFormers (if available)
    4. PyTorch SDPA (always available)
    """
    PYTORCH = "pytorch"
    XFORMERS = "xformers"
    FLASH_ATTENTION_2 = "flash_attention_2"
    FLASH_ATTENTION_3 = "flash_attention_3"
    DEFAULT = "default"

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self is AttentionFunction.PYTORCH:
            return PytorchAttention()(q, k, v, heads, mask)
        elif self is AttentionFunction.XFORMERS:
            return XFormersAttention()(q, k, v, heads, mask)
        elif self is AttentionFunction.FLASH_ATTENTION_2:
            return FlashAttention2()(q, k, v, heads, mask)
        elif self is AttentionFunction.FLASH_ATTENTION_3:
            return FlashAttention3()(q, k, v, heads, mask)
        else:
            # Default: best available in priority order
            # FA3 > FA2 > xFormers > PyTorch SDPA
            if HAS_FLASH_ATTN_3:
                return FlashAttention3()(q, k, v, heads, mask)
            elif HAS_FLASH_ATTN_2:
                return FlashAttention2()(q, k, v, heads, mask)
            elif memory_efficient_attention is not None:
                return XFormersAttention()(q, k, v, heads, mask)
            else:
                return PytorchAttention()(q, k, v, heads, mask)


class Attention(nn.Module):
    """
    Multi-head attention with RoPE and Q/K normalization.

    This is the core attention module used in LTX-2 transformer blocks.
    It supports:
    - Self-attention (context_dim=None)
    - Cross-attention (context_dim != None)
    - RoPE position embeddings
    - Q/K RMSNorm for stable training

    Args:
        query_dim: Dimension of query input
        context_dim: Dimension of context (key/value) input. None for self-attention.
        heads: Number of attention heads
        dim_head: Dimension per head
        norm_eps: Epsilon for RMSNorm
        rope_type: RoPE variant (INTERLEAVED or SPLIT)
        attention_function: Which backend to use (or callable)

    Example:
        # Self-attention (attn1 in transformer block)
        self_attn = Attention(query_dim=4096, heads=32, dim_head=128)
        out = self_attn(x, pe=position_embeddings)

        # Cross-attention (attn2 in transformer block)
        cross_attn = Attention(query_dim=4096, context_dim=4096, heads=32, dim_head=128)
        out = cross_attn(x, context=encoder_hidden_states)
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        attention_function: AttentionCallable | AttentionFunction = AttentionFunction.DEFAULT,
    ) -> None:
        super().__init__()
        self.rope_type = rope_type
        self.attention_function = attention_function

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head

        # Q/K RMSNorm for stable training
        self.q_norm = nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = nn.RMSNorm(inner_dim, eps=norm_eps)

        # Linear projections
        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=True)

        # Output projection with Identity placeholder (for dropout if needed)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, bias=True),
            nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        pe: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        k_pe: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass of attention.

        Args:
            x: Query input [B, T, query_dim]
            context: Key/Value input [B, S, context_dim]. If None, uses x (self-attention)
            mask: Optional attention mask [B, T, S] or [T, S]
            pe: Position embeddings (cos_freq, sin_freq) for query
            k_pe: Optional separate position embeddings for keys (for cross-modal attention)

        Returns:
            Output tensor [B, T, query_dim]
        """
        # Compute Q, K, V
        q = self.to_q(x)
        context = x if context is None else context
        k = self.to_k(context)
        v = self.to_v(context)

        # Apply Q/K RMSNorm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE position embeddings
        if pe is not None:
            q = apply_rotary_emb(q, pe, self.rope_type)
            k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)

        # Compute attention
        out = self.attention_function(q, k, v, self.heads, mask)

        return self.to_out(out)


def get_available_attention_backends() -> list[str]:
    """
    Get list of available attention backends.

    Returns:
        List of backend names that are available, in priority order
    """
    backends = []

    if HAS_FLASH_ATTN_3:
        backends.append("flash_attention_3")

    if HAS_FLASH_ATTN_2:
        backends.append("flash_attention_2")

    if memory_efficient_attention is not None:
        backends.append("xformers")

    backends.append("pytorch")  # Always available

    return backends


def get_default_attention_function() -> AttentionFunction:
    """
    Get the best available attention backend.

    Priority:
    1. FlashAttention3 (if available - Hopper GPUs)
    2. FlashAttention2 (if available - Ampere+ GPUs)
    3. xFormers (if available)
    4. PyTorch SDPA (always available)

    Returns:
        AttentionFunction enum value
    """
    if HAS_FLASH_ATTN_3:
        return AttentionFunction.FLASH_ATTENTION_3
    elif HAS_FLASH_ATTN_2:
        return AttentionFunction.FLASH_ATTENTION_2
    elif memory_efficient_attention is not None:
        return AttentionFunction.XFORMERS
    else:
        return AttentionFunction.PYTORCH


def is_compile_enabled() -> bool:
    """
    Check if torch.compile is enabled for attention kernels.

    Returns:
        True if LLM_DIT_COMPILE=1 is set
    """
    return LLM_DIT_COMPILE


def get_compile_mode() -> str:
    """
    Get the torch.compile mode being used.

    Returns:
        The compile mode string (e.g., "reduce-overhead", "max-autotune")
    """
    return LLM_DIT_COMPILE_MODE
