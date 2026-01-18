"""
Multi-backend attention for LTX-2 transformer.

Last Updated: 2026-01-18

Implements attention with support for multiple backends:
- PyTorch SDPA (default, always available)
- xFormers memory-efficient attention (optional)
- FlashAttention3 (optional, for Hopper GPUs)

Includes RoPE position embedding application and Q/K RMSNorm.

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
"""

from enum import Enum
from typing import Optional, Protocol, Tuple

import torch
import torch.nn as nn

from llm_dit.models.ltx2.rope import LTXRopeType, apply_rotary_emb

# Try to import optional attention backends
memory_efficient_attention = None
flash_attn_interface = None

try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    memory_efficient_attention = None

try:
    # FlashAttention3 and XFormersAttention cannot be used together
    if memory_efficient_attention is None:
        import flash_attn_interface
except ImportError:
    flash_attn_interface = None


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
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b, _, dim_head = q.shape
        dim_head //= heads

        # Reshape to [B, H, T, D]
        q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))

        if mask is not None:
            # Add batch dimension if missing
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # Add heads dimension if missing
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False
        )

        # Reshape back to [B, T, H*D]
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out


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


class FlashAttention3(AttentionCallable):
    """
    FlashAttention3 backend for Hopper GPUs.

    Uses flash_attn_interface for maximum performance on H100/H200 GPUs.
    Note: Mask support is not implemented for FA3.

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
        if flash_attn_interface is None:
            raise RuntimeError("FlashAttention3 was selected but `FlashAttention3` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            raise NotImplementedError("Mask is not supported for FlashAttention3")

        out = flash_attn_interface.flash_attn_func(q.to(v.dtype), k.to(v.dtype), v)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class AttentionFunction(Enum):
    """
    Enum for selecting attention backend.

    PYTORCH: Use PyTorch's SDPA (always available)
    XFORMERS: Use xFormers memory-efficient attention
    FLASH_ATTENTION_3: Use FlashAttention3 (Hopper only)
    DEFAULT: Auto-select best available backend
    """
    PYTORCH = "pytorch"
    XFORMERS = "xformers"
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
        elif self is AttentionFunction.FLASH_ATTENTION_3:
            return FlashAttention3()(q, k, v, heads, mask)
        else:
            # Default: XFormers if available, otherwise PyTorch
            if memory_efficient_attention is not None:
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
        # attention_function can be an AttentionFunction enum or a custom callable
        out = self.attention_function(q, k, v, self.heads, mask)

        return self.to_out(out)


def get_available_attention_backends() -> list[str]:
    """
    Get list of available attention backends.

    Returns:
        List of backend names that are available
    """
    backends = ["pytorch"]  # Always available

    if memory_efficient_attention is not None:
        backends.append("xformers")

    if flash_attn_interface is not None:
        backends.append("flash_attention_3")

    return backends


def get_default_attention_function() -> AttentionFunction:
    """
    Get the best available attention backend.

    Priority:
    1. FlashAttention3 (if available and on Hopper GPU)
    2. xFormers (if available)
    3. PyTorch SDPA (always available)

    Returns:
        AttentionFunction enum value
    """
    if flash_attn_interface is not None:
        return AttentionFunction.FLASH_ATTENTION_3
    elif memory_efficient_attention is not None:
        return AttentionFunction.XFORMERS
    else:
        return AttentionFunction.PYTORCH
