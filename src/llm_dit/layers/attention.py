"""
Attention layers - canonical implementations for all models.

Last Updated: 2026-02-01

This module consolidates attention implementations scattered across the codebase
into a single, configurable canonical implementation.

Previous implementations:
- embeddings_connector.py: Attention (inner_dim QK norm, bias=True)
- z_image/attention.py: Attention (per-head QK norm, bias=False)
- context_refiner.py: ContextRefinerAttention (per-head QK norm, bias=False)

Key Architectural Differences Preserved:

1. QK Normalization Strategy (CRITICAL):
   - INNER_DIM: norm(q) then reshape to heads - normalizes across all heads together
   - PER_HEAD: reshape to heads then norm(q) - normalizes each head independently
   These produce mathematically different outputs!

2. Projection Biases:
   - Connector: bias=True on all projections
   - Z-Image: bias=False on all projections

3. Output projection structure:
   - Connector: nn.Sequential with Identity placeholder
   - Z-Image: nn.ModuleList with single Linear

RoPE Note:
    RoPE implementations vary significantly (real vs complex, interleaved vs split).
    This module does NOT include RoPE - it should be applied externally and passed
    to forward() as pre-rotated Q/K tensors, OR RoPE can be applied within the
    caller after calling the Q/K projections separately.

    For models that need RoPE integrated into attention, use the model-specific
    attention classes or apply RoPE before calling this module's forward().
"""

from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalization import RMSNorm


class QKNormType(Enum):
    """QK normalization strategy.

    The normalization dimension significantly affects the output:
    - NONE: No QK normalization
    - INNER_DIM: Normalize on full inner_dim before reshaping to heads
    - PER_HEAD: Normalize per-head after reshaping

    Example of the difference:
        With dim=3840, num_heads=30, head_dim=128:
        - INNER_DIM: RMSNorm(3840) applied to [B, seq, 3840]
        - PER_HEAD: RMSNorm(128) applied to [B, seq, 30, 128]
    """

    NONE = "none"
    INNER_DIM = "inner_dim"
    PER_HEAD = "per_head"


class Attention(nn.Module):
    """
    Configurable multi-head attention with QK normalization.

    This is the canonical attention implementation supporting all model variants.
    Each model's original attention behavior can be replicated through configuration.

    Args:
        dim: Input dimension (query dimension).
        num_heads: Number of attention heads.
        head_dim: Dimension per head. inner_dim = num_heads * head_dim.
        context_dim: Context dimension for cross-attention. If None, uses dim
            (self-attention).
        qk_norm: QK normalization strategy (none, inner_dim, per_head).
        qk_norm_eps: Epsilon for QK normalization RMSNorm.
        bias: Whether to use bias in Q, K, V projections.
        bias_out: Whether to use bias in output projection. If None, uses `bias`.
        dropout: Dropout probability after attention (default 0.0).

    Shapes:
        - Input x: (batch, seq_len, dim)
        - Context (optional): (batch, ctx_len, context_dim)
        - Output: (batch, seq_len, dim)

    Architecture Variants:

    Connector (LTX-2 text encoder):
        qk_norm=QKNormType.INNER_DIM, qk_norm_eps=1e-6, bias=True
        - Normalizes Q/K across all heads together before reshape
        - Uses nn.Sequential for output with Identity placeholder

    Z-Image DiT:
        qk_norm=QKNormType.PER_HEAD, qk_norm_eps=1e-5, bias=False
        - Normalizes Q/K per-head after reshape
        - Uses nn.ModuleList for output

    Example:
        >>> # Connector-style attention
        >>> attn = Attention(3840, num_heads=30, head_dim=128,
        ...                  qk_norm=QKNormType.INNER_DIM, bias=True)
        >>>
        >>> # Z-Image-style attention
        >>> attn = Attention(3072, num_heads=24, head_dim=128,
        ...                  qk_norm=QKNormType.PER_HEAD, qk_norm_eps=1e-5, bias=False)

    Weight Loading Compatibility:
        Parameter names are designed for checkpoint compatibility:
        - to_q, to_k, to_v: QKV projections
        - norm_q, norm_k: QK normalization (if enabled)
        - to_out: Output projection (list for Z-Image compat, sequential for Connector)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        context_dim: Optional[int] = None,
        qk_norm: QKNormType = QKNormType.NONE,
        qk_norm_eps: float = 1e-6,
        bias: bool = True,
        bias_out: Optional[bool] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.context_dim = context_dim if context_dim is not None else dim
        self.qk_norm = qk_norm
        self.dropout_prob = dropout

        if bias_out is None:
            bias_out = bias

        # QKV projections
        self.to_q = nn.Linear(dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(self.context_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(self.context_dim, self.inner_dim, bias=bias)

        # QK normalization
        self._has_qk_norm = qk_norm != QKNormType.NONE
        if qk_norm == QKNormType.INNER_DIM:
            # Normalize across all heads together (Connector style)
            self.norm_q = RMSNorm(self.inner_dim, eps=qk_norm_eps)
            self.norm_k = RMSNorm(self.inner_dim, eps=qk_norm_eps)
        elif qk_norm == QKNormType.PER_HEAD:
            # Normalize per-head (Z-Image style)
            self.norm_q = RMSNorm(head_dim, eps=qk_norm_eps)
            self.norm_k = RMSNorm(head_dim, eps=qk_norm_eps)
        else:
            # Register as None to keep state_dict consistent
            self.register_module("norm_q", None)
            self.register_module("norm_k", None)

        # Output projection
        # Use nn.ModuleList for Z-Image compatibility (they use to_out[0])
        # This also works for Connector when loaded with Sequential wrapper
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, dim, bias=bias_out)])

        # Dropout
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for attention.

        Args:
            x: Query input tensor (batch, seq_len, dim).
            context: Key/value context tensor for cross-attention.
                If None, uses x (self-attention).
            mask: Attention mask. Supports multiple formats:
                - (batch, seq_len): Boolean mask where True = attend
                - (batch, 1, 1, seq_len): Additive mask where 0 = attend, -inf = mask
                - (batch, num_heads, seq_len, ctx_len): Full attention mask

        Returns:
            Output tensor (batch, seq_len, dim).

        Note:
            RoPE should be applied externally before calling this method,
            or use the project_qk() method to get Q/K for RoPE application.
        """
        batch_size, seq_len, _ = x.shape

        # Handle context for cross-attention
        if context is None:
            context = x
        ctx_len = context.shape[1]

        # Project to Q, K, V
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # Apply QK normalization based on type
        if self.qk_norm == QKNormType.INNER_DIM:
            # Inner-dim normalization: normalize before reshape
            q = self.norm_q(q)
            k = self.norm_k(k)
            # Then reshape to heads
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.view(batch_size, ctx_len, self.num_heads, self.head_dim)
            v = v.view(batch_size, ctx_len, self.num_heads, self.head_dim)

        elif self.qk_norm == QKNormType.PER_HEAD:
            # Per-head normalization: reshape first, then normalize
            q = q.unflatten(-1, (self.num_heads, self.head_dim))
            k = k.unflatten(-1, (self.num_heads, self.head_dim))
            v = v.unflatten(-1, (self.num_heads, self.head_dim))
            # Normalize per-head
            q = self.norm_q(q)
            k = self.norm_k(k)

        else:
            # No QK normalization
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.view(batch_size, ctx_len, self.num_heads, self.head_dim)
            v = v.view(batch_size, ctx_len, self.num_heads, self.head_dim)

        # Transpose for attention: (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Process attention mask
        attn_mask = self._process_mask(mask, q.dtype, batch_size, seq_len, ctx_len)

        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )

        # Reshape back: (batch, seq_len, inner_dim)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.inner_dim)

        # Apply dropout if configured
        if self.dropout is not None:
            out = self.dropout(out)

        # Output projection
        out = self.to_out[0](out)

        return out

    def _process_mask(
        self,
        mask: Optional[torch.Tensor],
        dtype: torch.dtype,
        batch_size: int,
        seq_len: int,
        ctx_len: int,
    ) -> Optional[torch.Tensor]:
        """
        Process attention mask to SDPA-compatible format.

        SDPA expects masks as:
        - None: no masking
        - (batch, 1, 1, ctx_len) or (batch, heads, seq, ctx) additive mask

        Supports input formats:
        - Boolean (batch, ctx_len): True = attend, False = mask
        - Additive (batch, 1, 1, ctx_len): 0 = attend, -inf/-10000 = mask
        - Full (batch, heads, seq, ctx): pass through
        """
        # Note: batch_size, seq_len, ctx_len reserved for future mask expansion
        del batch_size, seq_len, ctx_len  # Currently unused but part of interface

        if mask is None:
            return None

        # Handle different mask dimensions
        if mask.ndim == 2:
            # (batch, ctx_len) boolean mask
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, ctx_len)

        if mask.ndim == 3:
            # (batch, 1, ctx_len) -> add head dim
            mask = mask.unsqueeze(1)

        # Convert boolean to additive mask
        if mask.dtype == torch.bool:
            mask = torch.where(mask, 0.0, float("-inf"))

        return mask.to(dtype)

    def project_qk(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Project and optionally normalize Q and K without computing attention.

        Useful when you need to apply RoPE or other transformations to Q/K
        before computing attention.

        Args:
            x: Query input tensor (batch, seq_len, dim).
            context: Key/value context tensor. If None, uses x.

        Returns:
            Tuple of (q, k) tensors:
            - For INNER_DIM norm: (batch, seq_len, inner_dim), (batch, ctx_len, inner_dim)
            - For PER_HEAD norm: (batch, seq_len, num_heads, head_dim), same for k
            - For NONE: (batch, seq_len, inner_dim), (batch, ctx_len, inner_dim)
        """
        if context is None:
            context = x

        q = self.to_q(x)
        k = self.to_k(context)

        if self.qk_norm == QKNormType.INNER_DIM:
            q = self.norm_q(q)
            k = self.norm_k(k)
            # Return flat for external RoPE application
            return q, k

        elif self.qk_norm == QKNormType.PER_HEAD:
            q = q.unflatten(-1, (self.num_heads, self.head_dim))
            k = k.unflatten(-1, (self.num_heads, self.head_dim))
            q = self.norm_q(q)
            k = self.norm_k(k)
            return q, k

        else:
            return q, k

    def extra_repr(self) -> str:
        return (
            f"{self.dim}, num_heads={self.num_heads}, head_dim={self.head_dim}, "
            f"qk_norm={self.qk_norm.value}, dropout={self.dropout_prob}"
        )


# ============================================================================
# Preset Configurations
# ============================================================================

# These presets match the exact configuration of each original implementation.
# Use with: Attention(dim, num_heads=N, head_dim=D, **CONNECTOR_ATTN_PRESET)

CONNECTOR_ATTN_PRESET = dict(
    qk_norm=QKNormType.INNER_DIM,
    qk_norm_eps=1e-6,
    bias=True,
)
"""Connector Attention: Inner-dim QK norm, bias=True. Used by LTX-2 text encoder."""

ZIMAGE_ATTN_PRESET = dict(
    qk_norm=QKNormType.PER_HEAD,
    qk_norm_eps=1e-5,
    bias=False,
)
"""Z-Image Attention: Per-head QK norm, eps=1e-5, bias=False."""

CONTEXT_REFINER_ATTN_PRESET = dict(
    qk_norm=QKNormType.PER_HEAD,
    qk_norm_eps=1e-6,
    bias=False,
)
"""Context Refiner Attention: Per-head QK norm, eps=1e-6, bias=False."""
