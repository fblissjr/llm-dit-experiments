"""
Attention module for Z-Image DiT.

Last updated: 2026-02-01

Implements multi-head self-attention with:
- QK normalization (RMSNorm)
- Rotary Position Embeddings (RoPE)
- Multi-backend attention dispatch (Flash, SDPA, etc.)

Based on DiffSynth-Studio implementation.
"""

from typing import Optional

import torch
import torch.nn as nn

from llm_dit.layers import RMSNorm
from .rope import apply_rotary_emb


class Attention(nn.Module):
    """
    Multi-head Self-Attention with RoPE and QK Normalization.

    This attention module is designed for the Z-Image DiT transformer
    and supports rotary position embeddings applied after QK projection.

    Args:
        q_dim: Query input dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head
        kv_dim: Key/value input dimension (default: same as q_dim)
        bias_q: Use bias in query projection
        bias_kv: Use bias in key/value projections
        bias_out: Use bias in output projection
    """

    def __init__(
        self,
        q_dim: int,
        num_heads: int,
        head_dim: int,
        kv_dim: Optional[int] = None,
        bias_q: bool = False,
        bias_kv: bool = False,
        bias_out: bool = False,
    ):
        super().__init__()
        dim_inner = head_dim * num_heads
        kv_dim = kv_dim if kv_dim is not None else q_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Projections
        self.to_q = nn.Linear(q_dim, dim_inner, bias=bias_q)
        self.to_k = nn.Linear(kv_dim, dim_inner, bias=bias_kv)
        self.to_v = nn.Linear(kv_dim, dim_inner, bias=bias_kv)
        self.to_out = nn.ModuleList([nn.Linear(dim_inner, q_dim, bias=bias_out)])

        # QK normalization
        self.norm_q = RMSNorm(head_dim, eps=1e-5)
        self.norm_k = RMSNorm(head_dim, eps=1e-5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with RoPE and optional masking.

        Args:
            hidden_states: Input tensor (batch, seq_len, dim)
            freqs_cis: RoPE frequencies (seq_len, head_dim/2) complex
            attention_mask: Attention mask (batch, seq_len) boolean

        Returns:
            Output tensor (batch, seq_len, dim)
        """
        # Project to Q, K, V
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        # Reshape to (batch, seq_len, num_heads, head_dim)
        query = query.unflatten(-1, (self.num_heads, self.head_dim))
        key = key.unflatten(-1, (self.num_heads, self.head_dim))
        value = value.unflatten(-1, (self.num_heads, self.head_dim))

        # Apply QK normalization
        query = self.norm_q(query)
        key = self.norm_k(key)

        # Apply RoPE
        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        # Cast to consistent dtype
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # Compute attention using our attention backend
        hidden_states = self._attention_forward(query, key, value, attention_mask)

        # Reshape back: (batch, seq_len, num_heads * head_dim)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        # Output projection
        output = self.to_out[0](hidden_states)

        return output

    def _attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Dispatch attention computation to available backend.

        Args:
            query: (batch, seq_len, num_heads, head_dim)
            key: (batch, seq_len, num_heads, head_dim)
            value: (batch, seq_len, num_heads, head_dim)
            attention_mask: (batch, seq_len) boolean mask

        Returns:
            Attention output (batch, seq_len, num_heads, head_dim)
        """
        # Input is (batch, seq_len, num_heads, head_dim) - need (batch, heads, seq, dim) for backends
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Convert boolean mask to attention bias if provided
        attn_mask = None
        if attention_mask is not None:
            # attention_mask: (batch, seq_len) where True = attend, False = mask
            # SDPA expects: (batch, 1, 1, seq_len) where -inf = mask
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = torch.where(attn_mask, 0.0, float("-inf"))
            attn_mask = attn_mask.to(query.dtype)

        # Try to use our custom attention backend
        try:
            from llm_dit.utils.attention import attention_forward as custom_attn

            hidden_states = custom_attn(
                query,
                key,
                value,
                mask=attn_mask,
            )
        except ImportError:
            # Fallback to PyTorch SDPA
            hidden_states = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
            )

        # Transpose back to (batch, seq, heads, dim)
        hidden_states = hidden_states.transpose(1, 2)

        return hidden_states
