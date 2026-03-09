"""
Embeddings 1D Connector for LTX-2.3 text encoder.

Last Updated: 2026-03-06

Ported from:
  coderef/LTX-2/packages/ltx-core/src/ltx_core/text_encoders/gemma/embeddings_connector.py

This module implements a bidirectional transformer that processes text embeddings
after the feature extractor projection. V2.3 configuration:
- 8 transformer layers with RoPE positional encoding
- 128 learnable register tokens that replace padding
- Self-attention + feed-forward blocks with RMSNorm
- Per-head sigmoid gated attention (apply_gated_attention=True)
- Video: 32 heads * 128 head_dim = 4096 inner_dim
- Audio: 32 heads * 64 head_dim = 2048 inner_dim
"""

import functools
import logging
import math
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn

from llm_dit.layers import rms_norm

logger = logging.getLogger(__name__)


# ============================================================================
# RoPE (Rotary Position Embedding) Utilities
# ============================================================================

class RopeType(Enum):
    """RoPE implementation variant."""
    INTERLEAVED = "interleaved"
    SPLIT = "split"


def apply_rotary_emb(
    input_tensor: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    rope_type: RopeType = RopeType.INTERLEAVED,
) -> torch.Tensor:
    """Apply rotary position embeddings to input tensor."""
    cos_freqs, sin_freqs = freqs_cis
    if rope_type == RopeType.INTERLEAVED:
        return _apply_interleaved_rotary_emb(input_tensor, cos_freqs, sin_freqs)
    elif rope_type == RopeType.SPLIT:
        return _apply_split_rotary_emb(input_tensor, cos_freqs, sin_freqs)
    else:
        raise ValueError(f"Invalid rope type: {rope_type}")


def _apply_interleaved_rotary_emb(
    input_tensor: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
) -> torch.Tensor:
    """Interleaved RoPE: pairs of dimensions rotated together."""
    # Reshape to pairs: [..., d, 2]
    t_dup = input_tensor.reshape(*input_tensor.shape[:-1], -1, 2)
    t1, t2 = t_dup.unbind(dim=-1)

    # Rotate
    t_rot = torch.stack((-t2, t1), dim=-1)
    input_tensor_rot = t_rot.reshape(*input_tensor.shape)

    return input_tensor * cos_freqs + input_tensor_rot * sin_freqs


def _apply_split_rotary_emb(
    input_tensor: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
) -> torch.Tensor:
    """
    Split RoPE: first half and second half rotated separately.

    Matches reference implementation from LTX-2:
    coderef/LTX-2/packages/ltx-core/src/ltx_core/model/transformer/rope.py
    """
    from einops import rearrange

    needs_reshape = False
    if input_tensor.ndim != 4 and cos_freqs.ndim == 4:
        # Get batch from input_tensor (not cos_freqs, which may have batch=1 for broadcasting)
        b = input_tensor.shape[0]
        _, h, t, _ = cos_freqs.shape
        input_tensor = input_tensor.reshape(b, t, h, -1).swapaxes(1, 2)
        needs_reshape = True

    # Split into halves using einops: d=2 means first new dim is 2
    # [..., 128] -> [..., 2, 64]
    split_input = rearrange(input_tensor, "... (d r) -> ... d r", d=2)
    first_half_input = split_input[..., :1, :]
    second_half_input = split_input[..., 1:, :]

    # Apply rotation with in-place operations (matching reference)
    output = split_input * cos_freqs.unsqueeze(-2)
    first_half_output = output[..., :1, :]
    second_half_output = output[..., 1:, :]

    # addcmul_: a += value * b * c (in-place)
    first_half_output.addcmul_(-sin_freqs.unsqueeze(-2), second_half_input)
    second_half_output.addcmul_(sin_freqs.unsqueeze(-2), first_half_input)

    # Flatten back: [..., 2, 64] -> [..., 128]
    output = rearrange(output, "... d r -> ... (d r)")
    if needs_reshape:
        # At this point output is [b, h, t, d] - reshape back to [b, t, h*d]
        b_out, _, t_out, _ = output.shape
        output = output.swapaxes(1, 2).reshape(b_out, t_out, -1)

    return output


@functools.lru_cache(maxsize=5)
def _generate_freq_grid(
    theta: float,
    max_pos_count: int,
    inner_dim: int,
    use_double: bool = True,
) -> torch.Tensor:
    """Generate frequency grid for RoPE."""
    start = 1.0
    end = theta
    n_elem = 2 * max_pos_count

    if use_double:
        # Double precision (numpy-like)
        import numpy as np
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
    else:
        # Standard precision
        indices = theta ** torch.linspace(
            math.log(start, theta),
            math.log(end, theta),
            inner_dim // n_elem,
            dtype=torch.float32,
        )
        return indices * math.pi / 2


def _get_fractional_positions(
    indices_grid: torch.Tensor,
    max_pos: list[int],
) -> torch.Tensor:
    """Convert indices to fractional positions."""
    n_pos_dims = indices_grid.shape[1]
    assert n_pos_dims == len(max_pos), (
        f"Position dims ({n_pos_dims}) must match max_pos length ({len(max_pos)})"
    )
    fractional_positions = torch.stack(
        [indices_grid[:, i] / max_pos[i] for i in range(n_pos_dims)],
        dim=-1,
    )
    return fractional_positions


def _generate_freqs(
    indices: torch.Tensor,
    indices_grid: torch.Tensor,
    max_pos: list[int],
) -> torch.Tensor:
    """Generate frequencies from indices grid."""
    # Handle 4D indices_grid by taking first slice
    if len(indices_grid.shape) == 4:
        indices_grid = indices_grid[..., 0]

    fractional_positions = _get_fractional_positions(indices_grid, max_pos)
    indices = indices.to(device=fractional_positions.device)

    # Compute frequencies
    freqs = (indices * (fractional_positions.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)
    return freqs


def _split_freqs_cis(
    freqs: torch.Tensor,
    pad_size: int,
    num_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split frequencies into cos/sin components for split RoPE."""
    cos_freq = freqs.cos()
    sin_freq = freqs.sin()

    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = torch.zeros_like(sin_freq[:, :, :pad_size])
        cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
        sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)

    # Reshape for multi-head attention
    b, t, _ = cos_freq.shape
    cos_freq = cos_freq.reshape(b, t, num_heads, -1).swapaxes(1, 2)
    sin_freq = sin_freq.reshape(b, t, num_heads, -1).swapaxes(1, 2)

    return cos_freq, sin_freq


def _interleaved_freqs_cis(
    freqs: torch.Tensor,
    pad_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create interleaved cos/sin frequencies."""
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
    max_pos: Optional[list[int]] = None,
    num_attention_heads: int = 30,
    rope_type: RopeType = RopeType.INTERLEAVED,
    use_double_precision: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute RoPE frequencies for given indices.

    Args:
        indices_grid: Position indices [B, pos_dims, seq_len]
        dim: Hidden dimension
        out_dtype: Output dtype
        theta: RoPE theta parameter
        max_pos: Maximum positions per dimension
        num_attention_heads: Number of attention heads
        rope_type: SPLIT or INTERLEAVED
        use_double_precision: Use float64 for frequency computation
    """
    if max_pos is None:
        max_pos = [1]  # Default from LTX-2 reference (line 109)

    indices = _generate_freq_grid(theta, indices_grid.shape[1], dim, use_double_precision)
    freqs = _generate_freqs(indices, indices_grid, max_pos)

    if rope_type == RopeType.SPLIT:
        expected_freqs = dim // 2
        current_freqs = freqs.shape[-1]
        pad_size = expected_freqs - current_freqs
        cos_freq, sin_freq = _split_freqs_cis(freqs, pad_size, num_attention_heads)
    else:
        n_elem = 2 * indices_grid.shape[1]
        cos_freq, sin_freq = _interleaved_freqs_cis(freqs, dim % n_elem)

    return cos_freq.to(out_dtype), sin_freq.to(out_dtype)


# ============================================================================
# Transformer Components
# ============================================================================

# Note: rms_norm is imported from llm_dit.layers

class GELUApprox(nn.Module):
    """GELU with tanh approximation."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.proj(x), approximate="tanh")


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, dim: int, dim_out: int, mult: int = 4):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            GELUApprox(dim, inner_dim),
            nn.Identity(),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    """
    Multi-head attention with RoPE support.

    Supports both self-attention and cross-attention.
    Optional per-head gated attention (V2.3): gate = 2 * sigmoid(logits),
    applied BEFORE to_out projection. Zero-init gives identity (2 * 0.5 = 1.0).
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 30,
        dim_head: int = 128,
        norm_eps: float = 1e-6,
        rope_type: RopeType = RopeType.INTERLEAVED,
        apply_gated_attention: bool = False,
    ):
        super().__init__()
        self.rope_type = rope_type
        self.heads = heads
        self.dim_head = dim_head

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        # QKV projections
        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=True)

        # QK normalization (important for stability)
        # Note: naming matches checkpoint keys (norm_q, norm_k)
        self.q_norm = nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = nn.RMSNorm(inner_dim, eps=norm_eps)

        # Per-head gating (V2.3): applied before output projection
        if apply_gated_attention:
            self.to_gate_logits = nn.Linear(query_dim, heads, bias=True)
        else:
            self.to_gate_logits = None

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, bias=True),
            nn.Identity(),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        pe: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        q = self.to_q(x)
        context = x if context is None else context
        k = self.to_k(context)
        v = self.to_v(context)

        # Normalize Q and K
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE if provided
        if pe is not None:
            q = apply_rotary_emb(q, pe, self.rope_type)
            k = apply_rotary_emb(k, pe, self.rope_type)

        # Reshape for multi-head attention
        b, seq_len, _ = q.shape
        q = q.view(b, seq_len, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(b, seq_len, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(b, seq_len, self.heads, self.dim_head).transpose(1, 2)

        # Attention mask handling
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        # Scaled dot-product attention (PyTorch native)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
        )

        # Reshape back
        out = out.transpose(1, 2).reshape(b, seq_len, self.heads * self.dim_head)

        # Per-head gating: 2 * sigmoid(x), applied BEFORE to_out
        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)  # [B, T, heads]
            gates = 2.0 * torch.sigmoid(gate_logits)
            out = out.view(b, seq_len, self.heads, self.dim_head)
            out = out * gates.unsqueeze(-1)
            out = out.view(b, seq_len, self.heads * self.dim_head)

        return self.to_out(out)


class BasicTransformerBlock1D(nn.Module):
    """
    Basic transformer block for 1D sequence processing.

    Pre-norm architecture:
    1. RMSNorm -> Self-Attention -> Residual
    2. RMSNorm -> FeedForward -> Residual
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rope_type: RopeType = RopeType.INTERLEAVED,
        apply_gated_attention: bool = False,
    ):
        super().__init__()
        self.attn1 = Attention(
            query_dim=dim,
            heads=heads,
            dim_head=dim_head,
            rope_type=rope_type,
            apply_gated_attention=apply_gated_attention,
        )
        self.ff = FeedForward(dim, dim_out=dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pe: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # 1. Pre-norm self-attention (gating handled inside Attention)
        norm_hidden_states = rms_norm(hidden_states)
        if norm_hidden_states.ndim == 4:
            norm_hidden_states = norm_hidden_states.squeeze(1)

        attn_output = self.attn1(norm_hidden_states, mask=attention_mask, pe=pe)

        hidden_states = attn_output + hidden_states

        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 2. Pre-norm feed-forward
        norm_hidden_states = rms_norm(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states

        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


# ============================================================================
# Main Connector Module
# ============================================================================

class Embeddings1DConnector(nn.Module):
    """
    1D transformer connector for text embeddings.

    Processes embeddings after feature extraction with:
    - 2 transformer layers using RoPE
    - 128 learnable registers to replace padding tokens
    - Final RMSNorm

    Args:
        attention_head_dim: Dimension per attention head (default 128)
        num_attention_heads: Number of heads (default 30, giving 3840 dim)
        num_layers: Number of transformer layers (default 2)
        positional_embedding_theta: RoPE theta (default 10000.0)
        positional_embedding_max_pos: Max positions for RoPE (default [4096])
        num_learnable_registers: Registers replacing padding (default 128)
        rope_type: RoPE variant (default SPLIT)
        use_double_precision_rope: Use float64 for RoPE computation
    """

    def __init__(
        self,
        attention_head_dim: int = 128,
        num_attention_heads: int = 30,
        num_layers: int = 2,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: Optional[list[int]] = None,
        num_learnable_registers: int = 128,
        rope_type: RopeType = RopeType.INTERLEAVED,
        use_double_precision_rope: bool = False,
        apply_gated_attention: bool = False,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = (
            positional_embedding_max_pos if positional_embedding_max_pos is not None else [4096]
        )
        self.rope_type = rope_type
        self.use_double_precision_rope = use_double_precision_rope

        # Transformer blocks
        self.transformer_1d_blocks = nn.ModuleList([
            BasicTransformerBlock1D(
                dim=self.inner_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                rope_type=rope_type,
                apply_gated_attention=apply_gated_attention,
            )
            for _ in range(num_layers)
        ])

        # Learnable registers
        self.num_learnable_registers = num_learnable_registers
        if num_learnable_registers:
            # Initialize uniformly in [-1, 1] like LTX-2 reference
            # Don't hardcode dtype - let .to() calls handle dtype conversion
            self.learnable_registers = nn.Parameter(
                torch.rand(num_learnable_registers, self.inner_dim) * 2.0 - 1.0
            )

    def _replace_padded_with_learnable_registers(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Replace padding tokens with learnable registers.

        Valid tokens are compacted to the LEFT of the sequence, with
        learnable registers filling the remaining RIGHT positions.
        This matches the reference LTX-2 implementation and how the model
        was trained (valid tokens at positions 0..N-1, registers at N..seq_len-1).

        The reference uses a flip-based approach: extract valid tokens,
        left-align them in a zero-padded buffer, then use the flipped binary
        mask to select tokens vs registers element-wise.
        """
        seq_len = hidden_states.shape[1]
        assert seq_len % self.num_learnable_registers == 0, (
            f"Sequence length {seq_len} must be divisible by "
            f"num_learnable_registers {self.num_learnable_registers}"
        )

        num_duplications = seq_len // self.num_learnable_registers
        learnable_registers = self.learnable_registers.repeat(num_duplications, 1)
        learnable_registers = learnable_registers.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )

        # Convert additive mask to binary [B, T, 1]: 1 = valid, 0 = padding
        # Additive mask: 0 = valid, -10000 = padding
        attention_mask_binary = (attention_mask.squeeze(1).squeeze(1).unsqueeze(-1) >= -9000.0).int()

        # Per-batch: extract valid tokens, left-align, then fill right with registers.
        # This matches the reference _replace_padded_with_learnable_registers behavior.
        batch_size = hidden_states.shape[0]
        results = []

        for b in range(batch_size):
            mask_b = attention_mask_binary[b, :, 0].bool()  # [T]
            valid_tokens = hidden_states[b, mask_b, :]       # [num_valid, D]
            num_valid = valid_tokens.shape[0]
            pad_length = seq_len - num_valid

            # Left-align valid tokens, pad zeros on the right (same as reference F.pad)
            adjusted = torch.nn.functional.pad(
                valid_tokens.unsqueeze(0), (0, 0, 0, pad_length)
            ).squeeze(0)  # [seq_len, D]

            # Flip the per-sample mask along the sequence dim.
            # Original mask (left-padded input): [0,0,...,1,1,...] (padding left, valid right)
            # Flipped: [1,1,...,0,0,...] -- ones at the left where valid tokens now sit.
            mask_1d = attention_mask_binary[b, :, 0]  # [T] int
            flipped_mask = mask_1d.flip(0).unsqueeze(-1)  # [T, 1] int, 1 where valid lands

            result = (
                flipped_mask * adjusted
                + (1 - flipped_mask) * learnable_registers
            )
            results.append(result)

        hidden_states = torch.stack(results, dim=0)

        # Create all-valid additive mask (registers are now valid tokens)
        attention_mask = torch.full_like(
            attention_mask, 0.0,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        return hidden_states, attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process embeddings through transformer layers.

        Args:
            hidden_states: [B, seq_len, 3840] - Input embeddings
            attention_mask: [B, 1, 1, seq_len] - Additive attention mask
                           (0 = valid, -10000 = padding)

        Returns:
            hidden_states: [B, seq_len, 3840] - Processed embeddings
            attention_mask: [B, 1, 1, seq_len] or None - Updated mask (all valid after register insertion)
        """
        # Trace input
        logger.debug(
            f"[CONNECTOR] Input: shape={list(hidden_states.shape)}, "
            f"mean={hidden_states.float().mean():.4f}, std={hidden_states.float().std():.4f}"
        )

        # Replace padding with learnable registers
        if self.num_learnable_registers and attention_mask is not None:
            hidden_states, attention_mask = self._replace_padded_with_learnable_registers(
                hidden_states, attention_mask
            )
            logger.debug(
                f"[CONNECTOR] After registers: mean={hidden_states.float().mean():.4f}, "
                f"std={hidden_states.float().std():.4f}"
            )

        # Compute RoPE frequencies
        indices_grid = torch.arange(
            hidden_states.shape[1],
            dtype=torch.float32,
            device=hidden_states.device,
        )
        indices_grid = indices_grid[None, None, :]  # [1, 1, seq_len]

        freqs_cis = precompute_freqs_cis(
            indices_grid=indices_grid,
            dim=self.inner_dim,
            out_dtype=hidden_states.dtype,
            theta=self.positional_embedding_theta,
            max_pos=self.positional_embedding_max_pos,
            num_attention_heads=self.num_attention_heads,
            rope_type=self.rope_type,
            use_double_precision=self.use_double_precision_rope,
        )

        logger.debug(
            f"[CONNECTOR] RoPE config: type={self.rope_type.value}, "
            f"max_pos={self.positional_embedding_max_pos}, double_precision={self.use_double_precision_rope}"
        )

        # Process through transformer blocks
        for i, block in enumerate(self.transformer_1d_blocks):
            hidden_states = block(hidden_states, attention_mask=attention_mask, pe=freqs_cis)
            logger.debug(
                f"[CONNECTOR] After block {i}: mean={hidden_states.float().mean():.4f}, "
                f"std={hidden_states.float().std():.4f}"
            )

        # Final normalization
        hidden_states = rms_norm(hidden_states)

        return hidden_states, attention_mask

    @classmethod
    def from_config(cls, config: dict) -> "Embeddings1DConnector":
        """Create connector from config dict."""
        rope_type = RopeType(config.get("rope_type", "interleaved"))
        use_double_precision = config.get("rope_double_precision", False)
        max_pos = config.get("connector_positional_embedding_max_pos", [4096])

        return cls(
            attention_head_dim=config.get("video_connector_attention_head_dim", 128),
            num_attention_heads=config.get("video_connector_num_attention_heads", 30),
            num_layers=config.get("video_connector_num_layers", 2),
            positional_embedding_theta=config.get("rope_theta", 10000.0),
            positional_embedding_max_pos=max_pos,
            num_learnable_registers=config.get("video_connector_num_learnable_registers", 128),
            rope_type=rope_type,
            use_double_precision_rope=use_double_precision,
            apply_gated_attention=config.get("apply_gated_attention", False),
        )


def load_connector_weights(
    connector: Embeddings1DConnector,
    checkpoint_path: Path,
    prefix: str = "video_connector.",
) -> None:
    """
    Load weights into connector from checkpoint.

    Args:
        connector: Connector instance to load weights into
        checkpoint_path: Path to safetensors checkpoint
        prefix: Key prefix in checkpoint (default "video_connector.")
    """
    from safetensors import safe_open

    with safe_open(checkpoint_path, framework="pt") as f:
        state_dict = {}
        for key in f.keys():
            if key.startswith(prefix):
                # Strip prefix for loading
                new_key = key[len(prefix):]
                state_dict[new_key] = f.get_tensor(key)

    # Load state dict
    missing, unexpected = connector.load_state_dict(state_dict, strict=False)

    if missing:
        logger.warning(f"Missing keys when loading connector: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys when loading connector: {unexpected}")

    logger.info(f"Loaded {len(state_dict)} weights into embeddings connector")
