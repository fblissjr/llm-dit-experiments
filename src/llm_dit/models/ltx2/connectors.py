"""
LTX-2 Text Connectors for Video/Audio Generation.

Last Updated: 2026-01-18

Pure PyTorch implementation of the LTX-2 text connectors that process
Gemma3 hidden states before they're used in the DiT cross-attention.

Pipeline:
    Gemma3 (49 layers) -> Stack [B, T, 3840, 49]
        -> Normalize: 8 * (x - mean) / range
        -> Flatten to [B, T, 188160]
        -> text_proj_in: Linear(188160 -> 3840)
        -> video_connector: LTX2ConnectorTransformer1d (2 blocks, 128 registers)
        -> [B, T+128, 3840] ready for DiT caption_projection

The connector adds "thinking tokens" via learnable registers, enabling
the model to reason about the prompt before conditioning video generation.

Architecture Details:
    - 2 transformer blocks with self-attention + SwiGLU FFN
    - 30 heads x 128 head_dim = 3840 inner_dim
    - 128 learnable registers (prepended to sequence)
    - 1D RoPE for text position encoding (split variant)

Usage:
    from llm_dit.models.ltx2 import load_ltx2_connectors

    connectors = load_ltx2_connectors(
        "models/LTX-2/connectors/",
        device="cuda",
        dtype=torch.bfloat16,
    )

    # After encoding with Gemma3 and text_proj_in
    video_embeds, audio_embeds, attn_mask = connectors(
        packed_embeds,  # [B, T, 3840]
        attention_mask,  # [B, T]
    )
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# RoPE Implementation (1D for text sequences)
# =============================================================================

class LTX2RotaryPosEmbed1d(nn.Module):
    """
    1D rotary positional embeddings (RoPE) for text sequences.

    LTX-2 uses the "split" RoPE variant where cos/sin are applied to
    separate halves of each attention head's dimension.

    Args:
        dim: Dimension per attention head (will be split in half for rotation)
        base_seq_len: Maximum sequence length for normalization
        theta: RoPE base frequency
        num_attention_heads: Number of attention heads (for split variant reshaping)
    """

    def __init__(
        self,
        dim: int,
        base_seq_len: int = 4096,
        theta: float = 10000.0,
        num_attention_heads: int = 30,
    ):
        super().__init__()
        self.dim = dim
        self.base_seq_len = base_seq_len
        self.theta = theta
        self.num_attention_heads = num_attention_heads

    def forward(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute RoPE cos/sin frequencies for given sequence length.

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            device: Target device

        Returns:
            Tuple of (cos_freqs, sin_freqs), each [B, heads, T, dim//2]
        """
        # Position indices normalized by base_seq_len
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        positions = positions / self.base_seq_len  # [T]

        # Frequency bands: theta^(i / (dim/2))
        # We rotate dim/2 elements per head
        half_dim = self.dim // 2
        freqs_dtype = torch.float64  # Higher precision for frequency computation
        freq_indices = torch.arange(half_dim, dtype=freqs_dtype, device=device)
        freqs = torch.pow(self.theta, freq_indices / half_dim)
        freqs = freqs * torch.pi / 2.0  # Scale to [0, pi/2] range
        freqs = freqs.to(torch.float32)

        # Outer product: [T] x [dim/2] -> [T, dim/2]
        # Positions are shifted to [-1, 1] range before multiplication
        angles = (positions.unsqueeze(-1) * 2 - 1) * freqs  # [T, dim/2]

        cos_freqs = angles.cos()  # [T, dim/2]
        sin_freqs = angles.sin()  # [T, dim/2]

        # Reshape for multi-head attention: [B, H, T, dim/2]
        cos_freqs = cos_freqs.view(1, 1, seq_len, half_dim)
        sin_freqs = sin_freqs.view(1, 1, seq_len, half_dim)

        # Expand batch dimension
        cos_freqs = cos_freqs.expand(batch_size, self.num_attention_heads, -1, -1)
        sin_freqs = sin_freqs.expand(batch_size, self.num_attention_heads, -1, -1)

        return cos_freqs, sin_freqs


def apply_split_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply split RoPE to input tensor.

    The "split" variant applies rotation to the first and second halves
    of each head's dimension independently.

    Args:
        x: Input tensor [B, H, T, D] where D is head dimension
        cos: Cosine frequencies [B, H, T, D//2]
        sin: Sine frequencies [B, H, T, D//2]

    Returns:
        Rotated tensor [B, H, T, D]
    """
    # Preserve input dtype
    input_dtype = x.dtype

    # Cast cos/sin to input dtype
    cos = cos.to(input_dtype)
    sin = sin.to(input_dtype)

    # Split x into two halves along the head dimension
    x1, x2 = x.chunk(2, dim=-1)  # Each [B, H, T, D//2]

    # Apply rotation:
    # out1 = x1 * cos - x2 * sin
    # out2 = x2 * cos + x1 * sin
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin

    # Concatenate back
    return torch.cat([out1, out2], dim=-1)


# =============================================================================
# Attention and FFN Components
# =============================================================================

class LTX2Attention1d(nn.Module):
    """
    Multi-head self-attention for 1D sequences (text connector).

    Uses RMSNorm for Q/K normalization and supports RoPE position encoding.

    Args:
        dim: Model dimension (query_dim)
        num_heads: Number of attention heads
        head_dim: Dimension per head
        dropout: Dropout probability (default 0.0)
        bias: Use bias in projections (default True)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 30,
        head_dim: int = 128,
        dropout: float = 0.0,
        bias: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        # Q, K, V projections
        self.to_q = nn.Linear(dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=bias)

        # Q/K normalization (RMSNorm across heads)
        self.norm_q = nn.RMSNorm(self.inner_dim, eps=eps)
        self.norm_k = nn.RMSNorm(self.inner_dim, eps=eps)

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim, bias=True),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass for self-attention.

        Args:
            hidden_states: [B, T, D] input tensor
            attention_mask: [B, 1, 1, T] attention mask (additive, -inf for masked)
            rotary_emb: Tuple of (cos, sin) from RoPE [B, H, T, head_dim//2]

        Returns:
            [B, T, D] output tensor
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)

        # Normalize Q, K (before reshaping to heads)
        q = self.norm_q(q)
        k = self.norm_k(k)

        # Reshape to [B, H, T, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if provided
        if rotary_emb is not None:
            cos, sin = rotary_emb
            q = apply_split_rotary_emb(q, cos, sin)
            k = apply_split_rotary_emb(k, cos, sin)

        # Scaled dot-product attention
        hidden_states = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        # Reshape back to [B, T, inner_dim]
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, seq_len, self.inner_dim)

        # Output projection
        hidden_states = self.to_out(hidden_states)

        return hidden_states


class LTX2FeedForward1d(nn.Module):
    """
    Feed-forward network with SwiGLU activation for connector blocks.

    SwiGLU: Linear(dim, mult*dim) -> SiLU -> Linear(mult*dim, mult*dim) elementwise multiply -> Linear(mult*dim, dim)

    Diffusers uses their FeedForward with "gelu-approximate", but the connector
    actually matches the standard transformer FFN pattern.

    Args:
        dim: Model dimension
        mult: FFN hidden dimension multiplier (default 4)
        dropout: Dropout probability
    """

    def __init__(
        self,
        dim: int,
        mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = int(dim * mult)

        # SwiGLU: gate and up projections computed together
        self.proj_in = nn.Linear(dim, inner_dim)
        self.act = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.proj_out(x)
        return x


# =============================================================================
# Transformer Block
# =============================================================================

class LTX2TransformerBlock1d(nn.Module):
    """
    Transformer block for 1D sequences (text connector).

    Structure: RMSNorm -> Attention -> Residual -> RMSNorm -> FFN -> Residual

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head
        eps: RMSNorm epsilon
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 30,
        head_dim: int = 128,
        eps: float = 1e-6,
    ):
        super().__init__()

        # Self-attention with pre-norm
        self.norm1 = nn.RMSNorm(dim, eps=eps, elementwise_affine=False)
        self.attn1 = LTX2Attention1d(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # Feed-forward with pre-norm
        self.norm2 = nn.RMSNorm(dim, eps=eps, elementwise_affine=False)
        self.ff = LTX2FeedForward1d(dim=dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.

        Args:
            hidden_states: [B, T, D] input tensor
            attention_mask: [B, 1, 1, T] attention mask
            rotary_emb: RoPE frequencies (cos, sin)

        Returns:
            [B, T, D] output tensor
        """
        # Self-attention with residual
        norm_hidden = self.norm1(hidden_states)
        attn_out = self.attn1(norm_hidden, attention_mask, rotary_emb)
        hidden_states = hidden_states + attn_out

        # FFN with residual
        norm_hidden = self.norm2(hidden_states)
        ff_out = self.ff(norm_hidden)
        hidden_states = hidden_states + ff_out

        return hidden_states


# =============================================================================
# Main Connector Module
# =============================================================================

class LTX2ConnectorTransformer1d(nn.Module):
    """
    1D transformer connector for text sequences.

    Processes text embeddings through transformer blocks and adds learnable
    registers (thinking tokens) for enhanced reasoning before DiT conditioning.

    Architecture:
        - N transformer blocks (default 2)
        - Learnable registers prepended to non-padding positions
        - 1D RoPE for position encoding
        - Final RMSNorm

    Args:
        num_attention_heads: Number of attention heads (default 30)
        attention_head_dim: Dimension per head (default 128)
        num_layers: Number of transformer blocks (default 2)
        num_learnable_registers: Number of "thinking tokens" (default 128)
        rope_base_seq_len: RoPE base sequence length (default 4096)
        rope_theta: RoPE frequency base (default 10000.0)
        eps: Normalization epsilon
    """

    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 128,
        num_layers: int = 2,
        num_learnable_registers: int = 128,
        rope_base_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.num_learnable_registers = num_learnable_registers

        # Learnable registers (thinking tokens)
        if num_learnable_registers is not None:
            init_registers = torch.rand(num_learnable_registers, self.inner_dim) * 2.0 - 1.0
            self.learnable_registers = nn.Parameter(init_registers)
        else:
            self.learnable_registers = None

        # RoPE for position encoding
        self.rope = LTX2RotaryPosEmbed1d(
            dim=attention_head_dim,
            base_seq_len=rope_base_seq_len,
            theta=rope_theta,
            num_attention_heads=num_attention_heads,
        )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            LTX2TransformerBlock1d(
                dim=self.inner_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                eps=eps,
            )
            for _ in range(num_layers)
        ])

        # Final normalization
        self.norm_out = nn.RMSNorm(self.inner_dim, eps=eps, elementwise_affine=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attn_mask_binarize_threshold: float = -9000.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process text embeddings through connector.

        Args:
            hidden_states: [B, T, D] text embeddings after text_proj_in
            attention_mask: [B, T] or [B, 1, 1, T] attention mask
            attn_mask_binarize_threshold: Threshold for binarizing attention mask

        Returns:
            Tuple of:
                - processed_embeds: [B, T, D] processed text embeddings
                - new_attn_mask: [B, T] updated attention mask
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Handle learnable registers
        if self.learnable_registers is not None:
            if seq_len % self.num_learnable_registers != 0:
                raise ValueError(
                    f"Sequence length {seq_len} should be divisible by "
                    f"num_learnable_registers {self.num_learnable_registers}"
                )

            # Tile registers to match sequence length
            num_repeats = seq_len // self.num_learnable_registers
            registers = self.learnable_registers.tile(num_repeats, 1)  # [T, D]

            # Binarize attention mask
            if attention_mask is not None:
                binary_mask = (attention_mask >= attn_mask_binarize_threshold).int()
                if binary_mask.ndim == 4:
                    binary_mask = binary_mask.squeeze(1).squeeze(1)  # [B, 1, 1, T] -> [B, T]

                # Extract non-padding tokens and left-align
                hidden_states_list = []
                valid_lens = []
                for i in range(batch_size):
                    mask = binary_mask[i].bool()
                    valid = hidden_states[i, mask, :]  # [valid_len, D]
                    valid_lens.append(valid.shape[0])
                    # Pad to full length
                    padded = F.pad(valid, (0, 0, 0, seq_len - valid.shape[0]))
                    hidden_states_list.append(padded)

                hidden_states_packed = torch.stack(hidden_states_list, dim=0)

                # Replace padding positions with registers
                # Flip mask so registers go in the padding region (right side after packing)
                flipped_mask = torch.flip(binary_mask, dims=[1]).unsqueeze(-1)  # [B, T, 1]
                hidden_states = flipped_mask * hidden_states_packed + (1 - flipped_mask) * registers

                # New attention mask is all-ones (registers are valid)
                attention_mask = torch.zeros_like(attention_mask)
            else:
                # No mask provided - just use registers at the end
                hidden_states = hidden_states  # Keep as-is

        # Compute RoPE
        rotary_emb = self.rope(batch_size, seq_len, hidden_states.device)

        # Run transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, attention_mask, rotary_emb)

        # Final normalization
        hidden_states = self.norm_out(hidden_states)

        return hidden_states, attention_mask


class LTX2TextConnectors(nn.Module):
    """
    Complete text connector stack for LTX-2.

    Combines:
        1. text_proj_in: Linear(188160 -> 3840) - aggregates multi-layer features
        2. video_connector: Transformer for video conditioning
        3. audio_connector: Transformer for audio conditioning

    Args:
        caption_channels: Output dimension (3840 for Gemma3)
        text_proj_in_factor: Number of layers aggregated (49)
        video/audio connector parameters: See LTX2ConnectorTransformer1d
    """

    def __init__(
        self,
        caption_channels: int = 3840,
        text_proj_in_factor: int = 49,
        video_connector_num_attention_heads: int = 30,
        video_connector_attention_head_dim: int = 128,
        video_connector_num_layers: int = 2,
        video_connector_num_learnable_registers: int = 128,
        audio_connector_num_attention_heads: int = 30,
        audio_connector_attention_head_dim: int = 128,
        audio_connector_num_layers: int = 2,
        audio_connector_num_learnable_registers: int = 128,
        connector_rope_base_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        # Input projection: [B, T, 188160] -> [B, T, 3840]
        self.text_proj_in = nn.Linear(
            caption_channels * text_proj_in_factor,
            caption_channels,
            bias=False,
        )

        # Video connector
        self.video_connector = LTX2ConnectorTransformer1d(
            num_attention_heads=video_connector_num_attention_heads,
            attention_head_dim=video_connector_attention_head_dim,
            num_layers=video_connector_num_layers,
            num_learnable_registers=video_connector_num_learnable_registers,
            rope_base_seq_len=connector_rope_base_seq_len,
            rope_theta=rope_theta,
            eps=eps,
        )

        # Audio connector
        self.audio_connector = LTX2ConnectorTransformer1d(
            num_attention_heads=audio_connector_num_attention_heads,
            attention_head_dim=audio_connector_attention_head_dim,
            num_layers=audio_connector_num_layers,
            num_learnable_registers=audio_connector_num_learnable_registers,
            rope_base_seq_len=connector_rope_base_seq_len,
            rope_theta=rope_theta,
            eps=eps,
        )

    def forward(
        self,
        text_encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        additive_mask: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process text embeddings through both connectors.

        Args:
            text_encoder_hidden_states: [B, T, 188160] packed multi-layer features
            attention_mask: [B, T] binary mask (1=valid, 0=padding)
            additive_mask: If False, convert binary mask to additive format

        Returns:
            Tuple of:
                - video_text_embedding: [B, T, 3840]
                - audio_text_embedding: [B, T, 3840]
                - new_attn_mask: [B, T]
        """
        # Convert to additive attention mask if needed
        if not additive_mask:
            text_dtype = text_encoder_hidden_states.dtype
            # Binary (0,1) -> Additive (-inf, 0): mask = (mask - 1) * inf
            attn_mask = (attention_mask - 1).reshape(
                attention_mask.shape[0], 1, 1, attention_mask.shape[-1]
            )
            attn_mask = attn_mask.to(text_dtype) * torch.finfo(text_dtype).max
        else:
            attn_mask = attention_mask

        # Project multi-layer features to connector dimension
        text_encoder_hidden_states = self.text_proj_in(text_encoder_hidden_states)

        # Process through video connector
        video_text_embedding, new_attn_mask = self.video_connector(
            text_encoder_hidden_states, attn_mask
        )

        # Apply updated mask to video embeddings
        if new_attn_mask is not None:
            binary_mask = (new_attn_mask < 1e-6).to(torch.int64)
            binary_mask = binary_mask.reshape(
                video_text_embedding.shape[0], video_text_embedding.shape[1], 1
            )
            video_text_embedding = video_text_embedding * binary_mask
            new_attn_mask = binary_mask.squeeze(-1)

        # Process through audio connector
        audio_text_embedding, _ = self.audio_connector(
            text_encoder_hidden_states, attn_mask
        )

        return video_text_embedding, audio_text_embedding, new_attn_mask


# =============================================================================
# Weight Loading
# =============================================================================

def load_ltx2_connectors(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> LTX2TextConnectors:
    """
    Load LTX-2 text connectors from checkpoint.

    Args:
        model_path: Path to connector weights directory (containing model.safetensors)
                   or direct path to safetensors file
        device: Target device
        dtype: Model dtype

    Returns:
        Loaded LTX2TextConnectors module
    """
    from safetensors.torch import load_file

    model_path = Path(model_path)

    # Find weights file
    if model_path.is_file():
        weights_path = model_path
    else:
        # Try common names
        for name in ["model.safetensors", "connectors.safetensors", "diffusion_pytorch_model.safetensors"]:
            candidate = model_path / name
            if candidate.exists():
                weights_path = candidate
                break
        else:
            raise FileNotFoundError(f"No safetensors file found in {model_path}")

    logger.info(f"Loading connectors from {weights_path}")

    # Load state dict
    state_dict = load_file(str(weights_path))

    # Create model with default config (matches LTX-2 official)
    model = LTX2TextConnectors()

    # Map diffusers state dict keys to our module
    # Diffusers uses slightly different naming
    mapped_state_dict = _map_diffusers_state_dict(state_dict)

    # Load weights
    missing, unexpected = model.load_state_dict(mapped_state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys when loading connectors: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys when loading connectors: {unexpected}")

    model = model.to(device=device, dtype=dtype)
    model.requires_grad_(False)

    return model


def _map_diffusers_state_dict(state_dict: dict) -> dict:
    """
    Map diffusers connector state dict keys to our module structure.

    Diffusers LTX2TextConnectors has same structure, but attention modules
    use their processor pattern. Our implementation is self-contained.
    """
    mapped = {}

    for key, value in state_dict.items():
        new_key = key

        # Map attention processor patterns if any
        # Most keys should match directly since we use same structure

        # Handle attention.to_out which is ModuleList in diffusers
        if "attn1.to_out.0" in key:
            new_key = key  # Keep as-is, we use same structure
        elif "attn1.to_out.1" in key:
            new_key = key  # Keep as-is

        # Handle FeedForward mapping
        # Diffusers FeedForward has: net.0.proj (GEGLU) + net.2 (Linear out)
        # Our FFN has: proj_in, act, proj_out
        if ".ff.net.0.proj." in key:
            new_key = key.replace(".ff.net.0.proj.", ".ff.proj_in.")
        elif ".ff.net.2." in key:
            new_key = key.replace(".ff.net.2.", ".ff.proj_out.")

        mapped[new_key] = value

    return mapped


def load_video_connector_only(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> LTX2ConnectorTransformer1d:
    """
    Load only the video connector (for video-only experiments).

    Args:
        model_path: Path to connector weights
        device: Target device
        dtype: Model dtype

    Returns:
        Loaded LTX2ConnectorTransformer1d for video
    """
    connectors = load_ltx2_connectors(model_path, device, dtype)
    return connectors.video_connector
