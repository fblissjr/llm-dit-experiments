"""
Core building blocks for LTX-2 transformer.

Last Updated: 2026-01-18

Contains the fundamental components used in the LTX-2 DiT architecture:
- GELUApprox: GELU activation with linear projection
- FeedForward: Standard DiT feed-forward network
- Timesteps: Sinusoidal timestep embeddings
- TimestepEmbedding: MLP for projecting timestep embeddings
- PixArtAlphaCombinedTimestepSizeEmbeddings: Combined timestep embeddings
- AdaLayerNormSingle: Adaptive layer normalization for timestep conditioning
- PixArtAlphaTextProjection: Caption embedding projection

Ported from: coderef/LTX-2/packages/ltx-core/src/ltx_core/model/transformer/

Usage:
    from llm_dit.models.ltx2 import (
        FeedForward,
        AdaLayerNormSingle,
        PixArtAlphaTextProjection,
    )

    # Create feed-forward block
    ff = FeedForward(dim=4096, dim_out=4096, mult=4)

    # Create timestep conditioning
    adaln = AdaLayerNormSingle(embedding_dim=4096)
    timestep_emb, embedded_timestep = adaln(timestep, hidden_dtype=torch.bfloat16)

    # Project text embeddings
    text_proj = PixArtAlphaTextProjection(in_features=3840, hidden_size=4096)
    projected = text_proj(caption_embeds)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


class GELUApprox(nn.Module):
    """
    GELU activation with tanh approximation and linear projection.

    Combines a linear layer with GELU(tanh) activation in a single module.
    The tanh approximation is faster than the exact GELU implementation
    while maintaining nearly identical behavior.

    Args:
        dim_in: Input dimension
        dim_out: Output dimension
    """

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.proj(x), approximate="tanh")


class FeedForward(nn.Module):
    """
    Standard feed-forward network for diffusion transformers.

    Structure: Linear -> GELU(tanh) -> Identity -> Linear

    The Identity layer is kept for compatibility with the official implementation
    and may be used for dropout or other modifications in training.

    Args:
        dim: Input dimension
        dim_out: Output dimension
        mult: Hidden dimension multiplier (default 4, so hidden = dim * 4)
    """

    def __init__(self, dim: int, dim_out: int, mult: int = 4) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        project_in = GELUApprox(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Identity(),  # Placeholder for dropout if needed
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Based on the implementation from "Denoising Diffusion Probabilistic Models".
    Creates embeddings using sine and cosine functions at various frequencies.

    Args:
        timesteps: 1-D tensor of N timesteps (can be fractional)
        embedding_dim: Dimension of the output embeddings
        flip_sin_to_cos: If True, output is [cos, sin] instead of [sin, cos]
        downscale_freq_shift: Controls frequency spacing between dimensions
        scale: Scaling factor applied to embeddings
        max_period: Controls maximum frequency of embeddings

    Returns:
        [N, embedding_dim] tensor of positional embeddings
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2

    # Create frequency bands: exp(-log(max_period) * i / half_dim)
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # Apply scaling
    emb = scale * emb

    # Concatenate sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # Optionally flip order
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # Zero-pad if embedding_dim is odd
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

    return emb


class Timesteps(nn.Module):
    """
    Module wrapper for sinusoidal timestep embeddings.

    Wraps get_timestep_embedding() as an nn.Module for use in model architectures.

    Args:
        num_channels: Dimension of timestep embeddings
        flip_sin_to_cos: If True, output is [cos, sin] order
        downscale_freq_shift: Controls frequency spacing
        scale: Scaling factor for embeddings
    """

    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: int = 1
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb


class TimestepEmbedding(nn.Module):
    """
    MLP for projecting timestep embeddings.

    Takes raw sinusoidal timestep embeddings and projects them through
    a two-layer MLP with SiLU activation.

    Args:
        in_channels: Input dimension (from Timesteps)
        time_embed_dim: Hidden dimension
        out_dim: Output dimension (defaults to time_embed_dim)
        post_act_fn: Post-activation function (not used in LTX-2)
        cond_proj_dim: Optional conditioning projection dimension
        sample_proj_bias: Whether to use bias in linear layers
    """

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        out_dim: Optional[int] = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim: Optional[int] = None,
        sample_proj_bias: bool = True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = nn.SiLU()
        time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim

        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None

    def forward(
        self,
        sample: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if condition is not None:
            sample = sample + self.cond_proj(condition)

        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)

        return sample


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    """
    Combined timestep embeddings in PixArt-Alpha style.

    Creates timestep embeddings by:
    1. Computing sinusoidal embeddings (Timesteps)
    2. Projecting through MLP (TimestepEmbedding)

    Reference: https://arxiv.org/abs/2310.00426 (PixArt-Alpha)

    Args:
        embedding_dim: Output embedding dimension
        size_emb_dim: Size embedding dimension (embedding_dim // 3)
    """

    def __init__(
        self,
        embedding_dim: int,
        size_emb_dim: int,
    ):
        super().__init__()

        self.outdim = size_emb_dim
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_dtype: torch.dtype,
    ) -> torch.Tensor:
        # Create sinusoidal embedding
        timesteps_proj = self.time_proj(timestep)
        # Project to final dimension
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))
        return timesteps_emb


class AdaLayerNormSingle(nn.Module):
    """
    Adaptive Layer Normalization (adaLN-single) for timestep conditioning.

    Computes per-sample scale and shift parameters from timestep embeddings,
    which are then applied to normalize hidden states. This is the primary
    mechanism for conditioning the DiT on diffusion timestep.

    As proposed in PixArt-Alpha (https://arxiv.org/abs/2310.00426, Section 2.3).

    Args:
        embedding_dim: Size of each embedding vector
        embedding_coefficient: Number of scale-shift values to produce (default 6)
            - For transformer blocks: 6 values (shift_msa, scale_msa, gate_msa,
              shift_mlp, scale_mlp, gate_mlp)
            - For output: 2 values (shift, scale)
    """

    def __init__(self, embedding_dim: int, embedding_coefficient: int = 6):
        super().__init__()

        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim,
            size_emb_dim=embedding_dim // 3,
        )

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_coefficient * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adaptive normalization parameters from timestep.

        Args:
            timestep: Diffusion timesteps [B] or [B*T] if per-token timesteps
            hidden_dtype: Dtype for computation

        Returns:
            Tuple of:
                - scale_shift_values: [B, embedding_coefficient * embedding_dim]
                - embedded_timestep: [B, embedding_dim] - the raw timestep embedding
        """
        embedded_timestep = self.emb(timestep, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


class PixArtAlphaTextProjection(nn.Module):
    """
    Projects caption embeddings for cross-attention.

    Takes text embeddings from the text encoder (e.g., Gemma3 with 3840 dim)
    and projects them to the transformer's hidden dimension (4096).
    Also handles dropout for classifier-free guidance during training.

    Adapted from PixArt-Alpha: https://github.com/PixArt-alpha/PixArt-alpha

    Args:
        in_features: Input dimension (text encoder output, e.g., 3840 for Gemma3)
        hidden_size: Hidden/output dimension (transformer dim, e.g., 4096)
        out_features: Optional different output dimension
        act_fn: Activation function ("gelu_tanh" or "silu")
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: Optional[int] = None,
        act_fn: str = "gelu_tanh"
    ):
        super().__init__()
        if out_features is None:
            out_features = hidden_size

        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)

        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")

        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        """
        Project caption embeddings.

        Args:
            caption: Text embeddings [B, seq_len, in_features]

        Returns:
            Projected embeddings [B, seq_len, out_features]
        """
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


def rms_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Root-mean-square (RMS) normalize tensor over its last dimension.

    RMSNorm is used in LTX-2 instead of LayerNorm for attention Q/K normalization.
    It's computationally simpler as it doesn't require mean subtraction.

    Args:
        x: Input tensor
        weight: Optional learnable weight parameter
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor with same shape as input
    """
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)


@dataclass(frozen=True)
class Modality:
    """
    Input data for a single modality (video or audio) in the transformer.

    Bundles the latent tokens, timestep embeddings, positional information,
    and text conditioning context for processing by the diffusion transformer.

    Attributes:
        latent: [B, T, D] where B=batch, T=tokens, D=input dimension (128 for LTX-2)
        timesteps: [B, T] diffusion timesteps (can be per-token for IC-LoRA)
        positions: [B, 3, T] position indices for video (temporal, height, width)
        context: [B, seq_len, context_dim] text conditioning embeddings
        enabled: Whether this modality is active
        context_mask: Optional attention mask for text [B, seq_len]
    """
    latent: torch.Tensor
    timesteps: torch.Tensor
    positions: torch.Tensor
    context: torch.Tensor
    enabled: bool = True
    context_mask: Optional[torch.Tensor] = None
