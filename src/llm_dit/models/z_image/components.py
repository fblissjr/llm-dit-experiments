"""
Z-Image DiT building block components.

Last updated: 2026-02-01

Implements the core components of the Z-Image S3-DiT architecture:
- RMSNorm: Root Mean Square Layer Normalization (from llm_dit.layers)
- FeedForward: SiLU-gated FFN (SwiGLU variant)
- TimestepEmbedder: Sinusoidal timestep embedding with MLP
- FinalLayer: Output layer with adaptive modulation

Based on DiffSynth-Studio implementation.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_dit.layers import RMSNorm
from .constants import ADALN_EMBED_DIM

# Note: Z-Image uses RMSNorm with eps=1e-5 (different from default 1e-6).
# All usages in this module explicitly pass eps=1e-5 or use norm_eps parameter.


class FeedForward(nn.Module):
    """
    SiLU-gated Feed-Forward Network (SwiGLU variant).

    Uses two parallel linear projections with gating:
        output = w2(SiLU(w1(x)) * w3(x))

    This is the SwiGLU architecture from PaLM, which provides
    better performance than standard FFN.

    Args:
        dim: Input/output dimension
        hidden_dim: Hidden dimension (typically 8/3 * dim)
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: SiLU(w1(x)) * w3(x), then project down
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TimestepEmbedder(nn.Module):
    """
    Timestep embedding using sinusoidal encoding + MLP.

    Converts scalar timesteps to embeddings using:
    1. Sinusoidal frequency encoding (like positional encoding)
    2. Two-layer MLP with SiLU activation

    Args:
        out_size: Output embedding dimension
        mid_size: MLP hidden dimension (default: out_size)
        frequency_embedding_size: Size of sinusoidal encoding (default: 256)
    """

    def __init__(
        self,
        out_size: int,
        mid_size: Optional[int] = None,
        frequency_embedding_size: int = 256,
    ):
        super().__init__()
        if mid_size is None:
            mid_size = out_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor,
        dim: int,
        max_period: float = 10000.0,
    ) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: Timestep tensor of shape (batch,)
            dim: Embedding dimension
            max_period: Maximum period for frequencies

        Returns:
            Embedding tensor of shape (batch, dim)
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # Handle odd dimensions
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Embed timesteps.

        Args:
            t: Timestep tensor of shape (batch,) in range [0, 1000]

        Returns:
            Embedding tensor of shape (batch, out_size)
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(torch.bfloat16))
        return t_emb


class FinalLayer(nn.Module):
    """
    Final output layer with adaptive layer normalization.

    Applies adaptive scaling based on timestep conditioning
    before the final linear projection.

    Args:
        hidden_size: Input hidden dimension
        out_channels: Output channels (patch_size^2 * in_channels)
    """

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        noise_mask: Optional[torch.Tensor] = None,
        c_noisy: Optional[torch.Tensor] = None,
        c_clean: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply final layer with adaptive modulation.

        Args:
            x: Input tensor (batch, seq_len, hidden)
            c: Global conditioning (batch, embed_dim) - for basic mode
            noise_mask: Per-token noise indicator (batch, seq_len) - for omni mode
            c_noisy: Noisy token conditioning - for omni mode
            c_clean: Clean token conditioning - for omni mode

        Returns:
            Output tensor (batch, seq_len, out_channels)
        """
        seq_len = x.shape[1]

        if noise_mask is not None:
            # Per-token modulation (Omni mode)
            scale_noisy = 1.0 + self.adaLN_modulation(c_noisy)
            scale_clean = 1.0 + self.adaLN_modulation(c_clean)
            scale = select_per_token(scale_noisy, scale_clean, noise_mask, seq_len)
        else:
            # Global modulation (basic mode)
            assert c is not None, "Either c or (c_noisy, c_clean) must be provided"
            scale = 1.0 + self.adaLN_modulation(c)
            scale = scale.unsqueeze(1)

        x = self.norm_final(x) * scale
        x = self.linear(x)
        return x


def select_per_token(
    value_noisy: torch.Tensor,
    value_clean: torch.Tensor,
    noise_mask: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """
    Select between noisy and clean values based on per-token mask.

    Used in Omni mode for per-token adaptive modulation.

    Args:
        value_noisy: Values for noisy tokens (batch, dim)
        value_clean: Values for clean tokens (batch, dim)
        noise_mask: Binary mask (batch, seq_len) where 1=noisy, 0=clean
        seq_len: Sequence length

    Returns:
        Selected values (batch, seq_len, dim)
    """
    noise_mask_expanded = noise_mask.unsqueeze(-1)  # (batch, seq_len, 1)
    return torch.where(
        noise_mask_expanded == 1,
        value_noisy.unsqueeze(1).expand(-1, seq_len, -1),
        value_clean.unsqueeze(1).expand(-1, seq_len, -1),
    )
