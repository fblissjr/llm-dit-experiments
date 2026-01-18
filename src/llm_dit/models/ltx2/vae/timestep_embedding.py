"""
LTX-2 VAE Timestep Embedding.

Last Updated: 2026-01-18

Timestep embedding modules for conditioning the VAE decoder.

Ported from: ltx_core.model.transformer.timestep_embedding
Original source: https://github.com/Lightricks/LTX-2
License: LTX-2 Community License
Copyright (c) 2025 Lightricks Ltd.
"""

import math

import torch
from torch import nn


class Timesteps(nn.Module):
    """
    Sinusoidal timestep embeddings.

    Creates positional embeddings for timesteps using sine and cosine functions
    at different frequencies, similar to the positional encoding in transformers.
    """

    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool = False,
        downscale_freq_shift: float = 1.0,
    ):
        """
        Args:
            num_channels: Number of channels in the embedding (half for sin, half for cos).
            flip_sin_to_cos: If True, concatenate cos before sin.
            downscale_freq_shift: Shift applied to frequencies.
        """
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal embeddings for timesteps.

        Args:
            timesteps: 1D tensor of timestep values.

        Returns:
            Embeddings of shape (len(timesteps), num_channels).
        """
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - self.downscale_freq_shift)

        emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.flip_sin_to_cos:
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

        if self.num_channels % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

        return emb


class TimestepEmbedding(nn.Module):
    """
    MLP to project timestep embeddings to a higher dimension.
    """

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
    ):
        """
        Args:
            in_channels: Input dimension (from Timesteps).
            time_embed_dim: Output dimension.
            act_fn: Activation function ("silu" or "gelu").
        """
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)

        if act_fn == "silu":
            self.act = nn.SiLU()
        elif act_fn == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {act_fn}")

        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Project timestep embeddings.

        Args:
            sample: Timestep embeddings from Timesteps module.

        Returns:
            Projected embeddings.
        """
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    """
    Combined timestep embeddings for PixArt-Alpha style models.

    Used in the VAE decoder for timestep conditioning.

    Reference:
    https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L164C9-L168C29
    """

    def __init__(
        self,
        embedding_dim: int,
        size_emb_dim: int,
    ):
        """
        Args:
            embedding_dim: Output embedding dimension.
            size_emb_dim: Size embedding dimension (0 to disable).
        """
        super().__init__()

        self.outdim = size_emb_dim
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Create combined timestep embeddings.

        Args:
            timestep: Timestep values (B,).
            hidden_dtype: Data type for the output.

        Returns:
            Timestep embeddings (B, embedding_dim).
        """
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))
        return timesteps_emb
