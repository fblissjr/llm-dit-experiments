"""
LTX-2 VAE Operations.

Last Updated: 2026-01-18

Core operations for video VAE: patchify/unpatchify and per-channel statistics.

Ported from: ltx_core.model.video_vae.ops
Original source: https://github.com/Lightricks/LTX-2
License: LTX-2 Community License
Copyright (c) 2025 Lightricks Ltd.
"""

import torch
from einops import rearrange
from torch import nn


def patchify(x: torch.Tensor, patch_size_hw: int, patch_size_t: int = 1) -> torch.Tensor:
    """
    Rearrange spatial dimensions into channels (space-to-depth).

    Divides image into patch_size x patch_size blocks and moves pixels from
    each block into separate channels. This trades spatial resolution for
    channel depth, making subsequent convolutions more efficient.

    Args:
        x: Input tensor (4D or 5D)
        patch_size_hw: Spatial patch size for height and width.
            With patch_size_hw=4, divides HxW into 4x4 blocks.
        patch_size_t: Temporal patch size for frames. Default=1 (no temporal patching).

    Returns:
        For 5D: (B, C, F, H, W) -> (B, C*patch_hw^2*patch_t, F/patch_t, H/patch_hw, W/patch_hw)

    Example:
        (B, 3, 33, 512, 512) with patch_size_hw=4, patch_size_t=1 -> (B, 48, 33, 128, 128)
    """
    if patch_size_hw == 1 and patch_size_t == 1:
        return x

    if x.dim() == 4:
        x = rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size_hw, r=patch_size_hw)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b c (f p) (h q) (w r) -> b (c p r q) f h w",
            p=patch_size_t,
            q=patch_size_hw,
            r=patch_size_hw,
        )
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")

    return x


def unpatchify(x: torch.Tensor, patch_size_hw: int, patch_size_t: int = 1) -> torch.Tensor:
    """
    Rearrange channels back into spatial dimensions (depth-to-space).

    Inverse of patchify - moves pixels from channels back into patch_size x patch_size
    blocks. This restores spatial resolution from channel depth.

    Args:
        x: Input tensor (4D or 5D)
        patch_size_hw: Spatial patch size for height and width.
            With patch_size_hw=4, expands HxW by 4x.
        patch_size_t: Temporal patch size for frames. Default=1 (no temporal expansion).

    Returns:
        For 5D: (B, C*patch^2*patch_t, F, H, W) -> (B, C, F*patch_t, H*patch_hw, W*patch_hw)

    Example:
        (B, 48, 33, 128, 128) with patch_size_hw=4, patch_size_t=1 -> (B, 3, 33, 512, 512)
    """
    if patch_size_hw == 1 and patch_size_t == 1:
        return x

    if x.dim() == 4:
        x = rearrange(x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size_hw, r=patch_size_hw)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b (c p r q) f h w -> b c (f p) (h q) (w r)",
            p=patch_size_t,
            q=patch_size_hw,
            r=patch_size_hw,
        )

    return x


class PerChannelStatistics(nn.Module):
    """
    Per-channel statistics for normalizing and denormalizing the latent representation.

    This statistics is computed over the entire dataset and stored in the model's
    checkpoint under VAE state_dict. It enables the VAE to output normalized latents
    with zero mean and unit variance per channel, which improves training stability.

    The normalization formula:
        normalized = (x - mean_of_means) / std_of_means

    The denormalization formula:
        original = normalized * std_of_means + mean_of_means
    """

    def __init__(self, latent_channels: int = 128):
        """
        Args:
            latent_channels: Number of latent channels (default 128 for LTX-2).
        """
        super().__init__()
        # Register buffers - these will be loaded from checkpoint
        self.register_buffer("std-of-means", torch.empty(latent_channels))
        self.register_buffer("mean-of-means", torch.empty(latent_channels))
        self.register_buffer("mean-of-stds", torch.empty(latent_channels))
        self.register_buffer("mean-of-stds_over_std-of-means", torch.empty(latent_channels))
        self.register_buffer("channel", torch.empty(latent_channels))

    def un_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize latents back to original scale.

        Args:
            x: Normalized latents (B, C, F, H, W)

        Returns:
            Denormalized latents
        """
        std = self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(x)
        mean = self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(x)
        return x * std + mean

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize latents to zero mean and unit variance per channel.

        Args:
            x: Raw latents (B, C, F, H, W)

        Returns:
            Normalized latents
        """
        std = self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(x)
        mean = self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(x)
        return (x - mean) / std
