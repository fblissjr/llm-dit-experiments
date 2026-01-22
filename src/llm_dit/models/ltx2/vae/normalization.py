"""
LTX-2 VAE Normalization Layers.

Last Updated: 2026-01-18

Normalization layers used in the Video VAE.

Ported from: ltx_core.model.common.normalization
Original source: https://github.com/Lightricks/LTX-2
License: LTX-2 Community License
Copyright (c) 2025 Lightricks Ltd.
"""

from enum import Enum

import torch
from torch import nn


class NormType(Enum):
    """Normalization layer types: GROUP (GroupNorm) or PIXEL (per-location RMS norm)."""
    GROUP = "group"
    PIXEL = "pixel"


class PixelNorm(nn.Module):
    """
    Per-pixel (per-location) RMS normalization layer.

    For each element along the chosen dimension, this layer normalizes the tensor
    by the root-mean-square of its values across that dimension:
        y = x / sqrt(mean(x^2, dim=dim, keepdim=True) + eps)

    This is useful for normalizing feature maps where you want each spatial
    location to have unit norm across channels.
    """

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        """
        Args:
            dim: Dimension along which to compute the RMS (typically channels).
            eps: Small constant added for numerical stability.
        """
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization along the configured dimension."""
        mean_sq = torch.mean(x**2, dim=self.dim, keepdim=True)
        rms = torch.sqrt(mean_sq + self.eps)
        return x / rms


def build_normalization_layer(
    in_channels: int,
    *,
    num_groups: int = 32,
    normtype: NormType = NormType.GROUP,
) -> nn.Module:
    """
    Create a normalization layer based on the normalization type.

    Args:
        in_channels: Number of input channels
        num_groups: Number of groups for group normalization
        normtype: Type of normalization: "group" or "pixel"

    Returns:
        A normalization layer
    """
    if normtype == NormType.GROUP:
        return nn.GroupNorm(
            num_groups=num_groups,
            num_channels=in_channels,
            eps=1e-6,
            affine=True,
        )
    if normtype == NormType.PIXEL:
        return PixelNorm(dim=1, eps=1e-6)
    raise ValueError(f"Invalid normalization type: {normtype}")
