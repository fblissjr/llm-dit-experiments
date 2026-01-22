"""
Residual block for latent upsampler.

Last Updated: 2026-01-18

ResBlock implements a standard residual connection with GroupNorm and SiLU
activation. Supports both 2D (Conv2d) and 3D (Conv3d) convolutions.
"""

from typing import Optional

import torch


class ResBlock(torch.nn.Module):
    """
    Residual block with two convolutional layers, group normalization, and SiLU activation.

    Architecture:
        x -> Conv -> GroupNorm(32) -> SiLU -> Conv -> GroupNorm(32) -> Add(x) -> SiLU

    Args:
        channels: Number of input and output channels.
        mid_channels: Number of channels in the intermediate convolution layer.
            Defaults to `channels` if not specified.
        dims: Dimensionality of the convolution (2 for Conv2d, 3 for Conv3d).
            Defaults to 3.
    """

    def __init__(self, channels: int, mid_channels: Optional[int] = None, dims: int = 3):
        super().__init__()
        if mid_channels is None:
            mid_channels = channels

        conv = torch.nn.Conv2d if dims == 2 else torch.nn.Conv3d

        self.conv1 = conv(channels, mid_channels, kernel_size=3, padding=1)
        self.norm1 = torch.nn.GroupNorm(32, mid_channels)
        self.conv2 = conv(mid_channels, channels, kernel_size=3, padding=1)
        self.norm2 = torch.nn.GroupNorm(32, channels)
        self.activation = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x + residual)
        return x
